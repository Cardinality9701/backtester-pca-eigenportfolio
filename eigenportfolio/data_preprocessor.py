import polars as pl
import numpy as np
import yfinance as yf
import yaml
from typing import List, Tuple


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# ── Data loading ────────────────────────────────────────────────────────────────

def download_prices(tickers: List[str], start_date: str, end_date: str) -> pl.DataFrame:
    """Download split/dividend-adjusted close prices from Yahoo Finance."""
    raw = yf.download(tickers, start=start_date, end=end_date, auto_adjust=True, progress=False)
    prices_pd = raw["Close"][tickers].reset_index()
    df = pl.from_pandas(prices_pd)
    return df.rename({"Date": "date"})


def forward_fill_and_align(df: pl.DataFrame, tickers: List[str]) -> pl.DataFrame:
    """
    Forward-fill prices across exchange closures and public holidays.
    Treats a gap as a zero-return day (industry standard for liquid ETFs).
    Drops leading rows where any ticker has no valid price yet.
    """
    df = df.sort("date")
    valid_tickers = [t for t in tickers if t in df.columns]
    df = df.with_columns([pl.col(t).forward_fill() for t in valid_tickers])
    return df.drop_nulls(subset=valid_tickers)


def compute_log_returns(df: pl.DataFrame, tickers: List[str]) -> pl.DataFrame:
    """Compute log(P_t / P_{t-1}) for each ticker; append as <ticker>_ret columns."""
    valid_tickers = [t for t in tickers if t in df.columns]
    df = df.with_columns([
        (pl.col(t).log() - pl.col(t).shift(1).log()).alias(f"{t}_ret")
        for t in valid_tickers
    ])
    return df.drop_nulls(subset=[f"{t}_ret" for t in valid_tickers])


# ── NEW: VIX addition ─────────────────────────────────────────────────────────────

def _cast_date_col(df: pl.DataFrame) -> pl.DataFrame:
    """
    Normalise the 'date' column to pl.Date.

    yfinance may return DatetimeIndex with or without UTC timezone depending
    on the installed version.  Polars preserves that ambiguity, which causes
    a type mismatch when joining the sector ETF and VIX DataFrames.
    Casting both to pl.Date before the join eliminates this fragility.
    """
    dtype = df["date"].dtype
    if dtype == pl.Date:
        return df
    if dtype in (pl.Datetime, pl.Datetime("us"), pl.Datetime("ns"),
                 pl.Datetime("us", "UTC"), pl.Datetime("ns", "UTC")):
        return df.with_columns(pl.col("date").dt.date())
    return df.with_columns(pl.col("date").cast(pl.Date))


def download_vix(ticker: str, start_date: str, end_date: str) -> pl.DataFrame:
    """
    Download VIX index levels from Yahoo Finance.

    VIX is downloaded separately from the sector ETFs for two reasons:
      1. VIS is an index, not tradeable asset
      2. predictive_analysis.py uses VIX LEVELS (not log returns) as the
         dependent variable in the forward-predictive regressions.

    Forward-fills any gaps (VIX trades on US market days; gaps are rare
    and represent data vendor quirks rather than genuine missing closes).

    Returns
    -------
    pl.DataFrame with columns [date (pl.Date), VIX (f64)]
    """
    raw = yf.download(ticker, start=start_date, end=end_date,
                      auto_adjust=True, progress=False)
    # Single-ticker download always returns simple (non-MultiIndex) columns.
    vix_pd = raw[["Close"]].reset_index()
    vix_pd.columns = ["date", "VIX"]

    df = pl.from_pandas(vix_pd)
    df = _cast_date_col(df)
    df = df.sort("date").with_columns(pl.col("VIX").forward_fill())
    return df.drop_nulls(subset=["VIX"])


# ── MODIFIED: updated signature and body ─────────────────────────────────────

def preprocess(
    config_path: str = "config.yaml",
) -> Tuple[pl.DataFrame, np.ndarray, List[str], np.ndarray]:
    """
    Full preprocessing pipeline.

    Returns
    -------
    df             : polars DataFrame [date, <ticker>..., <ticker>_ret..., VIX]
    returns_matrix : np.ndarray (T, N)  log returns, sector ETFs only
    tickers        : ordered list of sector ETF ticker strings
    vix_series     : np.ndarray (T,)    VIX levels, date-aligned to returns_matrix

    Notes
    -----
    vix_series contains *levels* (not log changes).  predictive_analysis.py
    computes VIX[t+h] internally by shifting this array; no look-ahead is
    introduced here because the shift happens at analysis time, not here.

    The VIX column is also appended to df so all data lives in one place
    for downstream CSV persistence and debugging.
    """
    cfg         = load_config(config_path)
    tickers     = cfg["data"]["tickers"]
    vix_ticker  = cfg.get("vix", {}).get("ticker", "^VIX")

    # ── Sector ETF pipeline (unchanged logic) ────────────────────────────────
    df = download_prices(tickers, cfg["data"]["start"], cfg["data"]["end"])
    df = forward_fill_and_align(df, tickers)
    df = compute_log_returns(df, tickers)
    df = _cast_date_col(df)                    # normalise before join

    # ── VIX pipeline ─────────────────────────────────────────────────────────
    df_vix = download_vix(vix_ticker, cfg["data"]["start"], cfg["data"]["end"])

    # Left join: keep every sector ETF trading day; fill any residual VIX gaps.
    # A left join is correct here — we never want to lose a return row because
    # VIX data happened to be missing for that date.
    df = df.join(df_vix, on="date", how="left")
    df = df.with_columns(pl.col("VIX").forward_fill())
    df = df.drop_nulls(subset=["VIX"])         # drops leading rows before VIX history

    # ── Extract arrays ────────────────────────────────────────────────────────
    returns_matrix = df.select([f"{t}_ret" for t in tickers]).to_numpy()
    vix_series     = df["VIX"].to_numpy()

    return df, returns_matrix, tickers, vix_series

# ----- Train/val/test splits + helper fcns for dates ----------
def get_split_masks(dates: List, splits_cfg: dict) -> dict:
    """
    Return boolean masks for train / val / test splits.

    Parameters
    ----------
    dates      : full date list from df["date"].to_list()
    splits_cfg : cfg["splits"] — keys: train_end, val_end

    Returns
    -------
    dict with keys "train", "val", "test" — each a np.ndarray of bool,
    length = len(dates), True where the date falls in that split.

    # CORRECT USAGE — call on panel dates, not df dates
    panel = build_aligned_panel(dates, ep_results, monitor_results, hl=primary_hl)
    masks = get_split_masks(panel["date"].to_list(), cfg["splits"])

    panel_train = panel.filter(masks["train"])
    panel_val   = panel.filter(masks["val"])
    panel_test  = panel.filter(masks["test"])
    """
    train_end = pl.Series([splits_cfg["train_end"]]).cast(pl.Date)[0]
    val_end   = pl.Series([splits_cfg["val_end"]]).cast(pl.Date)[0]

    dates_pl  = pl.Series([str(d)[:10] for d in dates]).cast(pl.Date)

    return {
        "train": (dates_pl <= train_end).to_numpy(),
        "val":   ((dates_pl > train_end) & (dates_pl <= val_end)).to_numpy(),
        "test":  (dates_pl > val_end).to_numpy(),
    }

def build_aligned_panel(
    dates: List,
    ep_results: dict,
    monitor_results: dict,
    hl: int,
    warmup: int,
):
    """
    Align ep_returns and delta_ar onto a common date index for a given hl.

    ep_returns starts at dates[warmup + 1].
    delta_ar   starts at dates[warmup] but has leading NaNs from long_window.
    Inner join on date eliminates all alignment arithmetic from Phase 4 files.

    Returns
    -------
    pl.DataFrame with columns:
        date, pc1_ret, pc2_ret, ... pcK_ret, delta_ar

    All rows with NaN in any column are dropped — the usable window
    starts when delta_ar's long_window (252d) is fully populated.
    """
    ep     = ep_results[hl]
    offset = ep["dates_offset"]
    T_eff  = ep["ep_returns"].shape[0]
    K      = ep["ep_returns"].shape[1]

    ep_dates = [str(d)[:10] for d in dates[offset : offset + T_eff]]
    ep_dict  = {"date": ep_dates}
    for k in range(K):
        ep_dict[f"pc{k+1}_ret"] = ep["ep_returns"][:, k].tolist()

    cfg = load_config
    dar       = monitor_results[hl]["delta_ar"]
    dar_dates = [str(d)[:10] for d in dates[warmup : warmup + len(dar)]]
    dar_dict  = {"date": dar_dates, "delta_ar": dar.tolist()}

    df_ep  = pl.DataFrame(ep_dict)
    df_dar = pl.DataFrame(dar_dict)

    panel = df_ep.join(df_dar, on="date", how="inner")
    return panel.drop_nulls()
