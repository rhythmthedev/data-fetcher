"""
fetch_oi_history.py
====================
Downloads and stores Binance Open Interest (5m) history to disk.
- NEVER deletes existing data — only adds new data.
- Handles coins that didn't exist yet (starts from their first available candle).
- Run this first, then run backtest_from_cache.py.

USAGE:
    python fetch_oi_history.py

After the first run, to expand the date range just change START / END below
and re-run. It will only fetch what's missing.
"""

import asyncio
import aiohttp
import pandas as pd
import os
import time
from datetime import datetime, timezone

# ============================================================
# USER CONFIG
# ============================================================
BASE = "https://fapi.binance.com"

# ⚠️ Use UTC dates.  Change freely — existing data is NEVER deleted.
START = datetime(2026, 4, 15, 0, 0, 0, tzinfo=timezone.utc)   # earliest date you want
END   = datetime(2026, 5, 21, 23, 59, 59, tzinfo=timezone.utc) # latest  date you want

DATA_DIR      = "oi_cache"          # folder where parquet files are stored
MAX_CONCURRENT = 5                  # concurrent symbol fetches (be gentle on Binance)
BATCH_DELAY    = 0.2                # seconds between symbol batches
LIMIT_PER_CALL = 500                # max candles per API call (Binance max = 500)
PERIOD         = "5m"               # OI period — must match your backtest
# ============================================================

sem = asyncio.Semaphore(MAX_CONCURRENT)
os.makedirs(DATA_DIR, exist_ok=True)


def ms(dt: datetime) -> int:
    """datetime → unix milliseconds."""
    return int(dt.timestamp() * 1000)


def parquet_path(symbol: str) -> str:
    return os.path.join(DATA_DIR, f"{symbol}.parquet")


def load_existing(symbol: str) -> pd.DataFrame:
    """Load cached parquet for a symbol, or return empty DataFrame."""
    path = parquet_path(symbol)
    if os.path.exists(path):
        return pd.read_parquet(path)
    return pd.DataFrame()


def save(symbol: str, df: pd.DataFrame):
    """Save DataFrame to parquet (sorted, deduped)."""
    df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
    df.to_parquet(parquet_path(symbol), index=False)


async def get_symbols(session) -> list[str]:
    async with session.get(f"{BASE}/fapi/v1/exchangeInfo") as resp:
        data = await resp.json()
    return [
        s["symbol"] for s in data["symbols"]
        if s["contractType"] in ("PERPETUAL", "TRADIFI_PERPETUAL")
        and s["status"] == "TRADING"
        and s["quoteAsset"] == "USDT"
    ]


async def fetch_oi_range(session, symbol: str, start_ms: int, end_ms: int) -> list[dict]:
    """
    Fetch OI candles for [start_ms, end_ms] in paginated chunks.
    Returns list of raw dicts from Binance.
    """
    all_rows = []
    cursor = start_ms

    while cursor < end_ms:
        async with sem:
            try:
                async with session.get(
                    f"{BASE}/futures/data/openInterestHist",
                    params={
                        "symbol": symbol,
                        "period": PERIOD,
                        "limit": LIMIT_PER_CALL,
                        "startTime": cursor,
                        "endTime": end_ms,
                    },
                    timeout=aiohttp.ClientTimeout(total=20),
                ) as resp:
                    data = await resp.json()
            except Exception as e:
                print(f"  ✗ {symbol} fetch error: {e}")
                break

        if not data or isinstance(data, dict):   # dict = error response
            break

        all_rows.extend(data)

        # Advance cursor past the last returned candle
        last_ts = int(data[-1]["timestamp"])
        if len(data) < LIMIT_PER_CALL:
            break                               # no more pages
        cursor = last_ts + 1                    # next page starts here

        await asyncio.sleep(0.05)              # tiny pause between pages

    return all_rows


async def update_symbol(session, symbol: str, want_start_ms: int, want_end_ms: int):
    """
    Load existing data, figure out what's missing, fetch only that, merge & save.
    """
    existing = load_existing(symbol)

    fetch_start_ms = want_start_ms
    fetch_end_ms   = want_end_ms

    if not existing.empty:
        cached_min = int(existing["timestamp"].min())
        cached_max = int(existing["timestamp"].max())

        # Nothing new needed?
        if cached_min <= want_start_ms and cached_max >= want_end_ms:
            print(f"  ✓ {symbol:20s} — fully cached ({len(existing)} rows)")
            return

        # We may need to extend in both directions.
        # Collect gaps: [want_start → cached_min-1] and [cached_max+1 → want_end]
        new_rows = []

        if want_start_ms < cached_min:
            rows = await fetch_oi_range(session, symbol, want_start_ms, cached_min - 1)
            new_rows.extend(rows)

        if want_end_ms > cached_max:
            rows = await fetch_oi_range(session, symbol, cached_max + 1, want_end_ms)
            new_rows.extend(rows)

        if not new_rows:
            print(f"  ~ {symbol:20s} — no new data found")
            return

        new_df = _rows_to_df(new_rows)
        merged = pd.concat([existing, new_df], ignore_index=True)
        save(symbol, merged)
        print(f"  + {symbol:20s} — added {len(new_rows):4d} rows  (total {len(merged)})")

    else:
        # No cache yet: fetch the whole range
        rows = await fetch_oi_range(session, symbol, fetch_start_ms, fetch_end_ms)
        if not rows:
            print(f"  ✗ {symbol:20s} — no data returned (coin may not exist for this range)")
            return
        df = _rows_to_df(rows)
        save(symbol, df)
        print(f"  ✓ {symbol:20s} — saved {len(df):4d} rows")


def _rows_to_df(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df["timestamp"]       = df["timestamp"].astype("int64")
    df["sumOpenInterest"] = df["sumOpenInterest"].astype("float64")
    df["sumOpenInterestValue"] = df["sumOpenInterestValue"].astype("float64")
    return df[["timestamp", "sumOpenInterest", "sumOpenInterestValue", "symbol"]]


async def main():
    start_ms = ms(START)
    end_ms   = ms(END)

    connector = aiohttp.TCPConnector(limit=MAX_CONCURRENT)
    async with aiohttp.ClientSession(connector=connector) as session:
        symbols = await get_symbols(session)
        print(f"✓ {len(symbols)} USDT perpetual symbols found")
        print(f"📅 Range: {START.date()} → {END.date()}  |  Period: {PERIOD}")
        print(f"💾 Cache folder: {os.path.abspath(DATA_DIR)}\n")

        t0 = time.time()
        for i, symbol in enumerate(symbols, 1):
            print(f"[{i:3d}/{len(symbols)}] ", end="")
            await update_symbol(session, symbol, start_ms, end_ms)
            await asyncio.sleep(BATCH_DELAY)

        elapsed = time.time() - t0
        print(f"\n✅ Done in {elapsed/60:.1f} min.")
        print(f"   Run backtest_from_cache.py to analyse without API calls.")


if __name__ == "__main__":
    asyncio.run(main())
