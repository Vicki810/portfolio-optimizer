"""Download prices from Yahoo Finance, store them in SQLite, and read them back with SQL."""
import sqlite3

import pandas as pd
import yfinance as yf

from config import DB_PATH


def download_prices(tickers, start, end):
    """Return a DataFrame of daily closing prices: rows = dates, columns = tickers."""
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    prices = raw["Close"]
    if isinstance(prices, pd.Series):  # happens if only one ticker is given
        prices = prices.to_frame(name=tickers[0])
    return prices.dropna(how="all")


def save_prices(prices, db_path=DB_PATH):
    """Store prices in a 'long' table: one row per (date, ticker, close)."""
    prices = prices.copy()
    prices.index.name = "date"
    long = prices.reset_index().melt(id_vars="date", var_name="ticker", value_name="close").dropna()
    long["date"] = pd.to_datetime(long["date"]).dt.strftime("%Y-%m-%d")

    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS prices (
                date   TEXT,
                ticker TEXT,
                close  REAL,
                PRIMARY KEY (date, ticker)
            )
            """
        )
        conn.executemany(
            "INSERT OR REPLACE INTO prices (date, ticker, close) VALUES (?, ?, ?)",
            long[["date", "ticker", "close"]].itertuples(index=False, name=None),
        )
    return len(long)


def load_prices(tickers, start, end, db_path=DB_PATH):
    """Query the database and return the same wide table shape as download_prices."""
    placeholders = ",".join("?" * len(tickers))
    query = f"""
        SELECT date, ticker, close
        FROM prices
        WHERE ticker IN ({placeholders})
          AND date BETWEEN ? AND ?
        ORDER BY date
    """
    with sqlite3.connect(db_path) as conn:
        long = pd.read_sql_query(query, conn, params=[*tickers, start, end], parse_dates=["date"])

    # Keep only dates where every ticker has a price, so returns line up.
    return long.pivot(index="date", columns="ticker", values="close")[tickers].dropna()
