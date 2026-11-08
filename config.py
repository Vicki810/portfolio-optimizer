"""All the settings you might want to change, in one place."""

# Stocks/funds to analyze (Yahoo Finance symbols).
# Mix of tech, banks, healthcare, energy, consumer goods, gold, and bonds
# so the optimizer has genuinely different assets to combine.
TICKERS = ["AAPL", "MSFT", "JPM", "JNJ", "XOM", "PG", "GLD", "TLT"]

START_DATE = "2019-01-01"
END_DATE = "2026-10-31"

# Annual return of a "safe" investment (e.g. government T-bills), used in the Sharpe ratio.
RISK_FREE_RATE = 0.04

# No single asset may be more than this share of the portfolio (stops "put 100% in one stock").
MAX_WEIGHT = 0.40

DB_PATH = "portfolio.db"
OUTPUT_DIR = "outputs"
