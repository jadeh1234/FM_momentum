import pandas as pd
import numpy as np
import requests
from textblob import TextBlob
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import yfinance as yf

def backtest_sentiment_strategy(sentiment_df, price_dict, init_capital=100_000):
    """
    Backtest a simple one-day long/short sentiment strategy:
      - On each date: long the highest-sentiment ticker at OPEN, short the lowest at OPEN,
        close both at CLOSE, allocate 50% of capital to each leg, carry capital forward.
      - Returns one row per date with no NaNs (assuming yfinance had data for every ticker/date).
    """

    # 1) Normalize ticker names in sentiment_df to match price_dict keys:
    sent = sentiment_df.copy()
    # replace dots with dashes: e.g. 'BRK.B' → 'BRK-B'
    sent.columns = sent.columns.str.replace('.', '-', regex=False)
    sent.index   = pd.to_datetime(sent.index).normalize()

    # 2) Normalize price indices
    norm_price = {}
    for tkr, df in price_dict.items():
        df2 = df.copy()
        df2.index = pd.to_datetime(df2.index).normalize()
        norm_price[tkr] = df2[['Open','Close']]

    # 3) Run the backtest
    records = []
    portfolio_value = init_capital

    for date in sent.index:
        scores     = sent.loc[date]
        long_tkr   = scores.idxmax()
        short_tkr  = scores.idxmin()

        def get_prices(tkr):
            """Return (open, close) as floats or (None,None) if missing."""
            df = norm_price.get(tkr)
            if df is None or date not in df.index:
                return None, None
            row = df.loc[date]
            return float(row['Open']), float(row['Close'])

        o_l, c_l = get_prices(long_tkr)
        o_s, c_s = get_prices(short_tkr)

        # Now compute PnL — zero if we didn’t get prices
        alloc       = portfolio_value / 2
        pnl_long    = alloc / o_l * (c_l - o_l) if (o_l and c_l) else 0.0
        pnl_short   = alloc / o_s * (o_s - c_s) if (o_s and c_s) else 0.0
        daily_pnl   = pnl_long + pnl_short
        daily_ret   = daily_pnl / portfolio_value
        portfolio_value += daily_pnl

        records.append({
            'date':            date,
            'long':            long_tkr,
            'short':           short_tkr,
            'open_long':       o_l,
            'close_long':      c_l,
            'open_short':      o_s,
            'close_short':     c_s,
            'pnl_long':        pnl_long,
            'pnl_short':       pnl_short,
            'daily_pnl':       daily_pnl,
            'daily_return':    daily_ret,
            'portfolio_value': portfolio_value
        })

    return pd.DataFrame(records).set_index('date')