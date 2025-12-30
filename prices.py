import os
import yfinance as yf
import pandas as pd
import numpy as np
import mplfinance as mpf
import requests
from math import acos, degrees, atan, sqrt
from datetime import date, timedelta
from dateutil.relativedelta import relativedelta


def fetch_prices(TICKER, start, end, interval):
    df = yf.download(TICKER, start, end, interval, auto_adjust=False)

    if df.empty:
        raise ValueError(f"No data fetched for ticker: {TICKER}")

    # Handle MultiIndex columns (happens for some tickers/intervals)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]

    # Required columns including Volume
    required_cols = ("Open", "High", "Low", "Close", "Volume")
    for col in required_cols:
        if col not in df.columns:
            raise SystemExit(f"Missing column in data: {col}")

    return df


def heikin_ashi(data):
    d = data.copy()
    ha = pd.DataFrame(index=d.index)

    # Heikin Ashi Close
    ha["HA_Close"] = (d["Open"] + d["High"] + d["Low"] + d["Close"]) / 4

    # Heikin Ashi Open
    ha_open = []
    for i in range(len(d)):
        if i == 0:
            ha_open.append((d["Open"].iat[0] + d["Close"].iat[0]) / 2)
        else:
            ha_open.append((ha_open[i - 1] + ha["HA_Close"].iat[i - 1]) / 2)
    ha["HA_Open"] = ha_open

    # Heikin Ashi High / Low
    ha["HA_High"] = ha[["HA_Open", "HA_Close"]].join(d["High"]).max(axis=1)
    ha["HA_Low"] = ha[["HA_Open", "HA_Close"]].join(d["Low"]).min(axis=1)

    # Final output: HA OHLC + original Volume
    ha_out = ha.rename(columns={
        "HA_Open": "Open",
        "HA_High": "High",
        "HA_Low": "Low",
        "HA_Close": "Close"
    })[["Open", "High", "Low", "Close"]]

    # Preserve Volume (unchanged)
    ha_out["Volume"] = d["Volume"]

    return ha_out



def _true_range(data):
    prev_close = data["Close"].shift(1)
    tr1 = data["High"] - data["Low"]
    tr2 = (data["High"] - prev_close).abs()
    tr3 = (data["Low"] - prev_close).abs()
    return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

def rma(series, period):
    # Wilder's Moving Average (RMA)
    return series.ewm(alpha=1/period, adjust=False).mean()

def compute_supertrend(data, period=10, multiplier=3):
    df_local = data.copy()
    hl2 = (df_local["High"] + df_local["Low"]) / 2
    tr = _true_range(df_local)
    atr = rma(tr, period)

    basic_ub = hl2 + multiplier * atr
    basic_lb = hl2 - multiplier * atr

    final_ub = basic_ub.copy()
    final_lb = basic_lb.copy()

    # iterate to create final bands
    for i in range(1, len(df_local)):
        # final upper
        if (basic_ub.iat[i] < final_ub.iat[i-1]) or (df_local["Close"].iat[i-1] > final_ub.iat[i-1]):
            final_ub.iat[i] = basic_ub.iat[i]
        else:
            final_ub.iat[i] = final_ub.iat[i-1]
        # final lower
        if (basic_lb.iat[i] > final_lb.iat[i-1]) or (df_local["Close"].iat[i-1] < final_lb.iat[i-1]):
            final_lb.iat[i] = basic_lb.iat[i]
        else:
            final_lb.iat[i] = final_lb.iat[i-1]

    supertrend = pd.Series(index=df_local.index, dtype="float64")
    direction = pd.Series(index=df_local.index, dtype="int8")  # 1 bull, -1 bear

    # init
    direction.iat[0] = 1 if df_local["Close"].iat[0] > final_ub.iat[0] else -1
    supertrend.iat[0] = final_lb.iat[0] if direction.iat[0] == 1 else final_ub.iat[0]

    # print header
    # print(f"{'Date':<12} {'Close':>10} {'Final_UB':>12} {'Final_LB':>12} {'SuperTrend':>12} {'ST_dir':>6}")

    for i in range(1, len(df_local)):
        if df_local["Close"].iat[i] > final_ub.iat[i-1]:
            direction.iat[i] = 1
        elif df_local["Close"].iat[i] < final_lb.iat[i-1]:
            direction.iat[i] = -1
        else:
            direction.iat[i] = direction.iat[i-1]

        supertrend.iat[i] = final_lb.iat[i] if direction.iat[i] == 1 else final_ub.iat[i]


    df_local["ATR"] = atr
    df_local["Final_UB"] = final_ub
    df_local["Final_LB"] = final_lb
    df_local["SuperTrend"] = supertrend
    df_local["ST_dir"] = direction
    return df_local


def supertrend_momentum_filter(df_st, df_orig):
    """
    Conditions:
    1. SuperTrend green for >= 10 days
    2. Price change over last 5 days > +7%
    3. Volume condition:
        - If ST green < 20 days:
            avg vol (last 5 days) >
            1.6 * avg vol (20 days BEFORE ST turned green)
        - If ST green >= 20 days:
            ignore volume condition

    Returns:
        True / False
    """

    # --- Safety checks ---
    if len(df_st) < 30 or len(df_orig) < 30:
        return False

    # --- 1️⃣ Count consecutive green SuperTrend days ---
    st_dir = df_st["ST_dir"]

    green_days = 0
    for val in reversed(st_dir):
        if val == 1:
            green_days += 1
        else:
            break

    if green_days < 10:
        return False

    # --- 2️⃣ Price change over last 5 trading days ---
    close_now = df_orig["Close"].iloc[-1]
    close_5d_ago = df_orig["Close"].iloc[-6]

    price_change_pct = ((close_now - close_5d_ago) / close_5d_ago) * 100

    if price_change_pct <= 7:
        return False

    # --- 3️⃣ Volume condition ---
    if green_days < 20:
        # Find index where ST turned green
        st_turn_idx = len(st_dir) - green_days

        # Need at least 20 days BEFORE ST turned green
        if st_turn_idx < 20:
            return False

        avg_vol_last_5 = df_orig["Volume"].iloc[-5:].mean()
        avg_vol_pre_20 = df_orig["Volume"].iloc[st_turn_idx - 20 : st_turn_idx].mean()

        if avg_vol_last_5 <= 1.6 * avg_vol_pre_20:
            return False

    # --- All conditions satisfied ---
    return True



def send_telegram_alert(bot_token, chat_id, message):
    """
    Sends a Telegram message to a user or group
    """

    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"

    payload = {
        "chat_id": chat_id,
        "text": message,
        "parse_mode": "Markdown"
    }

    try:
        r = requests.post(url, data=payload, timeout=10)
        if r.status_code == 200:
            print("Telegram alert sent")
        else:
            print("Telegram alert failed:", r.text)

    except Exception as e:
        print("Telegram error:", e)


def calculate(TICKER,BOT_TOKEN,CHAT_ID):
    today = date.today()
    # today = date(2025, 12, 9)
    two_months_ago = today - relativedelta(months=6)
    tomorrow = str(today + timedelta(days=1))

    START_DATE= str(two_months_ago)
    END_DATE= str(tomorrow)
    TICKER = TICKER
    PERIOD = "1y"
    INTERVAL = "1d"
    ATR_PERIOD = 10     # as chosen
    MULTIPLIER = 3   
    BOT_TOKEN = BOT_TOKEN
    CHAT_ID = CHAT_ID






    # "853973272"
    try:
        df = fetch_prices(TICKER, START_DATE, END_DATE, INTERVAL)

        ha = heikin_ashi(df)

        st_df = compute_supertrend(
            ha,
            period=ATR_PERIOD,
            multiplier=MULTIPLIER
        )


        flag=supertrend_momentum_filter(st_df,df)

        if  flag==True:
               # graceful exit, no signal

            message=TICKER+' '+  str(today)
            send_telegram_alert(BOT_TOKEN, CHAT_ID, message)

        return 

    except Exception as e:
        send_telegram_alert(BOT_TOKEN, CHAT_ID, TICKER+' '+ str(e))
        return 

if __name__ == "__main__":
    calculate()


