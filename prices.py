import os
import yfinance as yf
import pandas as pd
import numpy as np
import mplfinance as mpf
import requests
from math import acos, degrees, sqrt
from datetime import date, timedelta
from dateutil.relativedelta import relativedelta


ATR_PERIOD = 10     # as chosen
MULTIPLIER = 3  

def fetch_prices(TICKER,start,end,interval):


    df = yf.download(TICKER, start, end, interval, auto_adjust=False)
    if df.empty:
        raise  ValueError(f"No data fetched for ticker: {TICKER}")
    # print(df)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]

    for col in ("Open", "High", "Low", "Close"):
        if col not in df.columns:
            raise SystemExit(f"Missing column in data: {col}")
    return df
    

def heikin_ashi(data):
    d = data.copy()
    ha = pd.DataFrame(index=d.index)

    ha["HA_Close"] = (d["Open"] + d["High"] + d["Low"] + d["Close"]) / 4

    ha_open = []
    for i in range(len(d)):
        if i == 0:
            ha_open.append((d["Open"].iat[0] + d["Close"].iat[0]) / 2)
        else:
            ha_open.append((ha_open[i-1] + ha["HA_Close"].iat[i-1]) / 2)
    ha["HA_Open"] = ha_open

    ha["HA_High"] = ha[["HA_Open", "HA_Close"]].join(d["High"]).max(axis=1)
    ha["HA_Low"] = ha[["HA_Open", "HA_Close"]].join(d["Low"]).min(axis=1)

    ha_out = ha.rename(columns={
        "HA_Open": "Open",
        "HA_High": "High",
        "HA_Low": "Low",
        "HA_Close": "Close"
    })[["Open", "High", "Low", "Close"]]
    # print(ha_out)
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



def plot_graph(st_df, ha,TICKER):
    st_series = st_df["SuperTrend"]
    dir_series = st_df["ST_dir"]

    apds = []

    st_up = st_series.where(dir_series == 1)
    if st_up.notna().any():
        apds.append(
            mpf.make_addplot(st_up, type="line", width=1.3, color="green")
        )

    st_down = st_series.where(dir_series == -1)
    if st_down.notna().any():
        apds.append(
            mpf.make_addplot(st_down, type="line", width=1.3, color="red")
        )

    mc = mpf.make_marketcolors(
        up='white',
        down='black',
        edge='inherit',
        wick='inherit'
    )
    s = mpf.make_mpf_style(
        base_mpf_style='classic',
        marketcolors=mc
    )

    mpf.plot(
        ha,
        type="candle",
        style=s,
        addplot=apds if apds else None,  # safe guard
        figsize=(14, 8),
        ylabel="Price",
        title=f"Heikin-Ashi Candles + SuperTrend ({TICKER})",
        savefig=OUTFILE,
        tight_layout=True
    )

    # print("Chart saved to:", OUTFILE)





def add_supertrend_angle(st_df):
    """
    Adds a column 'ST_angle_deg' representing the geometric angle
    between SuperTrend segments:

    Points:
    day before yesterday -> (x-1, y1)
    yesterday            -> (x,   y)   [vertex]
    today                -> (x+1, y2)

    Angle is computed ONLY if:
    - ST_dir[t] == 1
    - At least 3 rows exist
    """

    st_df = st_df.copy()
    st_df["ST_angle_deg"] = np.nan

    for i in range(2, len(st_df)):

        # today must be bullish
        if st_df["ST_dir"].iat[i] != 1:
            continue

        y2 = st_df["SuperTrend"].iat[i]     # today
        y  = st_df["SuperTrend"].iat[i - 1] # yesterday (vertex)
        y1 = st_df["SuperTrend"].iat[i - 2] # day before

        # vectors
        vAx, vAy = 1,  y2 - y
        vBx, vBy = -1, y1 - y

        # dot product
        dot = vAx * vBx + vAy * vBy

        # magnitudes
        magA = sqrt(vAx**2 + vAy**2)
        magB = sqrt(vBx**2 + vBy**2)

        if magA == 0 or magB == 0:
            continue

        cos_theta = dot / (magA * magB)

        # numerical safety
        cos_theta = max(-1.0, min(1.0, cos_theta))

        angle_deg = degrees(acos(cos_theta))

        st_df["ST_angle_deg"].iat[i] = angle_deg

    # print(st_df)
    # st_df.to_csv(f"{SAVE_DIR}/angle.csv")

    return st_df



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





def has_green_st_small_angle(st_df, target_date):
    """
    Returns True if target_date satisfies:
    - ST_dir == 1 (green SuperTrend)
    - 0 < ST_angle_deg < 30
    Else returns False
    """

    df = st_df.copy()
    df.index = pd.to_datetime(df.index)

    target_date = pd.to_datetime(target_date).normalize()

    if target_date not in df.index.normalize():
        return False

    row = df.loc[df.index.normalize() == target_date].iloc[0]

    return (
        row["ST_dir"] == 1 and
        pd.notna(row["ST_angle_deg"]) and
        0 < row["ST_angle_deg"] < 30
    )



def calculate(TICKER,BOT_TOKEN,CHAT_ID):
    today = date.today()
    # today = date(2025, 3, 20)
    two_months_ago = today - relativedelta(months=2)
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

        st_df = add_supertrend_angle(st_df)

        flag = has_green_st_small_angle(st_df, today)

        if  flag==False:
            return   # graceful exit, no signal

        message = TICKER+' '+  str(today)
        # print(st_df)
        send_telegram_alert(BOT_TOKEN, CHAT_ID, message)

        return 

    except Exception as e:
        send_telegram_alert(BOT_TOKEN, CHAT_ID, TICKER+' '+ str(e))
        return 

if __name__ == "__main__":
    calculate()
