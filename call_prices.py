import pandas
from prices import *

import pandas_market_calendars as mcal
from datetime import date
import sys
import os
from dotenv import load_dotenv

load_dotenv()

BOT_TOKEN = os.getenv("BOT_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")

if not BOT_TOKEN or not CHAT_ID:
    raise RuntimeError("Missing Telegram credentials")

today = date.today()

def is_nse_trading_day(check_date: date) -> bool:
    """
    Returns True if NSE is open on the given date
    """
    nse = mcal.get_calendar("NSE")

    schedule = nse.schedule(
        start_date=check_date,
        end_date=check_date
    )

    return not schedule.empty

if not is_nse_trading_day(today):
    print("Market closed. Skipping.")
    sys.exit(0)


df=pd.read_csv('hello.csv')


for x in df['TICKER']:
    print(f'now calculating for {x}')
    calculate(x,BOT_TOKEN,CHAT_ID)
send_telegram_alert(BOT_TOKEN, CHAT_ID, f'Ran for {str(today)}')
 