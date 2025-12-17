import pandas
from prices import *

import pandas_market_calendars as mcal
from datetime import date
import sys

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
BOT_TOKEN = "8254287542:AAEoVHtqwTrSSWpL06Fn58_lRsBVoNw5DEQ"
CHAT_ID = "-5037219263"

for x in df['TICKER']:
    print(f'now calculating for {x}')
    calculate(x)
send_telegram_alert(BOT_TOKEN, CHAT_ID, f'Ran for {str(today)}')
 