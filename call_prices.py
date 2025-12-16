import pandas
from prices import *

df=pd.read_csv('hello.csv')


for x in df['TICKER']:


    TICKER = x
    PERIOD = "1y"
    START_DATE = "2025-01-05"  # format: YYYY-MM-DD
    END_DATE = "2025-05-12"    # format: YYYY-MM-DD
    INTERVAL = "1d"
    ATR_PERIOD = 10     # as chosen
    MULTIPLIER = 3   
    BOT_TOKEN = "8254287542:AAEoVHtqwTrSSWpL06Fn58_lRsBVoNw5DEQ"
    CHAT_ID = "-5037219263"

    # print(x)

    calculate(TICKER,PERIOD,START_DATE,END_DATE,INTERVAL,ATR_PERIOD,MULTIPLIER,BOT_TOKEN,CHAT_ID)
 