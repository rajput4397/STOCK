import pandas
from prices import *

df=pd.read_csv('hello.csv')


for x in df['TICKER']:
    print(f'now calculating for {x}')
    calculate(x)
 