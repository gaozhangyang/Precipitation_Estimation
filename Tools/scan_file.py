import os
from datetime import date, timedelta

files=os.listdir('/usr/commondata/weather/IR_data/IR_2012')
files=sorted(files)
year=int(files[0][7:11])
start_date=date(year-1,12,31)

day_s=int(files[0][12:15])
date_s=start_date+timedelta(day_s)

day_e=int(files[-1][12:15])
date_e=start_date+timedelta(day_e)

print(date_s)
print(date_e)