from datetime import date
from datetime import timedelta


if __name__ =='__main__':
    start_date = date(2011, 12, 31)
    end_date = date(2012, 10, 1)
    print(start_date+timedelta(10))
    print(end_date-start_date)