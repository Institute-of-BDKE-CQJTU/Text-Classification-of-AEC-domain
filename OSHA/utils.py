# utils.py
import time
import datetime

def get_time_dif(start_time):

    end_time = time.time()
    time_dif = end_time - start_time
    return str(datetime.timedelta(seconds=int(round(time_dif))))
