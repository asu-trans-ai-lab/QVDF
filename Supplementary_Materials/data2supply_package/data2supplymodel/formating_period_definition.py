# In[0] 
import pandas as pd
import os
import time
import datetime
import numpy as np
global daily_starting_time


def _obtain_time_interval(hhmm_time_interval):
    start_time = datetime.datetime.strptime(
        hhmm_time_interval.split('_')[0][:2] + ':' + hhmm_time_interval.split('_')[0][2:], '%H:%M')
    start_time_minute = start_time.hour * 60 + start_time.minute
    end_time = datetime.datetime.strptime(
        hhmm_time_interval.split('_')[1][:2] + ':' + hhmm_time_interval.split('_')[1][2:], '%H:%M')
    end_time_minute = end_time.hour * 60 + end_time.minute
    time_interval = end_time_minute - start_time_minute
    return time_interval


def _get_index(tt):
    time_index = np.mod(float(tt.split('_')[0][0:2]) * 60 + float(tt.split('_')[0][2:4]) - daily_starting_time, 1440)
    return time_index


def _hhmm_to_minutes(time_period):
    var = time_period.split('_')[0]
    var_minutes = float(var[0:2]) * 60 + float(var[2:4])
    return var_minutes


# In[5] Step 3: convert observations to lane performance files 
def define_demand_period(period_list, measurement_file='link_performance.csv', output_folder="./"):
    data_df = pd.read_csv(measurement_file, encoding='UTF-8')
    data_df['time_minutes'] = data_df.apply(lambda x: _hhmm_to_minutes(x.time_period), axis=1)
    TIME_INTERVAL_IN_MIN = _obtain_time_interval(data_df.time_period[0])

    print("---Join Demand Periods---")
    print('obtaining time_index ....')
    time_start = time.time()
    global daily_starting_time
    daily_starting_time = 1440
    period_name_list = []
    start_time_list = []
    end_time_list = []
    all_data_df = pd.DataFrame()
    for period_name in period_list:  # parser HHMM time, period length
        time_start = time.time()
        period_name_list.append(period_name)
        var = period_name.split('_')[0]
        start_time_list.append(int(var[0:2]) * 60 + int(var[2:4]))
        start_time = float(var[0:2]) * 60 + float(var[2:4])
        var = period_name.split('_')[1]
        end_time_list.append(float(var[0:2]) * 60 + float(var[2:4]))
        end_time = float(var[0:2]) * 60 + float(var[2:4])
        df = data_df[(data_df['time_minutes'] + TIME_INTERVAL_IN_MIN <= end_time) &
                     (data_df['time_minutes'] >= start_time)].copy()
        df['assignment_period'] = period_name
        all_data_df = pd.concat([all_data_df, df], sort=False)
        time_end = time.time()
        print('join assignment period:', period_name, ',using time', time_end - time_start, '...\n')

    daily_starting_time = min(start_time_list)
    time_start = time.time()
    print('start add time index using...\n')
    all_data_df.reset_index(drop=True, inplace=True)
    iter_time_period = all_data_df['time_period'].to_list()
    map_result = list(map(_get_index, iter_time_period))
    time_end = time.time()
    all_data_df['time_index'] = np.array(map_result)
    time_end = time.time()
    print('add time index using:', time_end - time_start, 's...DONE.\n')

    time_start = time.time()
    all_data_df.to_csv(os.path.join(output_folder, 'corridor_measurement.csv'), index=False)
    time_end = time.time()
    return all_data_df
    print('using time:', time_end - time_start, 's...\n')
