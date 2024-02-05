import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt


def outlier_detector(speed, upper_bound, lower_bound):
    if (speed <= upper_bound) & (speed >= lower_bound):
        flag = 0
    else:
        flag = np.abs(speed - (upper_bound + lower_bound) / 2)
    return flag


def outlier_detector_1(index, outlier_list):
    if index in outlier_list:
        flag = 1
    else:
        flag = 0
    return flag


def _obtain_time_interval(hhmm_time_interval):
    start_time = datetime.datetime.strptime(
        hhmm_time_interval.split('_')[0][:2] + ':' + hhmm_time_interval.split('_')[0][2:], '%H:%M')
    start_time_minute = start_time.hour * 60 + start_time.minute
    end_time = datetime.datetime.strptime(
        hhmm_time_interval.split('_')[1][:2] + ':' + hhmm_time_interval.split('_')[1][2:], '%H:%M')
    end_time_minute = end_time.hour * 60 + end_time.minute
    time_interval = end_time_minute - start_time_minute
    return time_interval


def pivot_table(data_df, corridor_name):
    pivot_spd = pd.pivot_table(data_df, values='speed', columns=['time_period'],
                               index=['link_id', 'assignment_period', 'weekday'], aggfunc=np.mean)
    pivot_spd_std = pd.pivot_table(data_df, values='speed', columns=['time_period'],
                                   index=['link_id', 'assignment_period', 'weekday'], aggfunc=np.std)

    pivot_spd_ub = pivot_spd + pivot_spd_std
    pivot_spd_lb = pivot_spd - pivot_spd_std

    upper_bound_dict = {}
    lower_bound_dict = {}
    for ii in data_df.link_id.unique():
        for qq in data_df.assignment_period.unique():
            for jj in data_df.weekday.unique():
                for kk in data_df.time_period.unique():
                    up_value = pivot_spd_ub.loc[(ii, qq, jj)][kk]
                    low_value = pivot_spd_lb.loc[(ii, qq, jj)][kk]
                    upper_bound_dict[(ii, qq, jj, kk)] = up_value
                    lower_bound_dict[(ii, qq, jj, kk)] = low_value
    data_df["upper_speed_value"] = \
        data_df.apply(lambda x: upper_bound_dict[(x.link_id, x.assignment_period, x.weekday, x.time_period)], axis=1)
    data_df["lower_speed_value"] = \
        data_df.apply(lambda x: lower_bound_dict[(x.link_id, x.assignment_period, x.weekday, x.time_period)], axis=1)
    data_df['data_outlier_indicator'] = \
        data_df.apply(lambda x: outlier_detector(x.speed, x.upper_speed_value, x.lower_speed_value), axis=1)
    data_df['pair'] = data_df.apply(lambda x: (x.link_id, x.assignment_period, x.date), axis=1)

    outlier_df = pd.pivot_table(data_df, values='data_outlier_indicator',
                                index=['link_id', 'assignment_period', 'date'], aggfunc=np.sum)
    outlier_df['link_id'] = outlier_df.apply(lambda x: int(x.name[0]), axis=1)
    outlier_df['date'] = outlier_df.apply(lambda x: x.name[2], axis=1)
    outlier_df['assignment_period'] = outlier_df.apply(lambda x: x.name[1], axis=1)
    outlier_df['pair'] = outlier_df.apply(lambda x: (x.link_id, x.assignment_period, x.date), axis=1)
    outlier_df.index = range(0, len(outlier_df))
    outlier_df.to_csv("outlier.csv")

    outlier_list = []
    for link_id in outlier_df.link_id.unique():
        df = outlier_df[outlier_df.link_id == link_id]
        threshold = df.data_outlier_indicator.quantile(0.9)
        df1 = df[df.data_outlier_indicator > threshold]
        outlier_list.extend(df1.pair.to_list())

    data_df['outlier_flag'] = data_df.apply(lambda x: outlier_detector_1(x.pair, outlier_list), axis=1)
    data_df = data_df[data_df['outlier_flag'] != 1]

    weekly_pivot_spd = pd.pivot_table(data_df, values='speed', columns=['time_period'],
                                      index=['weekday','assignment_period'], aggfunc=np.mean)
    weekly_pivot_spd.to_csv('corridor_pivot_spd_weekly_' + corridor_name + '.csv')

    weekly_pivot_spd = pd.pivot_table(data_df, values='speed', columns=['time_period'],
                                      index=['link_id','assignment_period'], aggfunc=np.mean)
    weekly_pivot_spd.to_csv('corridor_pivot_spd_' + corridor_name + '.csv')

    pivot_spd = pd.pivot_table(data_df, values='speed', columns=['time_period'],
                               index=['link_id', 'assignment_period', 'weekday'], aggfunc=np.mean)
    pivot_spd.to_csv('corridor_pivot_wd_spd_' + corridor_name + '.csv')

    pivot_lower_spd = pd.pivot_table(data_df, values='lower_speed_value', columns=['time_period'],
                                     index=['link_id', 'assignment_period', 'weekday'], aggfunc=np.mean)

    pivot_upper_spd = pd.pivot_table(data_df, values='upper_speed_value', columns=['time_period'],
                                     index=['link_id', 'assignment_period', 'weekday'], aggfunc=np.mean)

    pivot_spd_ratio = pd.pivot_table(data_df, values='speed_ratio', columns=['time_period'],
                                     index=['link_id', 'assignment_period', 'weekday'], aggfunc=np.mean)

    pivot_vol = pd.pivot_table(data_df, values='volume_per_lane', columns=['time_period'],
                               index=['link_id', 'assignment_period', 'weekday'], aggfunc=np.mean)
    pivot_vol.to_csv('corridor_pivot_wd_vol_' + corridor_name + '.csv')
    pivot_df = pd.DataFrame()
    pivot_df = pivot_df.append(pd.Series(name='volume'))
    pivot_df = pd.concat([pivot_df, pivot_vol])
    pivot_df = pivot_df.append(pd.Series(name='speed'))
    pivot_df = pd.concat([pivot_df, pivot_spd])
    pivot_df = pivot_df.append(pd.Series(name='lower_speed_bound'))
    pivot_df = pd.concat([pivot_df, pivot_lower_spd])
    pivot_df = pivot_df.append(pd.Series(name='upper_speed_bound'))
    pivot_df = pd.concat([pivot_df, pivot_upper_spd])
    pivot_df = pivot_df.append(pd.Series(name='speed_ratio'))
    pivot_df = pd.concat([pivot_df, pivot_spd_ratio])
    pivot_df.to_csv('corridor_pivot_' + corridor_name + '.csv')

    pivot_spd = pd.read_csv('corridor_pivot_wd_spd_' + corridor_name + '.csv')
    pivot_vol = pd.read_csv('corridor_pivot_wd_vol_' + corridor_name + '.csv')
    return data_df



def define_corridor(corridor_name, link_measurement_file='link_performance.csv'):
    data_df = pd.read_csv(link_measurement_file, encoding='UTF-8')
    data_df = data_df.drop(
        data_df[(data_df.volume == 0) | (data_df.speed == 0)].index)  # drop all rows that have 0 volume or speed
    data_df = data_df[data_df['corridor_name'] == corridor_name]
    data_df['speed_ratio'] = data_df['speed'] / data_df['speed_limit']
    TIME_INTERVAL_IN_MIN = _obtain_time_interval(data_df.time_period[0])

    # Calculate some derived properties for each link
    data_df['volume_per_lane'] = data_df['volume'] / data_df['lanes']
    # add an additional column volume_per_lane in the dataframe

    data_df['hourly_volume_per_lane'] = data_df['volume_per_lane'] * (60 / TIME_INTERVAL_IN_MIN)
    # add a column hourly_volume_per_lane
    # in the link_performance.csv, the field "volume" is link volume within the time interval

    data_df['density'] = data_df['hourly_volume_per_lane'] / data_df['speed']
    # add a column density
    data_df = pivot_table(data_df, corridor_name)
    data_df.to_csv('corridor_measurement_' + corridor_name + '.csv', index=False)
    print('DONE')

