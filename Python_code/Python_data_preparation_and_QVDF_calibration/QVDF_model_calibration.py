from data2supplymodel import formating_period_definition
from datetime import datetime
import data2supplymodel as ds
import pandas as pd
import time
# In[0]
import pandas as pd
import os
import time
import datetime
import numpy as np
# In[0] Import necessary packages
# import pandas as pd
# import numpy as np
from scipy.optimize import curve_fit
# from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import datetime

global daily_starting_time

# SETTING ----
# fundamental diagram:
UPPER_BOUND_CRITICAL_DENSITY = 100  # we assume that the upper bound of critical density is 50 vehicle/mile
LOWER_BOUND_CRITICAL_DENSITY = 15  # we assume that the upper bound of critical density is 15 vehicle/mile
OUTER_LAYER_QUANTILE = 0.9  # The quantile threshold to generate the outer layer to calibrate traffic flow model
LOWER_BOUND_OF_OUTER_LAYER_SAMPLES = 40  # number_of_outer_layer_samples

LOWER_BOUND_ALPHA = 0.1
LOWER_BOUND_BETA = 1.01

UPPER_BOUND_ALPHA = 10
UPPER_BOUND_BETA = 10

UPPER_BOUND_DOC = 5

UPPER_BOUND_JAM_DENSITY = 220  # we assume that the upper bound of jam density is 220 vehicle/mile
OUTER_LAYER_QUANTILE = 0.9  # The quantile threshold to generate the outer layer to calibrate traffic flow model
LOWER_BOUND_OF_PHF = 2  # lower bound of peak hour factor
UPPER_BOUND_JAM_DENSITY = 220  # we assume that the upper bound of jam density is 220 vehicle/mile

LOWER_BOUND_OF_PHF_SAMPLES = 10
MIN_THRESHOLD_SAMPLING = 1  # if the missing data of a link during a peak period less than the threshold delete the data
WEIGHT_HOURLY_DATA = 1  # Weight of hourly data during calibration`
WEIGHT_PERIOD_DATA = 10  # 10 # Weight of average period speed and volume during the calibration
WEIGHT_UPPER_BOUND_DOC_RATIO = 100
# 100 # Weight of prompting the VDF curve to 0 when the DOC is close to its maximum values
queue_demand_factor_method = 'SBM'

MEASUREMENT_TYPE = ['single_day', 'multi_day']

# setting for giscsv2gmns

DICT_TYPE_LIST = ['SPEED', 'SPEED_LIMIT', 'AM_CAP', 'MD_CAP', 'PM_CAP', 'NT_CAP', '24H_CAP', 'CAP', 'ALPHA', 'BETA']
ADJACENT_TYPE_LIST = ['SPD_HOV_ID', 'VOL_HOV_ID', 'RAMP_ID', 'REVERSE_ID', 'REVERSE_HOV_ID']
ADJACENT_TYPE_COMPARISON_LIST = ['speed_hov_id', 'volume_hov_id', 'ramp_id', 'reverse_link_id', 'reverse_hov_id']

LINK_PERFORMANCE_TYPE = ['average', 'daily']

OPT_FIELD = ['AB_Direction', 'Direction']

# FACTYPE_DICT={1:'Fwy',2:'Arterial',4:'Arterial',6:'Arterial',9:'Arterial'}
AREATYPE_DICT = {1: 'CBD', 2: 'Outlying CBD', 3: 'Mixed Urban', 4: 'Suburban', 5: 'Rural'}

# mandatory field names
MADATORY_TWO_WAY_NODE_FIELD = ['ID', 'Longitude', 'Latitude', 'TAZ']
MADATORY_TWO_WAY_LINK_FIELD = ['ID', 'From ID', 'To ID', 'Dir', 'AB Length', 'BA Length', 'AB Lanes', 'BA Lanes',
                               'AB Speed', 'Link_type', 'BA Speed', 'FT', 'AT']

MADATORY_LINK_PERFORMANCE_FIELD = ['ID', 'VDF', 'time', 'volume_per_lane', 'speed', 'FT', 'AT', 'lane_name', 'date']

ORIGINAL_FIELD = ['ID', 'From ID', 'To ID', 'Dir']

# setting

DICT_TYPE_LIST = ['SPEED', 'SPEED_LIMIT', 'AM_CAP', 'MD_CAP', 'PM_CAP', 'NT_CAP', '24H_CAP', 'CAP', 'ALPHA', 'BETA']
ADJACENT_TYPE_LIST = ['SPD_HOV', 'VOL_HOV', 'RAMP', 'REVERSE_ID', 'REVERSE_HOV']
ADJACENT_TYPE_COMPARISON_LIST = ['speed_hov_id', 'volume_hov_id', 'ramp_id', 'reverse_link_id', 'reverse_hov_id']

MEASUREMENT_TYPE = ['single_day', 'multi_day']

MADATORY_FIELD = ['ID', 'VDF', 'time', 'volume_per_lane', 'speed', 'FT', 'AT', 'lane_name', 'date']

OPT_FIELD = ['AB_Direction', 'Direction']

FACTYPE_DICT = {1: 'Fwy', 2: 'Arterial', 4: 'Arterial', 6: 'Arterial', 9: 'Arterial'}# Arterial快速路主干路
AREATYPE_DICT = {1: 'CBD', 2: 'Outlying CBD', 3: 'Mixed Urban', 4: 'Suburban', 5: 'Rural'}


# #
# ds.define_demand_period(period_list, measurement_file='link_performance.csv')
# # #
# ds.define_corridor(corridor_name='I10', link_measurement_file='corridor_measurement.csv')
# # # #
# # # # # step 0 Define a VDF area
# ds.calibrate_fundamental_diagram(ft_list='all', at_list='all', measurement_file='corridor_measurement_I10.csv')
#
# # # # # Step 3 calibration
# ds.calibrate_vdf(measurement_file='fd_corridor_measurement_I10.csv', level_of_service=0.7)
#
# ds.generateTimeDependentQueue(link_file='fd_corridor_measurement_I10_training_set.csv')


def _hhmm_to_minutes(time_period):
    """
    example 1:
        input-> 0700_0715
        output-> 420
    example 2:
        input-> 0800_0815
        output-> 480
    """
    var = time_period.split('_')[0]
    var_minutes = float(var[0:2]) * 60 + float(var[2:4])
    return var_minutes


def _obtain_time_interval(hhmm_time_interval):
    """
    example 1:
        input-> 0700_0715
        output-> 15
    example 2:
        input-> 0700_0705
        output-> 5
    """
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


def pivot_table(data_df, corridor_name):
    pivot_spd = pd.pivot_table(data_df, values='speed', columns=['time_period'],
                               index=['link_id', 'assignment_period', 'weekday'], aggfunc=np.mean)  # pivot_spd_mean
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
    outlier_df.to_csv(".\output\outlier.csv")

    outlier_list = []
    for link_id in outlier_df.link_id.unique():
        df = outlier_df[outlier_df.link_id == link_id]
        threshold = df.data_outlier_indicator.quantile(0.9)
        df1 = df[df.data_outlier_indicator > threshold]
        outlier_list.extend(df1.pair.to_list())

    data_df['outlier_flag'] = data_df.apply(lambda x: outlier_detector_1(x.pair, outlier_list), axis=1)
    data_df = data_df[data_df['outlier_flag'] != 1]

    weekly_pivot_spd = pd.pivot_table(data_df, values='speed', columns=['time_period'],
                                      index=['weekday', 'assignment_period'], aggfunc=np.mean)
    weekly_pivot_spd.to_csv('./output/corridor_pivot_spd_weekly_' + corridor_name + '.csv')

    weekly_pivot_spd = pd.pivot_table(data_df, values='speed', columns=['time_period'],
                                      index=['link_id', 'assignment_period'], aggfunc=np.mean)
    weekly_pivot_spd.to_csv('./output/corridor_pivot_spd_' + corridor_name + '.csv')

    pivot_spd = pd.pivot_table(data_df, values='speed', columns=['time_period'],
                               index=['link_id', 'assignment_period', 'weekday'], aggfunc=np.mean)
    pivot_spd.to_csv('./output/corridor_pivot_wd_spd_' + corridor_name + '.csv')

    pivot_lower_spd = pd.pivot_table(data_df, values='lower_speed_value', columns=['time_period'],
                                     index=['link_id', 'assignment_period', 'weekday'], aggfunc=np.mean)

    pivot_upper_spd = pd.pivot_table(data_df, values='upper_speed_value', columns=['time_period'],
                                     index=['link_id', 'assignment_period', 'weekday'], aggfunc=np.mean)

    pivot_spd_ratio = pd.pivot_table(data_df, values='speed_ratio', columns=['time_period'],
                                     index=['link_id', 'assignment_period', 'weekday'], aggfunc=np.mean)

    pivot_vol = pd.pivot_table(data_df, values='volume_per_lane', columns=['time_period'],
                               index=['link_id', 'assignment_period', 'weekday'], aggfunc=np.mean)
    pivot_vol.to_csv('./output/corridor_pivot_wd_vol_' + corridor_name + '.csv')
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
    pivot_df.to_csv('./output/corridor_pivot_' + corridor_name + '.csv')

    pivot_spd = pd.read_csv('./output/corridor_pivot_wd_spd_' + corridor_name + '.csv')
    pivot_vol = pd.read_csv('./output/corridor_pivot_wd_vol_' + corridor_name + '.csv')
    return data_df


def outlier_detector(speed, upper_bound, lower_bound):
    # speed -> 60
    # upper_bound -> 50
    # lower_bound -> 0
    # speed not between ub and lb
    # need operation
    # flag = ABS(60-25) = 35
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


def _read_data(measurement_file, ft_list, at_list):
    data_df = pd.read_csv(measurement_file, encoding='UTF-8')  # create data frame from the input link_performance.csv
    if (ft_list != 'all') & (at_list != 'all'):
        data_df = data_df[data_df['FT'].isin(ft_list) & data_df['AT'].isin(at_list)]
    elif (ft_list == 'all') & (at_list != 'all'):
        data_df = data_df[data_df['AT'].isin(at_list)]
    elif (at_list == 'all') & (ft_list != 'all'):
        data_df = data_df[data_df['FT'].isin(ft_list)]

    data_df = data_df.drop(
        data_df[(data_df.volume == 0) | (data_df.speed == 0)].index
    )  # drop all rows that have 0 volume or speed
    data_df.dropna(axis=0, how='any', inplace=True)  # drop all rows that have any null value
    data_df.reset_index(drop=True, inplace=True)  # reset the index of the dataframe

    return data_df


def _calibrate_traffic_flow_model(vdf_training_set, vdf_index, sub_folder, reference_free_flow_speed):
    # 1. set the lower bound and upper bound of the free flow speed value
    lower_bound_FFS = np.minimum(vdf_training_set['speed'].quantile(0.95), reference_free_flow_speed)
    # Assume that the lower bound of freeflow speed should be larger than the mean value of speed
    upper_bound_FFS = np.minimum(vdf_training_set['speed'].max(), reference_free_flow_speed + 0.001)
    # np.maximum(reference_free_flow_speed+0.1, lower_bound_FFS+0.1)

    # Assume that the upper bound of freeflow speed should at least larger than the lower bound,
    # and less than the maximum value of speed

    # lower_bound_FFS = reference_free_flow_speed
    # upper_bound_FFS = reference_free_flow_speed + 0.1

    # 2. generate the outer layer of density-speed  scatters
    vdf_training_set_after_sorting = vdf_training_set.sort_values(by='speed')
    # sort speed value from the smallest to the largest

    vdf_training_set_after_sorting.reset_index(drop=True, inplace=True)
    # reset the index

    step_size = np.maximum(1, int((vdf_training_set['speed'].max() - vdf_training_set['speed'].min())
                                  / LOWER_BOUND_OF_OUTER_LAYER_SAMPLES))
    # determine the step_size of each segment to generate the outer layer

    X_data = []
    Y_data = []
    for k in range(0, int(np.ceil(vdf_training_set['speed'].max())), step_size):
        segment_df = vdf_training_set_after_sorting[
            (vdf_training_set_after_sorting.speed < k + step_size) & (vdf_training_set_after_sorting.speed >= k)]
        Y_data.append(segment_df.speed.mean())
        threshold = segment_df['density'].quantile(OUTER_LAYER_QUANTILE)
        X_data.append(segment_df[(segment_df['density'] >= threshold)]['density'].mean())
    XY_data = pd.DataFrame({'X_data': X_data, 'Y_data': Y_data})
    XY_data = XY_data[~XY_data.isin([np.nan, np.inf, -np.inf]).any(1)]  # delete all the infinite and null values
    if len(XY_data) == 0:
        print('WARNING: No available data within all speed segments')
        exit()
    density_data = XY_data.X_data.values
    speed_data = XY_data.Y_data.values

    # 3. calibrate traffic flow model using scipy function curve_fit.
    # #More information about the function, see
    # https://docs.scipy.org/doc/scipy-0.15.1/reference/generated/scipy.optimize.curve_fit.html
    popt, pcov = curve_fit(_density_speed_function, density_data, speed_data,
                           bounds=([lower_bound_FFS, LOWER_BOUND_CRITICAL_DENSITY, 0],
                                   [upper_bound_FFS, UPPER_BOUND_CRITICAL_DENSITY, 15]))

    free_flow_speed = popt[0]
    critical_density = popt[1]
    mm = popt[2]
    speed_at_capacity = free_flow_speed / np.power(2, 2 / mm)
    ultimate_capacity = speed_at_capacity * critical_density

    # draw pictures
    xvals = np.linspace(0, UPPER_BOUND_JAM_DENSITY, 100)
    # all the data points with density values
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 14
    plt.plot(vdf_training_set_after_sorting['density'], vdf_training_set_after_sorting['speed'], '*', c='k',
             label='observations', markersize=1)
    plt.plot(xvals, _density_speed_function(xvals, *popt), '--', c='b', markersize=6, label='speed-density curve')
    plt.scatter(density_data, speed_data, edgecolors='r', color='r', label='outer layer', s=6, zorder=30)
    plt.legend(loc='best')
    plt.grid(True)
    plt.title('Density-speed fundamental diagram, VDF: ' + str(vdf_index[0] + vdf_index[1] * 100))
    plt.xlabel('Density')
    plt.ylabel('Speed')
    plt.axis([0, 300, 0, 100])
    plt.savefig(sub_folder + '1_FD_speed_density_' + str(vdf_index[0] + vdf_index[1] * 100) + '.png')
    plt.close('all')

    plt.plot(vdf_training_set_after_sorting['hourly_volume_per_lane'], vdf_training_set_after_sorting['speed'], '*',
             c='k', label='Data', markersize=1)
    plt.plot(xvals * _density_speed_function(xvals, *popt), _density_speed_function(xvals, *popt), '--', c='b',
             markersize=6, label='Estimated curve')
    # plt.scatter(density_data*speed_data, speed_data,edgecolors='r',color='r',label ='outer layer',zorder=30)
    plt.legend(loc='best')
    plt.grid(True)
    plt.title('Volume-speed fundamental diagram, ID: ' + str(vdf_index[0] + vdf_index[1] * 100))
    plt.xlabel('Volume (vehicles per hour per lane)')
    plt.ylabel('Speed (mile/hour)')
    plt.axis([0, 2000, 0, 80])
    plt.savefig(sub_folder + '1_FD_speed_volume_' + str(vdf_index[0] + vdf_index[1] * 100) + '.png')
    plt.close('all')

    plt.plot(vdf_training_set_after_sorting['density'], vdf_training_set_after_sorting['hourly_volume_per_lane'], '*',
             c='k', label='original values', markersize=1)
    plt.plot(xvals, xvals * _density_speed_function(xvals, *popt), '--', c='b', markersize=6,
             label='density-volume curve')
    # plt.scatter(density_data,density_data*speed_data,edgecolors='r',color='r',label ='outer layer',zorder=30)
    plt.legend(loc='best')
    plt.grid(True)
    plt.title('Density-volume fundamental diagram,VDF: ' + str(vdf_index[0] + vdf_index[1] * 100))
    plt.xlabel('Density')
    plt.ylabel('Volume')
    plt.axis([0, 250, 0, 2200])
    plt.savefig(sub_folder + '1_FD_volume_density_' + str(vdf_index[0] + vdf_index[1] * 100) + '.png')
    plt.close('all')

    return speed_at_capacity, ultimate_capacity, critical_density, free_flow_speed, mm


def _density_speed_function(density, free_flow_speed, critical_density, mm):
    # fundamental diagram model (density-speed function):
    # More information the density-speed function:
    # Paper: An_s-shaped_three-dimensional_S3_traffic_stream_model_with_consistent_car_following_relationship

    k_over_k_critical = density / critical_density
    denominator = np.power(1 + np.power(k_over_k_critical, mm), 2 / mm)
    speed = free_flow_speed / denominator
    return speed


if __name__ == "__main__":
    st = time.time()

    # ----- STEP 1

    # input
    period_list = ['0700_2100']
    measurement_file = './input/link_performance.csv'
    output_folder = "./output"

    data_df = pd.read_csv(measurement_file, encoding='UTF-8')
    data_df['time_minutes'] = data_df.apply(lambda x: _hhmm_to_minutes(x.time_period), axis=1)
    TIME_INTERVAL_IN_MIN = _obtain_time_interval(data_df.time_period[0])

    print("---Join Demand Periods---")
    global daily_starting_time
    period_name_list = []
    start_time_list = []
    end_time_list = []
    all_data_df = pd.DataFrame()
    # 目前感觉这个for循环没啥意义
    for period_name in period_list:  # parser HHMM time, period length
        # time_start = time.time()
        period_name_list.append(period_name)
        # modify
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
        # time_end = time.time()
        # print('join assignment period:', period_name, ', using time', time_end - time_start, '...\n')
        print('join assignment period:', period_name)

    daily_starting_time = min(start_time_list)
    # time_start = time.time()
    # print('start add time index using...\n')
    all_data_df.reset_index(drop=True, inplace=True)
    time_period_list = all_data_df['time_period'].to_list()
    map_result = list(map(_get_index, time_period_list))
    # time_end = time.time()
    all_data_df['time_index'] = np.array(map_result)
    # time_end = time.time()
    # print('add time index using:', time_end - time_start, 's...DONE.\n')

    time_start = time.time()
    all_data_df.to_csv(os.path.join(output_folder, 'corridor_measurement.csv'), index=False)
    time_end = time.time()
    # return all_data_df

    # STEP 2 ---------

    # def define_corridor(corridor_name, link_measurement_file='link_performance.csv'):
    link_measurement_file = './input/link_performance.csv'
    data_df = pd.read_csv(link_measurement_file, encoding='UTF-8')
    corridor_name = 'I10'
    data_df = data_df.drop(
        data_df[(data_df.volume == 0) | (data_df.speed == 0)].index
    )  # drop all rows that have 0 volume or speed
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
    data_df.to_csv('.\output\corridor_measurement_' + corridor_name + '.csv', index=False)

    # STEP 3 ----------------------
    # ds.calibrate_fundamental_diagram(ft_list='all', at_list='all', measurement_file='corridor_measurement_I10.csv')
    output_folder = './output/output_fundamental_diagrams'

    measurement_file = './output/corridor_measurement_I10.csv'
    sub_folder = './' + output_folder + '/'
    ft_list = 'all'
    at_list = 'all'
    training_set = _read_data(measurement_file, ft_list, at_list)

    vdf_index = [100 * i + j for i in training_set.AT.unique() for j in training_set.FT.unique()]
    diagram_table_column = ['VDF', 'ultimate_capacity', 'speed_at_capacity', 'critical_density', 'free_flow_speed',
                            'flatness_of_curve']

    diagram_table = pd.DataFrame(columns=diagram_table_column, index=vdf_index)

    # Step 2: For each VDF type, calibrate basic coefficients for fundamental diagrams
    # Step 2.1: Group the data frame by VDF. Each VDF type is a combination of facility type (FT) and area type (AT)
    # Step 2.1: Group the data frame by VDF. Each VDF type is a combination of facility type (FT) and area type (AT)
    vdf_group = training_set.groupby(['FT', 'AT'])
    all_training_set = pd.DataFrame()
    for vdf_index, vdf_training_set in vdf_group:
        # vdf_index is a pair of facility type and area type e.g. vdf_index = (1,1) implies that FT=1 and AT=1
        # speed_limit is a mandatory field
        reference_free_flow_speed = vdf_training_set.speed_limit.mean()
        temp_vdf_training_set = vdf_training_set[vdf_training_set.speed <= reference_free_flow_speed].copy()
        # ----
        vdf_training_set.reset_index(drop=True, inplace=True)  # reset index of the sub dataframe
        speed_at_capacity, ultimate_capacity, critical_density, free_flow_speed, mm = _calibrate_traffic_flow_model(
            temp_vdf_training_set, vdf_index, sub_folder, reference_free_flow_speed)

        diagram_table.loc[100 * vdf_index[1] + vdf_index[0], 'VDF'] = 100 * vdf_index[1] + vdf_index[0]
        diagram_table.loc[100 * vdf_index[1] + vdf_index[0], 'ultimate_capacity'] = ultimate_capacity
        diagram_table.loc[100 * vdf_index[1] + vdf_index[0], 'speed_at_capacity'] = speed_at_capacity
        diagram_table.loc[100 * vdf_index[1] + vdf_index[0], 'critical_density'] = critical_density
        diagram_table.loc[100 * vdf_index[1] + vdf_index[0], 'free_flow_speed'] = free_flow_speed
        diagram_table.loc[100 * vdf_index[1] + vdf_index[0], 'flatness_of_curve'] = mm

        vdf_training_set['vdf_type'] = 100 * vdf_index[1] + vdf_index[0]
        vdf_training_set['ultimate_capacity'] = ultimate_capacity
        vdf_training_set['speed_at_capacity'] = speed_at_capacity
        vdf_training_set['critical_density'] = critical_density
        vdf_training_set['free_flow_speed'] = free_flow_speed
        vdf_training_set['flatness_of_curve'] = mm
        all_training_set = pd.concat([all_training_set, vdf_training_set])

        print('calibrate fundamental diagram of VDF type', vdf_index)
        print('--speed_at_capacity=', speed_at_capacity)
        print('--ultimate_capacity=', ultimate_capacity)
        print('--critical_density=', critical_density)
        print('--free_flow_speed=', free_flow_speed)
        print('--flatness_of_curve=', mm)

    diagram_table.dropna(axis=0, how='all', inplace=True)
    diagram_table.to_csv(sub_folder + 'fundamental_diagram_table.csv', index=False)  # day by day
    diagram_table.to_csv('fundamental_diagram_table.csv', index=False)  # day by day
    all_training_set.reset_index(drop=True, inplace=True)
    TIME_INTERVAL_IN_MIN = _obtain_time_interval(all_training_set.time_period[0])

    all_training_set['derived_speed'] = \
        np.minimum(all_training_set['speed'], all_training_set['free_flow_speed'] * 0.99)

    # 下面的这几行能不能重写一下？
    all_training_set['derived_volume_per_lane'] = \
        all_training_set['derived_speed'] * all_training_set['critical_density'] \
        * (((all_training_set['free_flow_speed'] ** all_training_set['flatness_of_curve'])
            / (all_training_set['derived_speed'] ** all_training_set['flatness_of_curve'])) ** 0.5 - 1) ** \
        (1 / all_training_set['flatness_of_curve'])

    #     data_df['time_minutes'] = data_df.apply(lambda x: _hhmm_to_minutes(x.time_period), axis=1)
    # all_training_set["derived_volume_per_lane2"]=all_training_set.apply(lambda x:, _density_speed_function(x.),axis=1)

    # all_training_set['volume_per_lane'] = all_training_set['volume_per_lane'] / (60 / TIME_INTERVAL_IN_MIN)
    # all_training_set['volume'] = all_training_set['volume_per_lane'] * all_training_set['lanes']
    all_training_set.to_csv("./output/output_fundamental_diagrams/fd_" + measurement_file, index=False)

    print(f"Elapsed time = {time.time() - st:.2f} sec")
    print("DONE!!")
