# In[0] Import necessary packages 
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression
from .setting import *
import matplotlib.pyplot as plt
import datetime

def _calibrate_traffic_flow_model(vdf_training_set, vdf_index, sub_folder, reference_free_flow_speed):
    # 1. set the lower bound and upper bound of the free flow speed value 
    lower_bound_FFS = np.minimum(vdf_training_set['speed'].quantile(0.95), reference_free_flow_speed)
    # Assume that the lower bound of freeflow speed should be larger than the mean value of speed
    upper_bound_FFS = np.minimum(vdf_training_set['speed'].max(), reference_free_flow_speed+0.001)
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


def _mkdir(path):
    import os
    path = path.strip()
    path = path.rstrip("\\")
    isExists = os.path.exists(path)

    if not isExists:
        os.makedirs(path)
        print(path + ' create the folder sucessfully')
        return True
    else:
        print(path + ' the folder already exists')
        return False


def _obtain_time_interval(hhmm_time_interval):
    start_time = datetime.datetime.strptime(
        hhmm_time_interval.split('_')[0][:2] + ':' + hhmm_time_interval.split('_')[0][2:], '%H:%M')
    start_time_minute = start_time.hour * 60 + start_time.minute
    end_time = datetime.datetime.strptime(
        hhmm_time_interval.split('_')[1][:2] + ':' + hhmm_time_interval.split('_')[1][2:], '%H:%M')
    end_time_minute = end_time.hour * 60 + end_time.minute
    time_interval = end_time_minute - start_time_minute
    return time_interval


def _read_data(measurement_file, ft_list, at_list):
    data_df = pd.read_csv(measurement_file, encoding='UTF-8')  # create data frame from the input link_performance.csv
    if (ft_list != 'all') & (at_list != 'all'):
        data_df = data_df[data_df['FT'].isin(ft_list) & data_df['AT'].isin(at_list)]
    elif (ft_list == 'all') & (at_list != 'all'):
        data_df = data_df[data_df['AT'].isin(at_list)]
    elif (at_list == 'all') & (ft_list != 'all'):
        data_df = data_df[data_df['FT'].isin(ft_list)]

    data_df = data_df.drop(
        data_df[(data_df.volume == 0) | (data_df.speed == 0)].index)  # drop all rows that have 0 volume or speed
    data_df.dropna(axis=0, how='any', inplace=True)  # drop all rows that have any null value
    data_df.reset_index(drop=True, inplace=True)  # reset the index of the dataframe

    return data_df


# main function

def calibrate_fundamental_diagram(ft_list='all', at_list='all', measurement_file='link_performance.csv'):
    output_folder = 'output_fundamental_diagrams'
    _mkdir(output_folder)

    sub_folder = './' + output_folder + '/'
    training_set = _read_data(measurement_file, ft_list, at_list)

    vdf_index = [100 * i + j for i in training_set.AT.unique() for j in training_set.FT.unique()]
    diagram_table_column = ['VDF', 'ultimate_capacity', 'speed_at_capacity', 'critical_density', 'free_flow_speed',
                            'flatness_of_curve']

    diagram_table = pd.DataFrame(columns=diagram_table_column, index=vdf_index)

    # Step 2: For each VDF type, calibrate basic coefficients for fundamental diagrams
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

    all_training_set['derived_volume_per_lane'] = \
        all_training_set['derived_speed'] * all_training_set['critical_density'] \
        * (((all_training_set['free_flow_speed'] ** all_training_set['flatness_of_curve'])
            / (all_training_set['derived_speed'] ** all_training_set['flatness_of_curve'])) ** 0.5 - 1) ** \
        (1 / all_training_set['flatness_of_curve'])

    # all_training_set['volume_per_lane'] = all_training_set['volume_per_lane'] / (60 / TIME_INTERVAL_IN_MIN)
    # all_training_set['volume'] = all_training_set['volume_per_lane'] * all_training_set['lanes']
    all_training_set.to_csv("fd_"+measurement_file, index=False)
