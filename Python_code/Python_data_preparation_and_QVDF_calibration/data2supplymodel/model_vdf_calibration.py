# In[0] Import necessary packages 
import pandas as pd
import numpy as np
import datetime
from scipy.optimize import curve_fit
from .setting import *
import matplotlib.pyplot as plt
import random

global TIME_INTERVAL_IN_MIN

plt.rc('font', family='Times New Roman')
plt.rc('font', size=10)

g_number_of_plink = 0
g_plink_id_dict = {}
g_plink_nb_seq_dict = {}
g_parameter_list = []
g_vdf_group_list = []


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


# In[1] input data
def _read_input_data(measurement_file):
    data_df = pd.read_csv(measurement_file, encoding='UTF-8')  # create data frame from the input link_performance.csv
    global TIME_INTERVAL_IN_MIN
    TIME_INTERVAL_IN_MIN = _obtain_time_interval(data_df.time_period[0])
    data_df.dropna(axis=0, how='any', inplace=True)  # drop all rows that have any null value
    data_df.reset_index(drop=True, inplace=True)  # reset the index of the dataframe
    return data_df


def _initialization(training_set):
    ASSIGNMENT_PERIOD = training_set.assignment_period.unique().tolist()  # array of assignment periods
    PERIOD_LENGTH = []  # list of the length of assignment periods
    NUMBER_OF_RECORDS = []  # list of the number of records of assignment periods
    period_start_time_list = []
    for period in ASSIGNMENT_PERIOD:  # parsor HHMM time, period length
        period_start_time_list.append(int(period.split('_')[0][0:2]) * 60 + int(period.split('_')[0][2:4]))
        time_ss = [int(var[0:2]) for var in period.split('_')]
        if time_ss[0] > time_ss[1]:
            period_length = time_ss[1] + 24 - time_ss[
                0]  # e.g. if the assignment period is 1800_0600, then we will calculate that 6-18+24=12
            number_of_records = period_length * (
                    60 / TIME_INTERVAL_IN_MIN)  # calculate the complete number of records in the time-series data of a link during an assignment period, e.g if  assignment period 0600_0900 should have 3 hours * 4 records (if time stamp is 15 minutes)
        else:
            period_length = time_ss[1] - time_ss[0]
            number_of_records = period_length * (60 / TIME_INTERVAL_IN_MIN)

        PERIOD_LENGTH.append(period_length)
        NUMBER_OF_RECORDS.append(number_of_records)

    period_length_dict = dict(zip(ASSIGNMENT_PERIOD, PERIOD_LENGTH))
    number_of_records_dict = dict(zip(ASSIGNMENT_PERIOD, NUMBER_OF_RECORDS))
    return training_set, period_length_dict, number_of_records_dict


def _bpr_func(x, ffs, alpha, beta):  # BPR volume delay function input: volume over capacity
    return ffs / (1 + alpha * np.power(x, beta))


def _calibrated_power_function(x, alpha, beta):
    return alpha * np.power(x, beta)


def _calibrated_linear_function(x, alpha, beta):
    return alpha * x + beta


def _calibrate_qvdf(x, alpha, beta, v_c):
    return v_c / (1 + alpha * (x ** beta))


def _uncongested_volume_speed_function(flow, kc, mm, vf, cap):
    # flow=flow/3600
    # vf=vf/3600
    vc = vf / np.power(2, 2 / mm)
    kc = cap / vc

    aa = kc ** mm
    bb = (kc ** mm) * ((vf ** mm) ** 0.5)
    if cap == flow:
        delta = 0
    else:
        delta = kc ** (2 * mm) * (vf ** mm) - 4 * (kc ** mm) * (flow ** mm)
    xx = (bb + np.power(delta, 0.5)) / (2 * aa)
    speed = np.power(np.power(xx, 2), 1 / mm)
    # speed=speed*3600
    return speed


# In[3] Calibrate traffic flow model

# In[4] VDF calibration


def _vdf_calculation_stepwise(internal_period_vdf_daily_link_df, period_index, vdf_index, link_id):
    lb_fitting = [0, 0]  # upper bound and lower bound of free flow speed, alpha and beta
    ub_fitting = [10, 10]

    print('1. Estimate congestion duration of link ' + str(link_id) + ' during time period:' + period_index + '...')
    X_data = []
    Y_data = []
    training_set = internal_period_vdf_daily_link_df[internal_period_vdf_daily_link_df['b_congestion_duration'] != 0]
    threshold_1 = training_set.b_congestion_duration.mean() + training_set.b_congestion_duration.std()
    training_set = \
        training_set[training_set['b_congestion_duration'] <= threshold_1]
    training_set.reset_index(inplace=True)
    training_set['vc/vt2-1'] = training_set.cut_off_speed / training_set.v_t2 - 1
    training_set['log_b_congestion_duration'] = np.log(training_set['b_congestion_duration'])
    training_set['log_vc/vt2-1'] = np.log(training_set['vc/vt2-1'])
    training_set['log_demand_over_capacity'] = np.log(training_set['demand_over_capacity'])

    training_set_0 = pd.pivot_table(training_set, values='demand_over_capacity', index=['b_congestion_duration'],
                                    aggfunc=np.mean)
    training_set_0['b_congestion_duration'] = training_set_0.index
    training_set_0.index = range(0, len(training_set_0))

    training_set_1 = pd.pivot_table(training_set, values='vc/vt2-1', index=['b_congestion_duration'], aggfunc=np.mean)
    training_set_1['b_congestion_duration'] = training_set_1.index
    training_set_1.index = range(0, len(training_set_1))

    # Step 1: y = ax^b
    for k in range(0, len(training_set_0)):
        # Hourly hourly_demand_over_capacity data 
        Y_data.append(training_set_0.loc[k, 'b_congestion_duration'])
        X_data.append(training_set_0.loc[k, 'demand_over_capacity'])

    x_demand_over_capacity = np.array(X_data)
    y_congestion_duration = np.array(Y_data)
    popt, pcov = curve_fit(_calibrated_power_function, x_demand_over_capacity, y_congestion_duration,
                           bounds=[lb_fitting, ub_fitting])
    RSE = np.sum(
        np.power((_calibrated_power_function(x_demand_over_capacity, *popt) - y_congestion_duration), 2)) / np.sum(
        np.power((_calibrated_power_function(x_demand_over_capacity, *popt) - y_congestion_duration.mean()), 2))

    xvals = np.linspace(0, np.ceil(x_demand_over_capacity.max()), 50)
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 14
    plt.plot(x_demand_over_capacity, y_congestion_duration,
             'o', c='r', markerfacecolor='none', label='Mean value at same P ', markersize=6)
    plt.scatter(training_set['demand_over_capacity'], training_set['b_congestion_duration'],
                edgecolors='k', color='k', label='Data', s=3, zorder=30)
    yvals = _calibrated_power_function(xvals, *popt)
    plt.plot(xvals, yvals, '--', c='b', markersize=6, label='Estimated curve')
    # plt.title('period:' + str(period_index) + ',ID:' + str(vdf_index) +
    #           ',f_d=' + str(round(popt[0], 4)) + ',n=' + str(round(popt[1], 4)))
    plt.title('ID:' + str(vdf_index) +
              ', fd=' + str(round(popt[0], 4)) + ', n=' + str(round(popt[1], 4)))
    plt.xlabel('D/C ratio')
    plt.ylabel('Congestion_duration P (hours)')
    plt.legend(loc='best')
    plt.grid(True)
    plt.savefig(folder + 'DOC_vs_CD_' + str(period_index) + '_' + str(vdf_index) + '_' + str(link_id) + '.png')
    plt.close('all')

    f_d = popt[0]
    nn = popt[1]

    # x_demand_over_capacity = np.array(X_data)
    # log_x_demand_over_capacity = np.log(x_demand_over_capacity)
    # y_congestion_duration = np.array(Y_data)
    # log_y_congestion_duration = np.log(y_congestion_duration)
    #
    # popt, pcov = curve_fit(_calibrated_linear_function, log_x_demand_over_capacity, log_y_congestion_duration)
    # RSE = \
    #     np.sum(np.power((_calibrated_linear_function(log_x_demand_over_capacity, *popt)
    #                      - log_y_congestion_duration), 2)) / \
    #     np.sum(np.power((_calibrated_linear_function(log_x_demand_over_capacity, *popt)
    #                      - log_y_congestion_duration.mean()), 2))
    # popt1 = popt.copy()
    # popt1[1] = popt[0]
    # popt1[0] = np.exp(popt[1])
    # xvals = np.linspace(np.floor(log_x_demand_over_capacity.min()), np.ceil(log_x_demand_over_capacity.max()), 50)
    # plt.plot(log_x_demand_over_capacity, log_y_congestion_duration, '*', c='r', label='mean value',
    #          markersize=8)
    # plt.plot(xvals, _calibrated_linear_function(xvals, *popt), '--', c='b', markersize=6, label='curve_fitting')
    # plt.scatter(training_set['log_demand_over_capacity'], training_set['log_b_congestion_duration'],
    #             edgecolors='k', color='k', label='Log_DOC vs.Log_congestion duration', s=3, zorder=30)
    # plt.title(str(period_index) + ',' + str(vdf_index) +
    #           ',f_d=' + str(round(popt1[0], 4)) + ',n=' + str(round(popt1[1], 4)) + ',RSE=' + str(round(RSE, 2)) + '%')
    # plt.xlabel('Log_Hourly_demand_over_capacity')
    # plt.ylabel('Log_b_congestion_duration')
    # plt.legend(loc='best')
    # plt.grid(True)
    # plt.savefig(folder + 'Log_DOC_vs_Log_CD_' + str(period_index) + '_' + str(vdf_index) + '_' + str(link_id) + '.png')
    # plt.close('all')
    #
    # internal_period_vdf_daily_link_df['n'] = round(popt[0], 2)

    # plastic beta

    lb_fitting = [0, 0]  # upper bound and lower bound of alpha and beta
    ub_fitting = [10, 10]

    print('2. Estimate the lowest speed of link ' + str(link_id) + ' during time period:' + period_index + '...')
    X_data = []
    Y_data = []
    # training_set = training_set_1.copy()

    for k in range(0, len(training_set_1)):
        # Hourly hourly_demand_over_capacity data 
        X_data.append(np.maximum(0, training_set_1.loc[k, 'vc/vt2-1']))
        Y_data.append(training_set_1.loc[k, 'b_congestion_duration'])

    x_congestion_duration = np.array(Y_data)
    y_vct_over_v_t2 = np.array(X_data)

    popt, pcov = curve_fit(_calibrated_power_function, x_congestion_duration, y_vct_over_v_t2,
                           bounds=[lb_fitting, ub_fitting])
    RSE = np.sum(np.power((_calibrated_power_function(x_congestion_duration, *popt) - y_vct_over_v_t2), 2)) / np.sum(
        np.power((_calibrated_power_function(x_congestion_duration, *popt) - y_vct_over_v_t2.mean()), 2))

    popt1 = popt.copy()
    xvals = np.linspace(0, np.ceil(x_congestion_duration.max()), 50)
    plt.plot(x_congestion_duration, y_vct_over_v_t2,
             'o', c='r', markerfacecolor='none', label='Mean value at same P', markersize=6)
    plt.plot(xvals, _calibrated_power_function(xvals, *popt1), '--', c='b', markersize=6, label='Estimated curve')
    plt.scatter(training_set['b_congestion_duration'], training_set['vc/vt2-1'],
                edgecolors='k', color='k', label='Data', s=3, zorder=30)

    # plt.title('period:' + str(period_index) + ', ID:' + str(vdf_index) +
    #           ',f_p=' + str(round(popt1[0], 5)) + ',s=' + str(round(popt1[1], 3)))
    plt.title('ID:' + str(vdf_index) +
              ', fp=' + str(round(popt1[0], 5)) + ', s=' + str(round(popt1[1], 3)))
    plt.xlabel('Congestion duration P (hours)')
    plt.ylabel('Magnitude of speed reduction')
    plt.legend(loc='best')
    plt.grid(True)
    plt.savefig(folder + 'CD_vs_vc_over_vt2-1' + str(period_index) + '_' + str(vdf_index) + '_' + str(link_id) + '.png')
    plt.close('all')
    f_p = popt[0]
    ss = popt1[1]

    # x_congestion_duration = np.array(Y_data)
    # log_x_congestion_duration = np.log(x_congestion_duration)
    # y_vct_over_v_t2 = np.array(X_data)
    # log_y_vct_over_v_t2 = np.log(y_vct_over_v_t2)
    #
    # popt, pcov = curve_fit(_calibrated_linear_function, log_x_congestion_duration, log_y_vct_over_v_t2)
    # RSE = np.sum(
    #     np.power((_calibrated_linear_function(log_x_congestion_duration, *popt) - log_y_vct_over_v_t2), 2)) / np.sum(
    #     np.power((_calibrated_linear_function(log_x_congestion_duration, *popt) - log_y_vct_over_v_t2.mean()), 2))
    #
    # popt1 = popt.copy()
    # popt1[1] = popt[0]
    # popt1[0] = np.exp(popt[1])
    # xvals = np.linspace(np.floor(log_x_congestion_duration.min()), np.ceil(log_x_congestion_duration.max()), 50)
    # plt.plot(log_x_congestion_duration, log_y_vct_over_v_t2, '*', c='r', label='mean value', markersize=8)
    # plt.plot(xvals, _calibrated_linear_function(xvals, *popt), '--', c='b', markersize=6, label='curve_fitting')
    # plt.scatter(training_set['log_b_congestion_duration'], training_set['log_vc/vt2-1'],
    #             edgecolors='k', color='k', label='congestion duration vs.vc/vt2-1', s=3, zorder=30)
    # plt.title(str(period_index) + ',' + str(vdf_index) +
    #           ',f_p=' + str(round(popt1[0], 5)) + ',s=' + str(round(popt1[1], 3)) + ',RSE=' + str(round(RSE, 2)) + '%')
    # plt.xlabel('congestion duration')
    # plt.ylabel('vc/vt2-1')
    # plt.legend(loc='best')
    # plt.grid(True)
    # plt.savefig(
    #     folder + 'LOG_CD_vs_vc_over_vt2-1' + str(period_index) + '_' + str(vdf_index) + '_' + str(link_id) + '.png')
    # plt.close('all')
    #
    # popt, pcov = curve_fit(_calibrated_power_function, y_vct_over_v_t2, x_congestion_duration,
    #                        bounds=[lb_fitting, ub_fitting])
    # RSE = np.sum(np.power((_calibrated_power_function(y_vct_over_v_t2, *popt) - x_congestion_duration), 2)) / np.sum(
    #     np.power((_calibrated_power_function(y_vct_over_v_t2, *popt) - x_congestion_duration.mean()), 2))
    #
    # popt1 = popt.copy()
    # popt1[1] = 1 / popt[1]
    # popt1[0] = 1 / ((popt[0]) ** popt1[1])
    # xvals = np.linspace(0, np.ceil(y_vct_over_v_t2.max()), 50)
    # plt.plot(y_vct_over_v_t2, x_congestion_duration, '*', c='r', label='mean value', markersize=8)
    # plt.plot(xvals, _calibrated_power_function(xvals, *popt), '--', c='b', markersize=6, label='curve_fitting')
    # plt.scatter(training_set['vc/vt2-1'], training_set['b_congestion_duration'],
    #             edgecolors='k', color='k', label='congestion duration vs.vc/vt2-1', s=3, zorder=30)
    #
    # plt.title(str(period_index) + ',' + str(vdf_index) +
    #           ',f_p=' + str(round(popt1[0], 5)) + ',s=' + str(round(popt1[1], 3)) + ',RSE=' + str(round(RSE, 2)) + '%')
    # plt.xlabel('vc/vt2-1')
    # plt.ylabel('congestion duration')
    # plt.legend(loc='best')
    # plt.grid(True)
    # plt.savefig(folder + 'vc_over_vt2-1_vs_CD' + str(period_index) + '_' + str(vdf_index) + '_' + str(link_id) + '.png')
    # plt.close('all')

    # internal_period_vdf_daily_link_df['cd_alpha'] = round(popt[0], 2)

    print('3. Estimate the VDF curve of link ' + str(link_id) + ' during time period:' + period_index + '...')
    X_data = []
    Y_data = []

    cd_alpha = f_p * (f_d ** ss) * (8 / 15)
    cd_beta = ss * nn
    internal_period_vdf_daily_link_df['cd_alpha'] = cd_alpha
    internal_period_vdf_daily_link_df['cd_beta'] = cd_beta
    internal_period_vdf_daily_link_df['f_p'] = f_p
    internal_period_vdf_daily_link_df['f_d'] = f_d
    internal_period_vdf_daily_link_df['ss'] = ss
    internal_period_vdf_daily_link_df['nn'] = nn

    xvals = np.linspace(0, np.ceil(training_set['demand_over_capacity'].max()), 50)
    plt.plot(training_set['demand_over_capacity'], training_set['cd_mean_speed'],
             '.', c='k', label='Data', markersize=8)
    vct = training_set.cut_off_speed.mean()
    popt = np.array([cd_alpha, cd_beta, vct])
    plt.plot(xvals, _calibrate_qvdf(xvals, *popt), '--', c='b', markersize=6, label='Estimated curve')
    plt.xlabel('D/C ratio')
    plt.ylabel('Mean speed during congestion duration (miles/hour)')
    plt.legend(loc='best')
    plt.grid(True)
    plt.title('ID:' + str(vdf_index) +
              ', alpha=' + str(round(cd_alpha, 3)) + ', beta=' + str(round(cd_beta, 3)))
    plt.savefig(folder + 'DOC_vs_CD_speed' + str(period_index) + '_' + str(vdf_index) + '_' + str(link_id) + '.png')
    plt.close('all')

    return internal_period_vdf_daily_link_df


# In[5] Calculate demand and congestion period
def _calculate_congestion_duration(speed_series, volume_per_lane_series, cut_off_speed, period_link_volume,
                                   time_interval_in_min, ultimate_capacity, period_index):
    nb_time_stamp = len(speed_series)
    min_speed = min(speed_series)
    max_speed = max(speed_series)
    t2_index = speed_series.index(min(speed_series))  # The index of speed with minimum value
    first_half_hour = np.ceil((60 / time_interval_in_min) / 2)
    second_half_hour = (60 / time_interval_in_min) - first_half_hour - 1
    # start time and ending time of preferred service time window
    peak_hour_start_time = max(t2_index - first_half_hour, 0)
    peak_hour_ending_time = min(t2_index + second_half_hour, nb_time_stamp)
    if peak_hour_ending_time - peak_hour_start_time < (60 / time_interval_in_min) - 1:
        if peak_hour_start_time == 0:
            peak_hour_ending_time = peak_hour_ending_time + (
                    (60 / time_interval_in_min) - 1 - (peak_hour_ending_time - peak_hour_start_time))
        if peak_hour_ending_time == nb_time_stamp:
            peak_hour_start_time = peak_hour_start_time - (
                    (60 / time_interval_in_min) - 1 - (peak_hour_ending_time - peak_hour_start_time))
    peak_hour = (peak_hour_ending_time - peak_hour_start_time + 1) * (time_interval_in_min / 60)

    # Determine
    t3_index = nb_time_stamp - 1
    t0_index = 0
    if min_speed <= cut_off_speed:
        for i in range(t2_index, nb_time_stamp):
            if speed_series[i] > cut_off_speed:
                t3_index = i - 1
                break
        for j in range(t2_index, -1, -1):
            if speed_series[j] > cut_off_speed:
                t0_index = j + 1
                break
        b_congestion_duration = (t3_index - t0_index + 1) * (time_interval_in_min / 60)
        # if b_congestion_duration <= peak_hour:
        #     t0_index = int(peak_hour_start_time)
        #     t3_index = int(peak_hour_ending_time)
    elif min_speed > cut_off_speed:
        t0_index = int(peak_hour_start_time)
        t3_index = int(peak_hour_ending_time)
        b_congestion_duration = 0

    cd_mean_speed = np.array(speed_series[t0_index:t3_index + 1]).mean()
    demand = np.array(volume_per_lane_series[t0_index:t3_index + 1]).sum()
    mu = np.array(volume_per_lane_series[t0_index:t3_index + 1]).mean() * (60 / time_interval_in_min)
    # mu = demand / b_congestion_duration
    qdf = 1 / (period_link_volume / demand)

    assignment_period_start_time = period_index.split('_')
    temp_time = pd.to_datetime(assignment_period_start_time[0], format='%H%M', errors='ignore')
    t0 = temp_time + datetime.timedelta(minutes=time_interval_in_min) * t0_index
    t3 = temp_time + datetime.timedelta(minutes=time_interval_in_min) * (t3_index + 1)
    t2 = temp_time + datetime.timedelta(minutes=time_interval_in_min) * t2_index
    t0 = str(t0.time())[0:5]
    t2 = str(t2.time())[0:5]
    t3 = str(t3.time())[0:5]
    t2_index_1 = temp_time.hour + t2_index * (time_interval_in_min / 60)
    DOC = demand / ultimate_capacity
    v_t2 = min_speed

    return b_congestion_duration, demand, cd_mean_speed, qdf, DOC, t0, t2_index_1, t3, mu, v_t2, max_speed


# In[6] Histogram

def calibrate_vdf(measurement_file, level_of_service):
    global DOC_RATIO_METHOD
    global folder
    output_folder = 'output_calibration' + measurement_file[:-4]
    _mkdir(output_folder)

    folder = './' + output_folder + '/'
    # step 1 input data
    pivot_vol = pd.read_csv("corridor_pivot_wd_vol_I10.csv")
    pivot_spd = pd.read_csv("corridor_pivot_wd_spd_I10.csv")

    training_set = _read_input_data(measurement_file)
    training_set, period_length_dict, number_of_records_dict = _initialization(training_set)
    # training_set['cut_off_speed'] = training_set['speed_at_capacity']
    training_set['cut_off_speed'] = training_set['free_flow_speed'] * level_of_service

    # training_set is a data frame of pandas to store the whole link_performance.csv file
    # Step 2: For each VDF and assignment period, calibrate congestion duration, demand, t0, t3, alpha and beta
    print('calibrate congestion period and vdf parameters....')
    all_training_set = pd.DataFrame()
    # Create empty dataframe
    period_vdf_group = training_set.groupby(['FT', 'AT', 'assignment_period'])
    # group the vdf_train set according to time periods

    for index, period_vdf_training_set in period_vdf_group:  # _period_based_vdf_training_set
        period_index = index[2]
        vdf_index = index[1] * 100 + index[0]
        period_vdf_training_set.reset_index(drop=True, inplace=True)
        # reset index of the sub dataframe
        time_interval_in_min = _obtain_time_interval(period_vdf_training_set.time_period[0])
        daily_link_group = period_vdf_training_set.groupby(['link_id'])
        start_time = \
            datetime.datetime.strptime(period_index.split('_')[0][:2] + ':' + period_index.split('_')[0][2:], '%H:%M')
        end_time = \
            datetime.datetime.strptime(period_index.split('_')[1][:2] + ':' + period_index.split('_')[1][2:], '%H:%M')

        tau_0 = start_time.hour
        tau_3 = end_time.hour

        all_period_link_training_set = pd.DataFrame()
        for link_id, link_training_set in daily_link_group:
            link_training_set.groupby(['date'])
            daily_link_list = []
            speed_at_cap = link_training_set.speed_at_capacity.mean()
            max_speed = 0
            for date_id, period_link_training_set in link_training_set.groupby(['date']):
                period_link_training_set = period_link_training_set.sort_values(by='time_index', ascending=True)
                period_link_training_set.reset_index(drop=True, inplace=True)  # reset index of the sub dataframe
                period_link_training_set['before_smooth_speed'] = period_link_training_set['speed'].copy()
                period_link_training_set['speed'] = period_link_training_set['speed'].rolling(5, min_periods=1).mean()
                # plt.plot(period_link_training_set['speed'])
                link_id = link_id
                road_order = period_link_training_set.road_order[0]
                date = date_id
                FT = index[0]
                AT = index[1]
                weekday = period_link_training_set.weekday[0]
                if (len(period_link_training_set) < number_of_records_dict[period_index]) and ((1 - len(
                        period_link_training_set) / number_of_records_dict[period_index]) >= MIN_THRESHOLD_SAMPLING):
                    print('WARNING:link ', link_id, 'does not have enough time series records '
                                                    'during the assignment period', period_index, 'at date ', date_id)
                    continue
                period_link_volume = period_link_training_set['volume_per_lane'].sum()
                # summation of all volume per lane within the period
                period_mean_link_speed = period_link_training_set['speed'].mean()
                period_mean_link_density = period_link_training_set['density'].mean()
                volume_per_lane_series = period_link_training_set.volume_per_lane.to_list()
                speed_series = period_link_training_set.speed.to_list()
                link_length = period_link_training_set.length.mean()  # obtain the length of the link
                link_lanes = period_link_training_set.lanes.mean()  # obtain the number of lanes of the link
                link_free_flow_speed = period_link_training_set['free_flow_speed'].mean()
                speed_at_capacity = link_training_set.speed_at_capacity.mean()
                cut_off_speed = link_training_set.cut_off_speed.mean()
                ultimate_capacity = period_link_training_set['ultimate_capacity'].mean()
                critical_density = period_link_training_set['critical_density'].mean()
                flatness_of_curve = period_link_training_set['flatness_of_curve'].mean()

                # Step 3.1 Calculate Demand over capacity and congestion duration
                b_congestion_duration, demand, cd_mean_speed, qdf, demand_over_capacity, t0, t2, t3, mu, v_t2, v_max \
                    = _calculate_congestion_duration(speed_series, volume_per_lane_series, cut_off_speed,
                                                     period_link_volume, time_interval_in_min, ultimate_capacity,
                                                     period_index)

                max_speed = max(max_speed, v_max)
                daily_link = [link_id, date, FT, AT, link_length, link_lanes, period_index,
                              period_link_volume, cd_mean_speed, mu, demand, demand_over_capacity, t2,
                              v_t2, qdf, link_free_flow_speed, speed_at_capacity, cut_off_speed, ultimate_capacity,
                              critical_density, flatness_of_curve,
                              b_congestion_duration, period_mean_link_speed, period_mean_link_density,
                              t0, t3, time_interval_in_min, tau_0, tau_3, road_order, weekday]
                daily_link_list.append(daily_link)

            internal_period_vdf_daily_link_df = pd.DataFrame(daily_link_list)
            internal_period_vdf_daily_link_df.rename(columns={0: 'link_id',
                                                              1: 'date',
                                                              2: 'FT',
                                                              3: 'AT',
                                                              4: 'length',
                                                              5: 'lanes',
                                                              6: 'assignment_period',
                                                              7: 'period_link_volume',
                                                              8: 'cd_mean_speed',
                                                              9: 'congestion_duration_mean_flow_rate',
                                                              10: 'demand',
                                                              11: 'demand_over_capacity',
                                                              12: 't2',
                                                              13: 'v_t2',
                                                              14: 'qdf',
                                                              15: 'free_flow_speed',
                                                              16: 'critical_speed',
                                                              17: 'cut_off_speed',
                                                              18: 'ultimate_capacity',
                                                              19: 'critical_density',
                                                              20: 'flatness_of_curve',
                                                              21: 'b_congestion_duration',
                                                              22: 'period_mean_link_speed',
                                                              23: 'period_mean_link_density',
                                                              24: 'benchmark_t0',
                                                              25: 'benchmark_t3',
                                                              26: 'time_interval_in_min',
                                                              27: 'tau_0',
                                                              28: 'tau_3',
                                                              29: 'road_order',
                                                              30: 'weekday'}, inplace=True)
            internal_period_vdf_daily_link_df['max_speed'] = max_speed
            internal_period_vdf_daily_link_df['assignment_period_volume'] \
                = internal_period_vdf_daily_link_df['period_link_volume']
            internal_period_vdf_daily_link_df = _vdf_calculation_stepwise(internal_period_vdf_daily_link_df,
                                                                          period_index, vdf_index, link_id)
            df = internal_period_vdf_daily_link_df[internal_period_vdf_daily_link_df.b_congestion_duration != 0]

            all_period_link_training_set = pd.concat([all_period_link_training_set, internal_period_vdf_daily_link_df],
                                                     sort=False)
        all_training_set = pd.concat([all_training_set, all_period_link_training_set], sort=False)

    # all_training_set.to_csv(measurement_file[:-4] + '_training_set_per_day_1.csv', index=False)

    all_mean_training_set = pd.DataFrame()
    all_training_set_group = all_training_set.groupby(['link_id', 'weekday', 'assignment_period'])
    for pair, sub_all_training_set_group in all_training_set_group:
        df = sub_all_training_set_group[sub_all_training_set_group.b_congestion_duration != 0]
        sub_all_training_set_group['avg_demand'] = df.demand.mean()
        all_mean_training_set = pd.concat([all_mean_training_set, sub_all_training_set_group], sort=False)
    all_mean_training_set['est_cd'] = \
        all_mean_training_set['f_d'] * all_mean_training_set['demand_over_capacity'] ** all_mean_training_set['nn']
    all_mean_training_set['est_vt2'] = \
        all_mean_training_set['cut_off_speed'] / \
        (1 + all_mean_training_set['f_p'] * all_mean_training_set['est_cd'] ** all_mean_training_set['ss'])
    all_mean_training_set['est_mu'] = \
        all_mean_training_set['demand'] / all_mean_training_set['est_cd']
    all_mean_training_set['est_v_bar'] = \
        all_mean_training_set['cut_off_speed'] / \
        (1 + all_mean_training_set['cd_alpha'] *
         (all_mean_training_set['demand_over_capacity'] ** all_mean_training_set['cd_beta']))

    all_mean_training_set['error_cd'] = \
        all_mean_training_set.apply(lambda x: np.abs(x.est_cd-x.b_congestion_duration)/x.b_congestion_duration, axis=1)
    all_mean_training_set['error_vt2'] = \
        all_mean_training_set.apply(lambda x: np.abs(x.est_vt2-x.v_t2)/x.v_t2, axis=1)
    all_mean_training_set['error_v_bar'] = \
        all_mean_training_set.apply(lambda x: np.abs(x.est_v_bar-x.cd_mean_speed)/x.cd_mean_speed, axis=1)

    all_mean_training_set.to_csv(measurement_file[:-4] + '_training_set_per_day.csv', index=False)

    # all_mean_training_set = pd.DataFrame()  # Create empty dataframe
    all_avg_training_set = all_mean_training_set.groupby(['link_id', 'assignment_period', 'weekday']).mean()
    #all_avg_training_set['qdf'] = all_avg_training_set.avg_demand / all_avg_training_set.assignment_period_volume
    all_avg_training_set['index_name'] = all_avg_training_set.index
    all_avg_training_set = all_avg_training_set.reset_index()
    all_avg_training_set['link_id'] = all_avg_training_set.apply(lambda x: x.index_name[0], axis=1)
    all_avg_training_set['assignment_period'] = all_avg_training_set.apply(lambda x: x.index_name[1], axis=1)
    all_avg_training_set['weekday'] = all_avg_training_set.apply(lambda x: x.index_name[2], axis=1)
    all_avg_training_set = all_avg_training_set.drop(['index_name'], axis=1)
    all_avg_training_set.to_csv(measurement_file[:-4] + '_training_set.csv', index=False)

    all_avg_training_set['tuple_index'] = \
        all_avg_training_set.apply(lambda x: (x.link_id, x.assignment_period, x.weekday), axis=1)

    period_link_volume_dict = {}
    cd_mean_speed_dict = {}
    congestion_duration_mean_flow_rate_dict = {}
    demand_dict = {}
    demand_over_capacity_dict = {}
    t2_dict = {}
    v_t2_dict = {}
    b_congestion_duration_dict = {}
    qdf_dict = {}
    for ii in range(len(pivot_spd)):
        speed_series = np.array(pivot_spd.loc[ii][3:]).tolist()
        volume_per_lane_series = np.array(pivot_vol.loc[ii][3:]).tolist()
        period_link_volume = np.sum(volume_per_lane_series)
        b_congestion_duration, demand, cd_mean_speed, qdf, demand_over_capacity, t0, t2, t3, mu, v_t2, v_max \
            = _calculate_congestion_duration(speed_series, volume_per_lane_series, cut_off_speed,
                                             period_link_volume, time_interval_in_min, ultimate_capacity,
                                             period_index)
        print(t2)
        tuple_index = (pivot_spd.loc[ii].link_id, pivot_spd.loc[ii].assignment_period, pivot_spd.loc[ii].weekday)
        print(tuple_index)
        period_link_volume_dict[tuple_index] = period_link_volume
        cd_mean_speed_dict[tuple_index] = cd_mean_speed
        congestion_duration_mean_flow_rate_dict[tuple_index] = mu
        demand_dict[tuple_index] = demand
        demand_over_capacity_dict[tuple_index] = demand_over_capacity
        t2_dict[tuple_index] = t2
        v_t2_dict[tuple_index] = v_t2
        b_congestion_duration_dict[tuple_index] = b_congestion_duration
        qdf_dict[tuple_index] = qdf

    all_avg_training_set['period_link_volume'] = \
        all_avg_training_set.apply(lambda x: period_link_volume_dict[x.tuple_index], axis=1)
    all_avg_training_set['cd_mean_speed'] = \
        all_avg_training_set.apply(lambda x: cd_mean_speed_dict[x.tuple_index], axis=1)
    all_avg_training_set['congestion_duration_mean_flow_rate'] = \
        all_avg_training_set.apply(lambda x: congestion_duration_mean_flow_rate_dict[x.tuple_index], axis=1)
    all_avg_training_set['demand'] = \
        all_avg_training_set.apply(lambda x: demand_dict[x.tuple_index], axis=1)
    all_avg_training_set['demand_over_capacity'] = \
        all_avg_training_set.apply(lambda x: demand_over_capacity_dict[x.tuple_index], axis=1)
    all_avg_training_set['t2'] = \
        all_avg_training_set.apply(lambda x: t2_dict[x.tuple_index], axis=1)
    all_avg_training_set['v_t2'] = \
        all_avg_training_set.apply(lambda x: v_t2_dict[x.tuple_index], axis=1)
    all_avg_training_set['b_congestion_duration'] = \
        all_avg_training_set.apply(lambda x: b_congestion_duration_dict[x.tuple_index], axis=1)
    all_avg_training_set['qdf'] = \
        all_avg_training_set.apply(lambda x: qdf_dict[x.tuple_index], axis=1)

    all_avg_training_set.to_csv(measurement_file[:-4] + '_training_set_1.csv', index=False)

    print('END...')
