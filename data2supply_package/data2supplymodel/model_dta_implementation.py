# In[0] Import necessary packages 
import csv
import time
import numpy as np
from .model_vdf_calibration import _uncongested_volume_speed_function
import pandas as pd

_NUMBER_OF_SECOND_PER_MIN = 60


class Link:
    def __init__(self, link_id, road_order, V, qdf, t2, L, NL, C, vc, vct, vf, mm, a, b, tau_0, tau_3,
                 time_interval_in_min, v_max, f_p, f_d, ss, nn, wkd):
        self.link_id = link_id
        self.road_order = int(float(road_order))
        self.V = float(V)
        self.ref_V = float(V)
        self.qdf = float(qdf)
        self.t2 = float(t2)
        self.L = float(L)
        self.NL = float(NL)
        self.C = float(C)
        self.vc = float(vc)
        self.vct = float(vct)
        self.kc = self.C / max(0.0001, self.vc)
        self.vf = float(vf)
        self.mm = float(mm)
        self.cd_alpha = float(a)
        self.cd_beta = float(b)
        self.tau_0 = int(float(tau_0))
        self.tau_3 = int(float(tau_3))
        self.v_max = np.minimum(float(v_max), float(vf))
        self.time_interval_in_min = float(time_interval_in_min)
        self.f_p = float(f_p)
        self.f_d = float(f_d)
        self.ss = float(ss)
        self.nn = float(nn)
        self.wkd = int(wkd)


def _congested_volume_speed_function(vct, D_C, cd_alpha, cd_beta):
    #speed = vct / ((7 / 15) + (8 / 15) * np.power(D_C, cd_alpha * cd_beta))
    speed = vct / (1 + cd_alpha * np.power(D_C, cd_beta))
    return speed


def generateTimeDependentQueue(link_file='link_performance.csv'):
    """
    Congestion Demand Based Calibration
    
    Input(link based):
    private:
    (1) Volume during assignment period——V
    (2) qdf factor=Volume/Demand——qdf
    (3) t2 intermediate time point of the congestion period——t2
    (4) link length——L
    public
    (5) ultimate capacity——C
    (6) critical speed——vc
    (7) free flow speed——vf
    (8) flatness of speed——mm
    (9) cd_alpha——a
    (10) cd_beta——b

    Output(link based):
    (1) Demand during assignment period——D
    (2) D/C
    (3) cd_speed——vcd
    (4) cd_travel_time——ttcd
    (5) average_discharge_rate——mu
    (6) cd_waiting_time——wcd
    (7) queue_demand——QD
    (8) Gamma
    (9) Congestion_duration——P
    (10) t0,t3,
    (11) t0',t3' no congestion 
    (12) Queue_max
    
    Time dependent queue
    Time dependent travel time

    """
    g_link_list = list()
    with open(link_file, "r") as fp:
        reader = csv.DictReader(fp)
        for line in reader:
            link = Link(line["link_id"], line['road_order'], line["assignment_period_volume"], line["qdf"], line["t2"],
                        line["length"], line['lanes'], line["ultimate_capacity"],
                        line["critical_speed"], line["cut_off_speed"],
                        line["free_flow_speed"], line["flatness_of_curve"], line["cd_alpha"], line["cd_beta"],
                        line['tau_0'], line['tau_3'], line['time_interval_in_min'], line['max_speed'] ,
                        line["f_p"], line["f_d"], line["ss"], line["nn"], line['weekday'])
            g_link_list.append(link)

    performance_fp = open("tmc_link_performance_" + link_file[:-4] + ".csv", "w", newline="")
    performance_writer = csv.writer(performance_fp)
    line = ["link_id", "road_order", "weekday","ref_period_volume", "assignment_period_volume", "qdf", "demand", "D/C",
            "cd_speed", "cd_travel_time", "average_discharge_rate",
            "cd_waiting_time", "queue_demand", "Gamma", "Congestion_duration", "v_t2", "t0", "t3", "Queue_max"]
    performance_writer.writerow(line)

    assign_period_start_time_in_hour = link.tau_0
    assign_period_end_time_in_hour = link.tau_3
    time_interval_in_min = link.time_interval_in_min
    number_of_interval = int(
        (assign_period_end_time_in_hour - assign_period_start_time_in_hour) * 60 / time_interval_in_min) + 1

    td_queue_fp = open("td_queue_" + link_file[:-4] + ".csv", "w", newline="")
    td_speed_fp = open("td_speed_" + link_file[:-4] + ".csv", "w", newline="")
    td_speed_ratio_fp = open("td_speed_ratio_" + link_file[:-4] + ".csv", "w", newline="")
    td_queue_writer = csv.writer(td_queue_fp)
    td_speed_writer = csv.writer(td_speed_fp)
    td_speed_ratio_writer = csv.writer(td_speed_ratio_fp)
    line = ["link_id", "road_order", 'weekday']
    init_time = time.struct_time([2021, 8, 15, assign_period_start_time_in_hour, 0, 0, 0, 0, 0])
    init_time_mktime = time.mktime(init_time)
    time_in_hour = list()
    for t in range(number_of_interval):
        time_str = time.strftime("%H:%M:%S", time.localtime(init_time_mktime))
        time_struc = time.localtime(init_time_mktime)
        line.append(time_str)
        time_in_hour.append(time_struc.tm_hour + time_struc.tm_min / 60 + time_struc.tm_sec / 3600)
        init_time_mktime += time_interval_in_min * _NUMBER_OF_SECOND_PER_MIN
    td_queue_writer.writerow(line)
    td_speed_writer.writerow(line)
    td_speed_ratio_writer.writerow(line)

    # df = pd.read_csv("tmc_link_future_year_s.csv")
    # V_dict = dict(zip(df.tmc, df['STA_volume3']))

    for link in g_link_list:
        # if link.V==0:
        #     continue
        #link.V = (V_dict.setdefault(link.link_id, link.ref_V) / (link.NL))
        demand = link.V * link.qdf
        D_C = demand / link.C

        P = np.maximum(link.f_d*np.power(D_C, link.nn), D_C)
        v_t2 = link.vct / (1+link.f_p*np.power(P, link.ss))
        wt_t2 = link.L / v_t2 - link.L / link.vct
        mu = np.minimum(demand / P, link.C)

        gamma = 4 * wt_t2 * mu / np.power(P / 2, 4)
        wcd = (gamma / (120 * mu)) * np.power(P, 4)
        link.L / link.vct
        QD = demand
        vcd = link.L / (link.L / link.vct + wcd)
        vcd_1 = _congested_volume_speed_function(link.vct, D_C, link.cd_alpha,
                                                 link.cd_beta)
        if vcd_1 != vcd:
            print(vcd_1-vcd)
        ttcd = link.L / vcd
        # if D_C < 1:
        #     QD = 0
        #     vcd = _uncongested_volume_speed_function(demand, link.kc, link.mm, link.vf, link.C)
        # elif D_C >= 1:
        #     QD = demand
        #     P = np.maximum(np.power(D_C, link.cd_beta), 1)
        #     vcd = link.L / (link.L / link.vct + wcd)
        #     vcd_1 = _congested_volume_speed_function(link.vct, D_C, link.cd_alpha,
        #                                              link.cd_beta)  # another way to derive vcd
        #     if vcd_1 == vcd:
        #         vcd = vcd_1


        # wcd = max(ttcd - link.L / link.vct, 0)
        # gamma_1 = 120 * mu * wcd * pow(mu / demand, 4)  # another way to derive gamma
        # if gamma_1 >= gamma:
        #     gamma = gamma_1

        t0 = link.t2 - 0.5 * P
        t3 = link.t2 + 0.5 * P
        t0_p = t0
        t3_p = t3
        # if P == 0:
        #     t0_p -= 0.5
        #     t3_p += 0.5

        Queue_max = 0.25 * gamma * pow((link.t2 - t0), 2) * pow((link.t2 - t3), 2)
        performance_line = [link.link_id, link.road_order, link.wkd, link.ref_V, link.V, link.qdf, demand, D_C, vcd, ttcd, mu,
                            wcd, QD, gamma, P, v_t2, t0_p, t3_p, round(Queue_max, 0)]
        performance_writer.writerow(performance_line)

        td_queue_line = [link.link_id, link.road_order, link.wkd]
        td_speed_line = [link.link_id, link.road_order, link.wkd]
        td_speed_ratio_line = [link.link_id, link.road_order]
        for t in time_in_hour:
            if t0 <= t <= t3:
                td_queue = 0.25 * gamma * pow((t - t0), 2) * pow((t - t3), 2)
            else:
                td_queue = 0

            if P < 0:
                if t < t0_p:
                    td_speed = link.v_max - ((link.v_max - vcd) / (t0_p - assign_period_start_time_in_hour)) * (
                                t - assign_period_start_time_in_hour)
                elif t > t3_p:
                    td_speed = link.v_max - ((link.v_max - vcd) / (assign_period_end_time_in_hour - t3_p)) * (
                                assign_period_end_time_in_hour - t)
                else:
                    td_speed = vcd
            else:
                if t < t0_p:
                    td_speed = link.v_max - ((link.v_max - link.vct) / (t0_p - assign_period_start_time_in_hour)) * (
                                t - assign_period_start_time_in_hour)
                elif t > t3_p:
                    td_speed = link.v_max - ((link.v_max - link.vct) / (assign_period_end_time_in_hour - t3_p)) * (
                                assign_period_end_time_in_hour - t)
                else:
                    td_speed = link.L / ((td_queue / mu) + (link.L / link.vct))
            td_speed_ratio = td_speed / link.vf
            td_queue_line.append(round(td_queue, 0))
            td_speed_line.append(td_speed)
            td_speed_ratio_line.append(td_speed_ratio)
        td_queue_writer.writerow(td_queue_line)
        td_speed_writer.writerow(td_speed_line)
        td_speed_ratio_writer.writerow(td_speed_ratio_line)

    performance_fp.close()
    td_queue_fp.close()
    td_speed_fp.close()
    td_speed_ratio_fp.close()
    print("DTA DONE")
