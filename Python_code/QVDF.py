# -*- coding:utf-8 -*-
##############################################################
# Created Date: Thursday, December 1st 2022
# Contact Info: luoxiangyong01@gmail.com
# Author/Copyright: Mr. Xiangyong Luo
##############################################################


import math


class PeriodVDF:
    def __init__(self, **kwargs):

        self.vdf_type = 0

        self.vf = 60
        self.v_congestion_cutoff = 45
        self.k_critical = 45

        self.Q_cd = 0.954946463
        self.Q_n = 1.141574427
        self.Q_cp = 0.400089684
        self.Q_s = 4
        self.Q_alpha = 0.272876961
        self.Q_beta = 4
        self.Q_mu = 1800
        self.Q_gamma = 0

        self.FFTT = 1
        self.peak_load_factor = 1
        self.alpha = 0.39999993
        self.beta = 4

        self.starting_time_in_hour = 0
        self.ending_time_in_hour = 0

        self.cycle_length = -1
        self.red_time = 0
        self.saturation_flow_rate = 1800
        self.L = 1
        self.num_lanes = 1
        self.t2 = 1

        self.lane_based_ultimate_hourly_capacity = 0

        self.__init_not_used()

        # update parameters from kwargs
        if kwargs:
            for key in kwargs:
                setattr(self, key, kwargs[key])

    def __init_not_used(self):
        self.vdf_data_count = 0
        self.DOC = 0
        self.VOC = 0
        self.vt2 = -1
        self.rho = 1
        self.preload = 0
        self.penalty = 0
        self.sa_lanes_change = 0
        self.LR_price = [0]
        self.LR_RT_price = [0]
        self.effective_green_time = 0
        self.t0 = -1
        self.t3 = -1
        self.start_green_time = -1
        self.end_green_time = -1
        self.queue_length = 0
        self.obs_count = 0
        self.upper_bound_flag = 1
        self.est_count_dev = 0
        self.avg_waiting_time = 0
        self.P = -1
        self.Severe_Congestion_P = -1
        self.lane_based_D = 0
        self.lane_based_Vph = 0
        self.avg_speed_BPR = -1
        self.avg_queue_speed = -1
        self.sa_volume = 0
        self.link_volume = 0
        self.network_design_flag = 0
        self.volume_before = 0
        self.speed_before = 0
        self.DoC_before = 0
        self.P_before = 0
        self.volume_after = 0
        self.speed_after = 0
        self.DoC_after = 0
        self.P_after = 0

        self.toll = [0 for _ in range(10)]
        self.pce = [1 for _ in range(10)]
        self.occ = [1 for _ in range(10)]
        self.RT_allowed_use = [True for _ in range(10)]

        self.dsr = []  # desired speed ratio with respect to free-speed

    def perform_signal_VDF(self, hourly_per_lane_volume: float, red_: float, cycle_length: float) -> float:
        lambda_ = hourly_per_lane_volume
        mu = 1800  # default saturation flow rate
        # 60.0 is used to convert sec to min
        s_bar = 1.0 / 60.0 * red_ * red_ / (2 * cycle_length)
        uniform_delay = s_bar / max(1 - lambda_ / mu, 0.1)

        return uniform_delay

    def get_speed_from_volume(self, hourly_volume: float, vf: float, k_critical: float, s3_m: float) -> float:
        # test data free_speed = 55.0f;
        # speed = 52
        # k_critical = 23.14167648

        max_lane_capacity = k_critical * vf / math.pow(2, 2 / s3_m)

        hourly_volume = min(hourly_volume, max_lane_capacity)
        # we should add a capacity upper bound on hourly_volume

        coef_a = math.pow(k_critical, s3_m)
        coef_b = math.pow(k_critical, s3_m) * math.pow(vf, s3_m / 2)
        coef_c = hourly_volume ** s3_m
        # D is hourly demand volume, which is equivalent to flow q in S3 model

        speed = (coef_b + coef_b * coef_b - 4.0 * coef_a *
                 coef_c ** 0.5) / (2.0 * coef_a) ** 2.0 / s3_m
        speed = math.pow(coef_b + math.pow(coef_b * coef_b -
                         4 * coef_a * coef_c, 0.5) / (2 * coef_a), 2 / s3_m)

        # under un-congested condition
        speed = min(speed, vf)
        speed = max(speed, 0)
        return speed

    def get_volume_from_speed(self, speed: float, vf: float, k_critical: float, s3_m: float) -> float:
        # test data free_speed = 55.0f;
        # speed = 52
        # k_critical = 23.14167648

        if speed < 0:
            return -1

        speed_ratio = vf / max(1.0, speed)
        speed_ratio = max(speed_ratio, 1.00001)

        #   float volume = 0
        ratio_difference = math.pow(speed_ratio, s3_m / 2) - 1

        ratio_difference_final = max(ratio_difference, 0.00000001)

        volume = speed * k_critical * math.pow(ratio_difference_final, 1 / s3_m)
        # volume per hour per lane

        return volume

    def calculate_travel_time_based_on_QVDF(self, volume: float,
                                            model_speed: list = [0] * 300,
                                            est_volume_per_hour_per_lane: list = [0] * 300):

        dc_transition_ratio = 1

        # step 1: calculate lane_based D based on plf and num_lanes from link volume V over the analysis period  take non-negative values
        lane_based_D = max(0.0, volume) / max(0.000001, self.num_lanes) / self.L / self.peak_load_factor

        # step 2: D_C ratio based on lane-based D and lane-based ultimate hourly capacity,
        # un-congested states D < C
        # congested states D > C, leading to P > 1
        DOC = lane_based_D / max(0.00001, self.lane_based_ultimate_hourly_capacity)

        # step 3.1 fetch vf and v_congestion_cutoff based on FFTT, VCTT (to be comparable with transit data, such as waiting time )
        # we could have a period based FFTT, so we need to convert FFTT to vfree
        # if we only have one period, then we can directly use vf and v_congestion_cutoff.

        # step 3.2 calculate speed from VDF based on D/C ratio
        avg_queue_speed = self.v_congestion_cutoff / (1.0 + self.Q_alpha * math.pow(DOC, self.Q_beta))

        # step 3.3 taking the minimum of BPR- v and Q VDF v based on log sum function

        # let us use link_length_in_km = 1 for the following calculation
        link_length_in_1km = 1.0
        RTT = link_length_in_1km / self.v_congestion_cutoff

        Q_n_current_value = self.Q_n

        if DOC < dc_transition_ratio:  # free flow regime

            vf_alpha = (1.0 + self.Q_alpha) * self.vf / max(0.0001, self.v_congestion_cutoff) - 1.0
            # fixed to pass through vcutoff point vf/ (1+vf_alpha) = vc / (1+ qvdf_alpha) ->
            # 1+vf_alpha = vf/vc *(1+qvdf_alpha)
            # vf_qlpha =  vf/vc *(1+qvdf_alpha) - 1
            # revised BPR DC
            vf_beta = self.beta  # to be calibrated

            vf_avg_speed = self.vf / (1.0 + vf_alpha * math.pow(DOC, vf_beta))

            avg_queue_speed = vf_avg_speed  # rewrite with vf based speed

            Q_n_current_value = self.beta

            RTT = link_length_in_1km / max(0.01, vf_avg_speed)  # in hour

        # BPR
        # step 2: D_ C ratio based on lane-based D and lane-based ultimate hourly capacity,
        # uncongested states D <C
        # congested states D > C, leading to P > 1
        VOC = DOC

        # step 3.1 fetch vf and v_congestion_cutoff based on FFTT, VCTT (to be compartible with transit data, such as waiting time )
        # we could have a period based FFTT, so we need to convert FFTT to vfree
        # if we only have one period, then we can directly use vf and v_congestion_cutoff.

        # step 3.2 calculate speed from VDF based on D/C ratio
        avg_speed_BPR = self.vf / (1.0 + self.alpha * VOC ** self.beta)

        if self.vdf_type == 0:
            # Mark: FFTT should be vctt
            avg_travel_time = self.FFTT * self.vf / max(0.1, avg_queue_speed)
        else:
            avg_travel_time = self.FFTT * self.vf / max(0.1, avg_speed_BPR)

        if self.cycle_length >= 1:  # signal delay
            # 60.0 is used to convert sec to min
            s_bar = 1.0 / 60 * self.red_time * self.red_time / (2 * self.cycle_length)
            lambda_ = lane_based_D
            uniform_delay = s_bar / max(1 - lambda_ / self.saturation_flow_rate, 0.1)
            avg_travel_time = uniform_delay + self.FFTT

        avg_waiting_time = avg_travel_time - self.FFTT

        # step 4.4 compute vt2
        # vt2 = avg_queue_speed * 8.0 / 15.0;  // 8/15 is a strong assumption

        # applifed for both uncongested and congested conditions
        P = self.Q_cd * (DOC ** Q_n_current_value)

        base = self.Q_cp * (P ** self.Q_s) + 1.0
        vt2 = self.v_congestion_cutoff / max(0.001, base)

        # step 4.1: compute congestion duration P
        non_peak_hourly_flow = 0

        if self.L - P >= 10.0 / 60.0:
            non_peak_hourly_flow = (volume * (1 - self.peak_load_factor)) / max(1.0, self.num_lanes) / max(0.1, min(self.L - 1, self.L - P - 5.0 / 60))
            # 5.0/60 as one 5 min interval, P includes both boundary points

        # setup the upper bound on nonpeak flow rates
        non_peak_hourly_flow = min(non_peak_hourly_flow, self.lane_based_ultimate_hourly_capacity)

        # later we will use piecewise approximation
        non_peak_avg_speed = (self.vf + self.v_congestion_cutoff) / 2.0

        # step 4.2 t0 and t3
        t0 = self.t2 - 0.5 * P
        t3 = self.t2 + 0.5 * P

        wt2 = None

        Q_mu = self.lane_based_ultimate_hourly_capacity

        if P > 0.15:
            # work on congested condition
            # step 4.3 compute mu
            Q_mu = min(self.lane_based_ultimate_hourly_capacity, lane_based_D / P)

            # use  as the lower speed compared to 8/15 values for the congested states

            wt2 = link_length_in_1km / self.vt2 - RTT  # in hour

            # step 5 compute gamma parameter is controlled by the maximum queue
            # because q_tw = w*mu =1/4 * gamma (P/2)^4, => 1/vt2 * mu = 1/4 * gamma  * (P/2)^4
            self.Q_gamma = wt2 * 64 * Q_mu / P ** 4

        # QL(t2) = gamma / (4 * 4 * 4) * power(P, 4)
        test_QL_t2 = self.Q_gamma / 64.0 * P ** 4
        test_wt2 = test_QL_t2 / Q_mu

        # L/[(w(t)+RTT_in_hour]
        test_vt2 = link_length_in_1km / (test_wt2 + RTT)

        # ensure
        # ensure diff_v_t2 = 0
        diff = test_vt2 - self.vt2
        td_w = 0

        # default: get_volume_from_speed(td_speed, vf, k_critical, s3_m);
        td_flow = 0

        for t_in_min in range(self.starting_time_in_hour * 60, self.ending_time_in_hour * 60, 5):

            t = t_in_min / 60.0  # t in hour
            td_queue = 0
            td_speed = 0

            # within congestion duration P
            if t0 <= t <= t3:
                # 1/4*gamma*(t-t0)^2(t-t3)^2
                td_queue = 0.25 * self.Q_gamma * (t - t0) ** 2 * (t - t3) ** 2
                td_w = td_queue / max(0.001, Q_mu)
                # L/[(w(t)+RTT_in_hour]
                td_speed = link_length_in_1km / (td_w + RTT)

            elif t < t0:  # outside
                td_queue = 0
                factor = (t - self.starting_time_in_hour) / max(0.001, t0 - self.starting_time_in_hour)
                td_speed = (1 - factor) * self.vf + factor * max(self.v_congestion_cutoff, avg_queue_speed)

            else:  # t> t3
                td_queue = 0
                factor = (t - self.t3) / max(0.001, self.ending_time_in_hour - self.t3)
                td_speed = (1 - factor) * max(self.v_congestion_cutoff, avg_queue_speed) + (factor) * self.vf

            t_interval = t_in_min / 5

            model_speed[t_interval] = td_speed
            est_volume_per_hour_per_lane[t_interval] = td_flow

            if td_speed < self.vf * 0.5:
                self.Severe_Congestion_P += 5.0 / 60  # 5 min interval

        return avg_travel_time
