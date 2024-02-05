import csv
import time


class Link:
    def __init__(self, link_id, V, QDF, t2, L, C, vc, vf, mm, a, b):
        self.link_id = link_id
        self.V = float(V)
        self.QDF = float(QDF)
        self.t2 = float(t2)
        self.L = float(L)
        self.C = float(C)
        self.vc = float(vc)
        self.kc = self.C / max(0.0001, self.vc)
        self.vf = float(vf)
        self.mm = float(mm)
        self.a = float(a)
        self.b = float(b)


def CDBCalibration(_NUMBER_OF_SECOND_PER_MIN, assign_period_start_time_in_hour, assign_period_end_time_in_hour
                   , time_interval_in_min, number_of_interval):
    """
    Congestion Demand Based Calibration
    
    Input(link based):
    private:
    (1) Volume during assignment period——V
    (2) QDF factor=Volume/Demand——QDF
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
    with open("link.csv", "r") as fp:
        reader = csv.DictReader(fp)
        for line in reader:
            link = Link(line["link_id"], line["assignment_period_volume"], line["QDF"], line["t2"],
                        line["length"], line["ultimate_capacity"], line["critical_speed"],
                        line["free_flow_speed"], line["flatness_of_curve"], line["cd_alpha"], line["cd_beta"])
            g_link_list.append(link)

    performance_fp = open("link_performance.csv", "w", newline="")
    performance_writer = csv.writer(performance_fp)
    line = ["link_id", "demand", "D/C", "cd_speed", "cd_travel_time", "average_discharge_rate",
            "cd_waiting_time", "queue_demand", "Gamma", "Congestion_duration", "t0", "t3", "t0'", "t3'", "Queue_max"]
    performance_writer.writerow(line)

    td_queue_fp = open("td_queue.csv", "w", newline="")
    td_speed_fp = open("td_speed.csv", "w", newline="")
    td_queue_writer = csv.writer(td_queue_fp)
    td_speed_writer = csv.writer(td_speed_fp)
    line = ["link_id"]
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

    for link in g_link_list:
        if link.V == 0:
            continue
        demand = link.V / link.QDF
        D_C = demand / link.C
        vcd = link.vf / (1 + link.a * pow(D_C, link.b))
        ttcd = link.L / vcd
        mu = vcd * link.kc * pow((pow(link.vf / vcd, link.mCm / 2) - 1), 1 / link.mm)
        wcd = max(ttcd - link.L / link.vc, 0)
        QD = demand
        if wcd == 0:
            QD = 0
        gamma = 120 * mu * wcd * pow(mu / demand, 4)
        P = QD / mu
        t0 = link.t2 - 0.5 * P
        t3 = link.t2 + 0.5 * P
        t0_p = t0
        t3_p = t3
        if P == 0:
            t0_p -= 0.5
            t3_p += 0.5
        Queue_max = 0.25 * gamma * pow((link.t2 - t0), 2) * pow((link.t2 - t3), 2)
        performance_line = [link.link_id, demand, D_C, vcd, ttcd, mu,
                            wcd, QD, gamma, P, t0, t3, t0_p, t3_p, round(Queue_max, 0)]
        performance_writer.writerow(performance_line)

        td_queue_line = [link.link_id]
        td_speed_line = [link.link_id]
        for t in time_in_hour:
            if t0 <= t <= t3:
                td_queue = 0.25 * gamma * pow((t - t0), 2) * pow((t - t3), 2)
            else:
                td_queue = 0
            if P == 0:
                if t < t0_p:
                    td_speed = link.vf - ((link.vf - vcd) / (t0_p - assign_period_start_time_in_hour)) * (t - assign_period_start_time_in_hour)
                elif t < t3_p:
                    td_speed = link.vf - ((link.vf - vcd) / (assign_period_end_time_in_hour - t3_p)) * (assign_period_end_time_in_hour - t)
                else:
                    td_speed = vcd
            else:
                if t < t0_p:
                    td_speed = link.vf - ((link.vf - link.vc) / (t0_p - assign_period_start_time_in_hour)) * (t - assign_period_start_time_in_hour)
                elif t < t3_p:
                    td_speed = link.vf - ((link.vf - link.vc) / (assign_period_end_time_in_hour - t3_p)) * (assign_period_end_time_in_hour - t)
                else:
                    td_speed = td_queue / mu

            td_queue_line.append(round(td_queue, 0))
            td_speed_line.append(td_speed)
        td_queue_writer.writerow(td_queue_line)
        td_speed_writer.writerow(td_speed_line)

    performance_fp.close()
    td_queue_fp.close()
    td_speed_fp.close()


if __name__ == "__main__":
    _NUMBER_OF_SECOND_PER_MIN = 60
    assign_period_start_time_in_hour = 6
    assign_period_end_time_in_hour = 10
    time_interval_in_min = 5
    number_of_interval = int((assign_period_end_time_in_hour - assign_period_start_time_in_hour) * 60 / 5) + 1
    CDBCalibration(_NUMBER_OF_SECOND_PER_MIN, assign_period_start_time_in_hour, assign_period_end_time_in_hour
                   , time_interval_in_min, number_of_interval)
