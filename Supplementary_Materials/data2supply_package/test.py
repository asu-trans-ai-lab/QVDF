from data2supplymodel import formating_period_definition
from datetime import datetime
import data2supplymodel as ds
import pandas as pd

# #
period_list = ['0700_2100']
ds.define_demand_period(period_list, measurement_file='link_performance.csv')
# #
ds.define_corridor(corridor_name='I10', link_measurement_file='corridor_measurement.csv')
# # #
# # # # step 0 Define a VDF area
ds.calibrate_fundamental_diagram(ft_list='all', at_list='all', measurement_file='corridor_measurement_I10.csv')

# # # # Step 3 calibration
ds.calibrate_vdf(measurement_file='fd_corridor_measurement_I10.csv', level_of_service=0.7)

ds.generateTimeDependentQueue(link_file='fd_corridor_measurement_I10_training_set.csv')
