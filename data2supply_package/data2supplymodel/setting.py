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

FACTYPE_DICT = {1: 'Fwy', 2: 'Arterial', 4: 'Arterial', 6: 'Arterial', 9: 'Arterial'}
AREATYPE_DICT = {1: 'CBD', 2: 'Outlying CBD', 3: 'Mixed Urban', 4: 'Suburban', 5: 'Rural'}
