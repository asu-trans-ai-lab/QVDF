# In[0] 
import pandas as pd
import os 
import time
import datetime
import sqlite3
import numpy as np
from .setting import *
import csv

def data_filtering(intrix_measurement_raw_file,intrix_measurement_file,START_DATE,END_DATE,weekdayflag=1):
    data_df=pd.read_csv(intrix_measurement_raw_file,encoding='UTF-8')
    #translate the time stamp to a datetime series
    data_df['measurement_tstamp']=pd.to_datetime(data_df.measurement_tstamp)
    data_df = data_df[(data_df['measurement_tstamp']<END_DATE)&(data_df['measurement_tstamp']>=START_DATE)].copy()
    # delete the weekend data
    if weekdayflag == 1: 
        data_df=data_df[data_df.measurement_tstamp.dt.weekday<5].copy() # only consider weekday
    data_df.to_csv(intrix_measurement_file,index=False)
    print('filter the data...')

def _create_data_base(data_df,measurement_type,output_measurement_folder='./'):
    data_df.dropna(axis=0, how='any', inplace=True)


    print('connect sqlite database...')
    dbPath = output_measurement_folder+'dataset.db'
    if os.path.exists(dbPath):
        os.remove(dbPath)

    conn = sqlite3.connect(dbPath)
    curs = conn.cursor()

    print('create TABLE in database...')
    curs.execute (""" CREATE TABLE tmc_link_measurement (
                    tmc varchar(10) not null,
                    measurement_tstamp text not null,
                    speed real not null,
                    reference_speed real not null,
                    level_of_service real not null, 
                    date real not null,
                    time text not null,
                    road text not null,
                    miles real not null,
                    direction text not null,
                    road_order int not null,
                    FT int ,
                    AT int ,
                    lanes int,
                    geometry, text); """)

    print('import tmc_link_measurement file into TABLE...')
    data_df.to_sql("tmc_link_measurement", conn, if_exists='append', index=False)


    if measurement_type == "single_day":
        curs.execute (""" CREATE TABLE mean_tmc_link_measurement as
            SELECT tmc, AVG(speed) as speed, reference_speed as reference_speed,level_of_service as level_of_service, time, road, miles as miles, direction,road_order,FT,AT,lanes,geometry
            FROM tmc_link_measurement
            GROUP BY tmc, time; """)


        sql="""SELECT * FROM mean_tmc_link_measurement;"""

        

        data_df = pd.read_sql(sql,conn)

        data_df['date']='Representive_day'
    
    #df.to_csv(os.path.join(output_measurement_folder,'tmc_link_measurement.csv'),index=False)

    print('database.db DONE')
    return data_df






def convertInrixToLinkMeasurement(intrix_measurement_file,tmc_identification_file,time_interval_in_min,measurement_type='single_day'):
    data_df=pd.read_csv(intrix_measurement_file,encoding='UTF-8')
    #data_df=data_df[['tmc','measurement_tstamp','speed','reference_speed']]
    data_df.rename(columns={'xd_id':'tmc'}, inplace=True)
    data_df=data_df[['tmc','measurement_tstamp','speed','reference_speed']]
    #data_df['reference_critical_speed']=data_df['reference_speed']*0.75
    tmc_mean_speed_dict={}
    tmc_reference_speed_dict={}
    #tmc_reference_critical_speed_dict={}
    tmc_group=data_df.groupby(['tmc'])
    for tmc_index,tmc_set in tmc_group:
        tmc_mean_speed_dict[tmc_index]=tmc_set.speed.mean()
        tmc_reference_speed_dict[tmc_index]=tmc_set.reference_speed.mean()
        #tmc_reference_critical_speed_dict[tmc_index]=tmc_set.reference_critical_speed.mean()
    
    tmc_identification_df=pd.read_csv(tmc_identification_file,encoding='UTF-8')
    #tmc_identification_df['geometry']=tmc_identification_df.apply(lambda x: "LINESTRING("+str(x.start_longitude)+' '+str(x.start_latitude)+", "+str(x.end_longitude)+' '+str(x.end_latitude)+")",axis=1)
    tmc_identification_df['reference_speed']=tmc_identification_df.apply(lambda x: tmc_reference_speed_dict[x.tmc],axis=1)
    tmc_identification_df['mean_speed']=tmc_identification_df.apply(lambda x: tmc_mean_speed_dict.setdefault(x.tmc,'non_matched'),axis=1)
    tmc_identification_df['free_speed_net']=tmc_identification_df['free_speed']
    tmc_identification_df['free_speed']=tmc_identification_df.apply(lambda x: np.ceil(x.reference_speed+2) if x.mean_speed>x.free_speed_net else x.free_speed_net,axis=1)
    tmc_identification_df.to_csv(tmc_identification_file,index=False)
    dict_type_list=['road','corridor','miles','direction','FT','AT','road_order','lanes','free_speed','capacity','LOS']
    dict_list={}
    for dict_type in dict_type_list:
        dict_list[dict_type]=dict(zip(tmc_identification_df['tmc'],tmc_identification_df[dict_type]))
    #print(1)

    
    chunk_list=[]
    chunk_data_df=pd.read_csv(intrix_measurement_file,chunksize=100000)
    # Each chunk is in df format with 100000 rows
    iter=1
    for sub_data_df in chunk_data_df:
        print('Loop'+str(iter)+'...')
        sub_data_df=sub_data_df[['tmc','measurement_tstamp','speed']]
        # perform data filtering 
        sub_data_df=_joinInrixField(dict_type_list,sub_data_df,dict_list,lookup_field='tmc')       
        #sub_data_df=_joinInrixField(['road'],sub_data_df,tmc_identification_df,lookup_field='tmc')       
        
        # Once the data appendence is done, append the chunk to list
        sub_data_df=sub_data_df[sub_data_df.road.isna()==0]
        print('start joining time fields..')
        sub_data_df['measurement_tstamp']=pd.to_datetime(sub_data_df.measurement_tstamp)
        sub_data_df['date']=pd.to_datetime(sub_data_df.measurement_tstamp).dt.date
        sub_data_df['time']=pd.to_datetime(sub_data_df.measurement_tstamp).dt.time
        sub_data_df=convertTimeToHHMM(sub_data_df,time_interval_in_min=5)   
        chunk_list.append(sub_data_df)
        iter+=1
    data_df = pd.concat(chunk_list)
    print('join inrix fields Done..')


    data_df.rename(columns={'free_speed':'free_flow_speed','capacity':'ultimate_capacity'}, inplace=True)

    #data_df=_joinInrixField(['miles','direction','FT','AT','road_order','lanes'],data_df,tmc_identification_df,lookup_field='tmc')
    # filter the records that cannot match 
    #data_df=data_df[data_df.road.isna()==0]

    # data_df['measurement_tstamp']=pd.to_datetime(data_df.measurement_tstamp)

    # data_df['date']=pd.to_datetime(data_df.measurement_tstamp).dt.date
    # data_df['time']=pd.to_datetime(data_df.measurement_tstamp).dt.time

    #data_df=data_df[data_df['measurement_tstamp'].dt.hour.isin(np.arange(0,24))]

    # if measurement_type == "single_day":
    #     data_df=_create_data_base(data_df,measurement_type)

    # data_df=convertTimeToHHMM(data_df,time_interval_in_min=5)

    print('corridor_name:',data_df.corridor.unique())
    print('road_name:',data_df.road.unique())
    print('direction:',data_df.direction.unique())
    data_df.to_csv('tmc_link_measurement.csv',index=False)



def _joinInrixField(dict_type_list,data_df,dict_list,lookup_field='tmc'):
    for dict_type in dict_type_list:
        time_start =time.time()
        #data_df=data_df.merge(tmc_identification_df[[lookup_field, dict_type]], 'left')
        data_df[dict_type]=data_df.apply(lambda x:dict_list[dict_type].setdefault(x[lookup_field],np.nan),axis=1)
        data_df.drop_duplicates()
        data_df=data_df.dropna()
        time_end=time.time()
        print('Add new field name', dict_type ,'in the intrix_measurement_file, CPU time:',time_end-time_start)
    return data_df

def convertTimeToHHMM(data_df,time_interval_in_min,output_folder='./'):
    #data_df=pd.read_csv(measurement_file,encoding='UTF-8')
    print ('convert timestamp to hhmm format ....')
    time_start=time.time()
    time_df=pd.DataFrame()
    time_df['time_st']=pd.to_datetime(data_df.time.astype(str))
    time_df['time_ed']=pd.to_datetime(data_df.time.astype(str))+datetime.timedelta(minutes=time_interval_in_min)
    time_df['hour']=time_df['time_st'].dt.hour
    time_df['minute']=time_df['time_st'].dt.minute
    time_df['hour_s']=time_df['time_ed'].dt.hour
    time_df['minute_s']=time_df['time_ed'].dt.minute
    print('please waiting ....')

    time_df['hour']=time_df['hour'].apply(lambda x: str(0)+str(int(x)) if len(str(int(x)))==1 else str(int(x)))
    time_df['minute']=time_df['minute'].apply(lambda x: str(0)+str(int(x)) if len(str(int(x)))==1 else str(int(x)))
    time_df['hour_s']=time_df['hour_s'].apply(lambda x: str(0)+str(int(x)) if len(str(int(x)))==1 else str(int(x)))
    time_df['minute_s']=time_df['minute_s'].apply(lambda x: str(0)+str(int(x)) if len(str(int(x)))==1 else str(int(x)))
    print('please waiting ....')

    data_df['time_period']=time_df['hour']+time_df['minute']+"_"+time_df['hour_s']+time_df['minute_s']
    time_end=time.time()
    print('CPU time:',time_end-time_start,'s...\n')
    return data_df


# def generateLinkMeasurement(intrix_measurement_file,tmc_identification_file):
#     convertInrixToLinkMeasurement(intrix_measurement_file,tmc_identification_file)
#     generateMeanSpeedData(measurement_file='tmc_link_measurement.csv')
#     convertTimeToHHMM(time_interval_in_min=5,measurement_file='tmc_link_measurement.csv')