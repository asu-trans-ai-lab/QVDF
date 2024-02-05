# In[0] 
import pandas as pd
import os 
import sqlite3
from .setting import *


def convertLaneToLinkMeasurement(measurement_type='single_day',measurement_file='lane_measurement.csv',output_measurement_folder='./'):
    print('Create Data Base...')
    print('Delete null line in lane measurement file...')
    if measurement_type not in MEASUREMENT_TYPE:
        all_list=','.join(MEASUREMENT_TYPE)
        print('measurement_file ',measurement_type,' is not included in the list:',all_list,'...')
        exit()

    data_df=pd.read_csv(measurement_file,encoding='UTF-8')
    data_df.dropna(axis=0, how='any', inplace=True)
    data_df= data_df.drop(data_df[data_df['link_id']=='None'].index)
    data_df= data_df.drop(data_df[data_df['lanes']==-1].index)
    data_df= data_df.drop(data_df[data_df['from_node_id']=='None'].index)
    data_df= data_df.drop(data_df[data_df['to_node_id']=='None'].index)
    data_df= data_df.drop(data_df[data_df['geometry']=='None'].index)
    data_df['volume']=data_df['volume_per_lane']*data_df['lanes']

    print('connect sqlite database...')
    dbPath = output_measurement_folder+'dataset.db'
    if os.path.exists(dbPath):
        os.remove(dbPath)

    conn = sqlite3.connect(dbPath)
    curs = conn.cursor()

    print('create TABLE in database...')

    curs.execute (""" CREATE TABLE lane_measurement (
                    link_id varchar(10) not null,
                    from_node_id int not null,
                    to_node_id int not null,
                    dir_flag real,
                    lane_name text,
                    lanes int not null,
                    length real not null,
                    FT int not null,
                    AT int not null,
                    volume_per_lane real not null,
                    volume real not null,
                    speed real not null,
                    date varchar(20) not null,
                    time varchar(10) not null, 
                    geometry varchar(20)); """)

    print('import lane measurement file into TABLE...')
    data_df.to_sql("lane_measurement", conn, if_exists='append', index=False)

    print('generate link measurement TABLE...')

    curs.execute (""" CREATE TABLE singleday_link_measurement as
        SELECT link_id, lanes, length, from_node_id, to_node_id, FT, AT, AVG(volume) as volume, AVG(speed) as speed, date, time, geometry
        FROM lane_measurement
        GROUP BY link_id, time; """)


    curs.execute (""" CREATE TABLE multiday_link_measurement as
        SELECT link_id, lanes, length, from_node_id, to_node_id, FT, AT, AVG(volume) as volume, AVG(speed) as speed, date, time,geometry
        FROM lane_measurement
        GROUP BY link_id, time, date; """)


    print('export link measurement file from database to csv')
    
    if measurement_type =='single_day':
        sql="""SELECT * FROM singleday_link_measurement;"""
    elif measurement_type =='multi_day':
        sql="""SELECT * FROM multiday_link_measurement;"""       
       

    df = pd.read_sql(sql,conn)

    if measurement_type == "single_day":
        df['date']='Representive_day'
    
    df.to_csv(os.path.join(output_measurement_folder,'link_measurement.csv'),index=False)

    print('database.db DONE; link_measurement.csv DONE')