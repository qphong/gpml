import pandas as pd 
import pickle
import numpy as np 
import sys

import matplotlib.pyplot as plt


loc = pd.read_csv("./mote_locs.txt", delimiter=" ", header=None)
loc.columns = ["moteid", "x", "y"]

df = pd.read_csv("./data.txt", delimiter=" ", header=None)
df.columns = ["date", "time", "epoch", "moteid", "temp", "humidity", "light", "voltage"]


df = df.dropna(axis=0, how="any")

df['hour'] = df['time'].map(
    lambda str: int(str[:str.find(':')])
)
df['minute'] = df['time'].map(
    lambda str: int(str[(str.find(':')+1):str.find(':',str.find(':')+1)])
)

print("computing date_hour_minute feature")
sys.stdout.flush()

df['date_hour_minute'] = df.apply(
    lambda x: 
        str(x['date']) + '_h' + str(x['hour']) + '_m' + str(x['minute']),
    axis=1
)

time_groups = df.groupby(['date_hour_minute'])
keys = time_groups.groups.keys() 

sufficient_date_hour_minutes = []

for k in keys:
    n_unique_moteids = time_groups.get_group(k).moteid.nunique()

    if n_unique_moteids > 35:
        sufficient_date_hour_minutes.append(k)

print("filter by date_hour_minutes")
sys.stdout.flush()
df = df[ df['date_hour_minute'].isin(sufficient_date_hour_minutes) ]

print("save filtered data to file")
sys.stdout.flush()
df.to_csv("./sufficient_sensor_data1.csv", index=False)
