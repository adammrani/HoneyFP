import pandas as pd
import numpy as np
import glob #for working on many csv files at once
import os
from sklearn.preprocessing import LabelEncoder

path = './dataset'
all_files = glob.glob(os.path.join(path, "*.csv"))

ind_df_list = []

for filename in all_files:
    df = pd.read_csv(filename, header=0)
    #Just to know from which csv file the column came from:
    df['source_file'] = os.path.basename(filename)
    ind_df_list.append(df)

#Concatenate all the dfs
beth_df = pd.concat(ind_df_list, axis=0, ignore_index=True)

print(beth_df.columns.tolist()) #Key Columns in BETH

print("Total rows:", len(beth_df))
print("Normal (evil=0):", (beth_df['evil'] == 0).sum())
print("Anomalous (evil=1):", (beth_df['evil'] == 1).sum())
print(df.head())

FEATURES = ['processName', 'eventName', 'eventId', 'argsNum', 'returnValue', 'args', 'userId',
            'SourceIP', 'DestinationIP', 'DnsQuery',
             'Timestamp', 'hostName', 'SensorId', 'sus', 'evil' ]

df_clean = beth_df[FEATURES].copy()

text_columns = ['processName', 'eventName', 'args', 'hostName', 'SensorId', 'SourceIP', 'DestinationIP', 'DnsQuery', 'Timestamp']
df_clean[text_columns] = df_clean[text_columns].fillna('None')

numeric_columns = ['eventId', 'argsNum', 'returnValue', 'userId']
df_clean[numeric_columns] = df_clean[numeric_columns].fillna(-1)

le = LabelEncoder()
for col in text_columns:
    df_clean[col] = le.fit_transform(df_clean[col].astype(str))

normal_df = df_clean[df_clean['evil'] == 0].copy()
anomalous_df = df_clean[df_clean['evil'] == 1].copy()

print(f"Features ready for ML and Visualization: {FEATURES}")
