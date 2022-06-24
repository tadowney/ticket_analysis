# This file applies thresholds to all the relevant columns in small.csv and outputs a preprocessed data file called clean_data.csv

import os, sys


columns = [
'Summons Number', 
'Plate ID',
'Registration State',
'Plate Type',
'Issue Date',
'Violation Code',
'Vehicle Body Type',
'Vehicle Make',
'Issuing Agency',
'Street Code1',
'Street Code2',
'Street Code3',
'Vehicle Expiration Date',
'Violation Location',
'Violation Precinct',
'Issuer Precinct',
'Issuer Code',
'Issuer Command',
'Issuer Squad',
'Violation Time',
'Time First Observed',
'Violation County',
'Violation In Front Of Or Opposite',
'House Number',
'Street Name',
'Intersecting Street',
'Date First Observed',
'Law Section',
'Sub Division',
'Violation Legal Code',
'Days Parking In Effect',
'From Hours In Effect',
'To Hours In Effect',
'Vehicle Color',
'Unregistered Vehicle',
'Vehicle Year',
'Meter Number',
'Feet From Curb',
'Violation Post Code',
'Violation Description',
'No Standing or Stopping Violation',
'Hydrant Violation',
'Double Parking Violation',
'Latitude',
'Longitude',
'Community Board',
'Community Council',
'Census Tract',
'BIN',
'BBL',
'NTA'
]
len(columns)


import pandas as pd
import numpy as np

df = pd.read_csv(filepath_or_buffer= './small.csv', delimiter=',', header=None, names=columns)
df.tail()

df_edited = df[1:][columns]

df_edited.head()

columns_edited = [
'Registration State',
'Plate Type',
'Issue Date',
'Violation Code',
'Vehicle Body Type',
'Vehicle Make',
'Issuing Agency',
'Violation Precinct',
'Issuer Precinct',
'Violation Time', 
'Violation County',
'Street Name',
'Vehicle Color']

df_edited = df_edited[columns_edited]
df_edited.tail()

# thresholds
thres = 100
vc = df_edited['Violation Code'].value_counts()
u  = [i not in set(vc[vc<thres].index) for i in df_edited['Violation Code']]
df_edited = df_edited[u]

# thresholds
thres = 100
vc = df_edited['Registration State'].value_counts()
u  = [i not in set(vc[vc<thres].index) for i in df_edited['Registration State']]
df_edited = df_edited[u]

# thresholds
thres = 100
vc = df_edited['Plate Type'].value_counts()
u  = [i not in set(vc[vc<thres].index) for i in df_edited['Plate Type']]
df_edited = df_edited[u]

# convert date to day of week
dates = df_edited['Issue Date']
df_date_adj = pd.to_datetime(dates)
df_date_adj

df_edited['Issue Date'] = df_date_adj.dt.day_name()
df_edited

print('data frame length', len(df_edited))

# thresholds
thres = 5000
vc = df_edited['Vehicle Body Type'].value_counts()
u  = [i not in set(vc[vc<thres].index) for i in df_edited['Vehicle Body Type']]
df_edited = df_edited[u]

print('data frame length', len(df_edited))

# thresholds
thres = 220
vc = df_edited['Vehicle Make'].value_counts()
u  = [i not in set(vc[vc<thres].index) for i in df_edited['Vehicle Make']]
df_edited = df_edited[u]

print('data frame length', len(df_edited))

# thresholds
thres = 1000
vc = df_edited['Issuing Agency'].value_counts()
u  = [i not in set(vc[vc<thres].index) for i in df_edited['Issuing Agency']]
df_edited = df_edited[u]

print('data frame length', len(df_edited))

# thresholds
thres = 100
vc = df_edited['Violation Precinct'].value_counts()
u  = [i not in set(vc[vc<thres].index) for i in df_edited['Violation Precinct']]
df_edited = df_edited[u]

print('data frame length', len(df_edited))



# thresholds
thres = 100
vc = df_edited['Issuer Precinct'].value_counts()
u  = [i not in set(vc[vc<thres].index) for i in df_edited['Issuer Precinct']]
df_edited = df_edited[u]

print('data frame length', len(df_edited))

# thresholds
thres = 20
vc = df_edited['Violation Time'].value_counts()
u  = [i not in set(vc[vc<thres].index) for i in df_edited['Violation Time']]
df_edited = df_edited[u]

print('data frame length', len(df_edited))

# thresholds
thres = 2000
vc = df_edited['Violation County'].value_counts()
u  = [i not in set(vc[vc<thres].index) for i in df_edited['Violation County']]
df_edited = df_edited[u]

print('data frame length', len(df_edited))

# thresholds
thres = 100
vc = df_edited['Street Name'].value_counts()
u  = [i not in set(vc[vc<thres].index) for i in df_edited['Street Name']]
df_edited = df_edited[u]

print('data frame length', len(df_edited))

# thresholds
thres = 200
vc = df_edited['Vehicle Color'].value_counts()
u  = [i not in set(vc[vc<thres].index) for i in df_edited['Vehicle Color']]
df_edited = df_edited[u]

print('data frame length', len(df_edited))

# editing colors
df_edited['Vehicle Color'] = df_edited['Vehicle Color'].str.lower()

#  remove nans
df_edited = df_edited.dropna()

print('data frame length', len(df_edited))

#  write dataframe to csv
df_edited.to_csv('./clean_data.csv', index=False)