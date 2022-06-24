import pandas as pd
import numpy as np
from collections import Counter 

columns = [
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

df = pd.read_csv(filepath_or_buffer= 'clean_data.csv', delimiter=',', header=None, names=columns)
df = df[1:][columns]    # Remove column titles in data

startLen = len(df)
print('Starting DF Length', len(df))

# Remove '99' from Registration State
df = df[df['Registration State'] != '99']

# Remove '999' from Plate Type
df = df[df['Plate Type'] != '999']

# Combine Colors Together
df['Vehicle Color'].replace(['bk'], ['black'], inplace=True)
df['Vehicle Color'].replace(['blk'], ['black'], inplace=True)
df['Vehicle Color'].replace(['bl'], ['blue'], inplace=True)
df['Vehicle Color'].replace(['dkb'], ['blue'], inplace=True)
df['Vehicle Color'].replace(['blu'], ['blue'], inplace=True)
df['Vehicle Color'].replace(['ltb'], ['blue'], inplace=True)
df['Vehicle Color'].replace(['bn'], ['brown'], inplace=True)
df['Vehicle Color'].replace(['brn'], ['brown'], inplace=True)
df['Vehicle Color'].replace(['br'], ['brown'], inplace=True)
df['Vehicle Color'].replace(['gr'], ['green'], inplace=True)
df['Vehicle Color'].replace(['gn'], ['green'], inplace=True)
df['Vehicle Color'].replace(['blg'], ['green'], inplace=True)
df['Vehicle Color'].replace(['grn'], ['green'], inplace=True)
df['Vehicle Color'].replace(['dkg'], ['green'], inplace=True)
df['Vehicle Color'].replace(['ltg'], ['green'], inplace=True)
df['Vehicle Color'].replace(['wh'], ['white'], inplace=True)
df['Vehicle Color'].replace(['wht'], ['white'], inplace=True)
df['Vehicle Color'].replace(['wh'], ['white'], inplace=True)
df['Vehicle Color'].replace(['wt'], ['white'], inplace=True)
df['Vehicle Color'].replace(['rd'], ['red'], inplace=True)
df['Vehicle Color'].replace(['rd/'], ['red'], inplace=True)
df['Vehicle Color'].replace(['mr'], ['red'], inplace=True)
df['Vehicle Color'].replace(['dkr'], ['red'], inplace=True)
df['Vehicle Color'].replace(['tn'], ['tan'], inplace=True)
df['Vehicle Color'].replace(['yw'], ['yellow'], inplace=True)
df['Vehicle Color'].replace(['yello'], ['yellow'], inplace=True)
df['Vehicle Color'].replace(['sl'], ['silver'], inplace=True)
df['Vehicle Color'].replace(['silvr'], ['silver'], inplace=True)
df['Vehicle Color'].replace(['sl'], ['silver'], inplace=True)
df['Vehicle Color'].replace(['sil'], ['silver'], inplace=True)
df['Vehicle Color'].replace(['silve'], ['silver'], inplace=True)
df['Vehicle Color'].replace(['gry'], ['gray'], inplace=True)
df['Vehicle Color'].replace(['gy'], ['gray'], inplace=True)
df['Vehicle Color'].replace(['grey'], ['gray'], inplace=True)
df['Vehicle Color'].replace(['or'], ['orange'], inplace=True)
df['Vehicle Color'].replace(['orang'], ['orange'], inplace=True)
df['Vehicle Color'].replace(['pr'], ['purple'], inplace=True)
df['Vehicle Color'].replace(['purpl'], ['purple'], inplace=True)
df['Vehicle Color'].replace(['laven'], ['purple'], inplace=True)
df['Vehicle Color'].replace(['gl'], ['gold'], inplace=True)
df['Vehicle Color'].replace(['dk/'], ['nan'], inplace=True)
df['Vehicle Color'].replace(['lt/'], ['nan'], inplace=True)
df['Vehicle Color'].replace(['other'], ['nan'], inplace=True)
df['Vehicle Color'].replace(['noc'], ['nan'], inplace=True)

df = df[df['Vehicle Color'] != 'nan']

regState =  df['Vehicle Color']
cnt = Counter(regState)
cntSort = sorted(cnt.items(), key=lambda pair: pair[0], reverse=False)
print(cntSort)

# Convert Violation Time
from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder(sparse = False)
violation_time = df["Violation Time"]
start_morning = 5
end_morning = 11
start_noon = 11
end_noon = 17
start_evening = 17
end_evening = 5

violation_hour = []
for i, time in enumerate(violation_time):
    hour = int(time[0:2])
    if(hour == 12 and time[4] == 'A'):
        hour = 0
    elif(hour != 12 and time[4] == 'P'):
        hour = hour + 12

    if hour >= start_morning and hour < end_morning:
        violation_hour.append("Morning")
    elif hour >= start_noon and hour < end_noon:
        violation_hour.append("Afternoon")
    elif hour >= start_evening or hour < end_evening:
        violation_hour.append("Evening")
        
df["Violation Hour"] = violation_hour
df = df.drop(columns="Violation Time")

endLen = len(df)
print('Ending DF Length', len(df))
print('Rows Removed:', startLen - endLen)

#  write dataframe to csv
df.to_csv('./clean_data2.csv', index=False)


