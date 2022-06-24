import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.decomposition import SparsePCA
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import timeit

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
'Violation County',
'Street Name',
'Vehicle Color',
'Violation Hour']

df = pd.read_csv(filepath_or_buffer= 'clean_data2.csv', delimiter=',', header=None, names=columns, low_memory=False)

# Remove column titles in data
df = df[1:][columns]

print('finished reading the data...')

print('Number of Samples:', len(df))

print('number of features before one hot encoding: ')
print('--------------------------------------------')
print(df.info())


# class feature
y = df['Violation Code']
# one-hot encoding on class feature
y = pd.get_dummies(y, columns=['Violation Code'])

# training features
train_col = columns.copy()
train_col.remove('Violation Code')

# remove class feature from dataframe
del df['Violation Code']

# one-hot encoding on each features of training dataframe
for col in train_col:
    feature = df[col]
    feature = pd.get_dummies(feature,prefix=col)
    df = pd.concat([feature,df], axis=1)
    del df[col]

print('finished one hot encoding...')

print('number of features after one hot encoding: ')
print('--------------------------------------------')
print(df.info())

# training dataframe
X = df

# split data into train and test with ratio of 80% and 20%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)

# starting a timer for calculating the program running time
start = timeit.default_timer()

classifier = DecisionTreeClassifier()
classifier.fit(X_train,y_train)
print('finished creating the model...')

y_pred = classifier.predict(X_test)
print('finished predicting...')


# print(y_pred)
np.savetxt('dt_results.txt', y_pred, fmt="%s")
print('save results to file...')

# metrics
score =  accuracy_score(y_test,y_pred)
msg = 'Accuracy Score:'+ str(score)
print(msg)

#  calculating the program running time
stop = timeit.default_timer()
duration = stop - start
msg = 'Running Time: '+ str(duration)
print(msg)

print('END ...')
