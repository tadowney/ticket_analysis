# ******************************************************************
#                       Imported Libraries 
# ******************************************************************
import pandas as pd
import numpy as np
from collections import Counter 
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RepeatedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import sys
from time import time
from scipy import sparse

# ******************************************************************
#                       Functions 
# ******************************************************************
def LogInfo(s, dump=True):
    if(dump):
        print(s)
    with open("knnSweepLog.txt", "a") as myfile:
        myfile.write(s)
        myfile.write('\n')

# ******************************************************************
#                       Main
# ******************************************************************

kNNList = [1,2,3,5,10,25,50,100,250,500]
distance = [1,2]
w = ['uniform', 'distance']
t0 = time()

with open("knnSweepLog.txt", "w") as myfile:
    myfile.write("Starting KNN.py...\n")
    myfile.write("Begin Time: {}\n".format(t0))

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
'Violation Time', 
]

trainColumns = columns.copy()
trainColumns.remove('Violation Code')

df = pd.read_csv(filepath_or_buffer= 'clean_data2.csv', delimiter=',', header=None, names=columns, low_memory=False)

# Remove column titles in data
df = df[1:][columns]
df = df[:1000]

# One-Hot Encode categorical columns
LogInfo("One-Hot Encoding...")
for col in trainColumns:
    feature = df[col]
    feature = pd.get_dummies(feature,prefix=col)
    df = pd.concat([feature,df], axis=1)
    del df[col]

# Remove violation code from dataframe
vcClass = df['Violation Code'].copy().to_numpy(dtype=int)
del df['Violation Code']

# Vectorize dataframe
LogInfo("Vectorizing...")
samples = df.to_numpy()

nsamp, nfeat = samples.shape

LogInfo("Number of Starting Features: {}".format(nfeat))

nsamp, nfeat = samples.shape

testClassPredicted = []

# Train/Test Split using KNN for classification
X_train, X_test, y_train, y_test = train_test_split(samples, vcClass, test_size=0.2, random_state=42)
    
LogInfo("Training with {} samples & Testing with {} samples".format(len(X_train), len(X_test)))

trainSparse=sparse.csr_matrix(X_train)
testSparse = sparse.csr_matrix(X_test)
for wt in w:
    for dist in distance:
        for k in kNNList:

            # KNN
            neigh = KNeighborsClassifier(n_neighbors=k, p=dist, weights=wt)
            print("Fitting...")
            neigh.fit(trainSparse, y_train)

            print("Predicting...")

            testClassPredicted = neigh.predict(testSparse)

            acc = accuracy_score(y_test, testClassPredicted)
            LogInfo("Accuracy at k={}, p={}, w={}: {}".format(k,dist,wt,acc))

t1 = time()

LogInfo("Time End: {}".format(t1))
LogInfo("Total Runtime: {}".format(t1-t0))

LogInfo("Completed!")

