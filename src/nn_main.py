from keras.models import Sequential, load_model
from keras.layers import Dense, Input
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import OneHotEncoder
from keras.layers import Dense, LeakyReLU
from keras.utils import plot_model, print_summary
from keras.utils.vis_utils import model_to_dot
from keras import initializers, optimizers, backend
from sklearn.metrics import accuracy_score
from sklearn import tree, svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import SparsePCA
import pickle
import keras
import matplotlib.pyplot as plt
import random
import pandas as pd
import numpy as np 
import csv

def write_file(classification_list):
    out_dir = "results.dat"
    
    with open(out_dir, "w") as f:
        for c in classification_list:
            f.write(str(c) + "\n")
    print("Prediction results written to {}".format(out_dir))

def neural_network():

    f1 = pd.read_csv("clean_data2.csv", low_memory = False)
    

    f1 = f1[['Registration State','Plate Type','Issue Date','Violation Code','Vehicle Body Type','Vehicle Make','Issuing Agency','Violation Precinct',
    'Issuer Precinct','Violation Hour','Violation County','Street Name','Vehicle Color']] 
    
    f = f1.dropna()
    train_thres = int((80/100)*len(f))
    
    data = {}

    label_enc = {}
    for (columnName, columnData) in f.iteritems(): # this performs fit_transform on the easy types
        if columnName == "Violation Code":
            label_enc["Violation Code"] = OneHotEncoder(sparse = False)
            col_data = columnData.values.reshape(-1,1)
            enc_col_data = label_enc["Violation Code"].fit_transform(col_data)
        else:
            enc = OneHotEncoder(sparse = False)
            col_data = columnData.values.reshape(-1,1)
            enc_col_data = enc.fit_transform(col_data)
        data[columnName] = enc_col_data

    zipped_data = [np.concatenate(list(a)).ravel() for a in zip(data['Registration State'],data['Plate Type'],data['Issue Date'],data['Vehicle Body Type'],data['Vehicle Make'],data['Issuing Agency'],data['Violation Precinct'],
    data['Issuer Precinct'],data['Violation Hour'],data['Violation County'], data['Vehicle Color'], data['Street Name'])]

    labels = [np.concatenate(list(a)).ravel() for a in zip(data['Violation Code'])]

    train_data = np.array(zipped_data[:train_thres])
    test_data = np.array(zipped_data[train_thres:])
    train_labels = np.array(labels[:train_thres])
    test_labels = np.array(labels[train_thres:])

    print("Input shape: {}".format(len(train_data[0])))
    print("Output shape: {}".format(train_labels[0].shape))
    model = Sequential()
    model.add(Dense(units = 128, input_dim = len(train_data[0]))) 
    model.add(LeakyReLU())
    model.add(Dense(units = 128, kernel_initializer = initializers.he_uniform(seed = None)))
    model.add(LeakyReLU())
    model.add(Dense(units = 128, kernel_initializer = initializers.he_uniform(seed = None)))
    model.add(LeakyReLU())
    model.add(Dense(units = 128, kernel_initializer = initializers.he_uniform(seed = None)))
    model.add(LeakyReLU())
    model.add(Dense(units = 128, kernel_initializer = initializers.he_uniform(seed = None)))
    model.add(LeakyReLU())
    model.add(Dense(units = 128, kernel_initializer = initializers.he_uniform(seed = None)))
    model.add(LeakyReLU())
    model.add(Dense(units = len(train_labels[0]), activation = 'softmax', kernel_initializer = initializers.he_uniform(seed = None)))
    
    log_file = "best_checkpoint"
    early_stopping = EarlyStopping(monitor='val_categorical_accuracy', mode='max', verbose=1, patience = 30) #sets up early stopping
    checkpoint = ModelCheckpoint(log_file, monitor='val_categorical_accuracy', verbose=1, save_best_only=True, mode='max') #sets up checkpoint
    model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['categorical_accuracy']) #sgd, sigmoid, no last layer
    history = model.fit(train_data, train_labels, batch_size = 32, validation_split = 0.2, epochs = 100, callbacks = [checkpoint, early_stopping])
    
    with open('./train_history', 'wb') as file:
        pickle.dump(history.history, file)

    # Plot training & validation accuracy values
    plt.figure()
    plt.plot(history.history['categorical_accuracy'])
    plt.plot(history.history['val_categorical_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.savefig("accuracy.png")
    #plt.show()

    # Plot training & validation loss values
    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.savefig("loss.png")
    model = load_model("best_checkpoint")
    res = model.evaluate(test_data, test_labels)
    print("Test categorical accuracy is {}".format(res[1]))

    preds = model.predict(test_data)
    preds = label_enc["Violation Code"].inverse_transform(preds)

    return preds

def main():
    preds = neural_network()
    write_file(preds)

if __name__ == "__main__":
    main()