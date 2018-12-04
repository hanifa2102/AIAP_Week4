#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 15:08:31 2018

@author: hanifa
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
import tarfile,urllib,os
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import NearestNeighbors


class Person:
  def __init__(self, name, age):
    self.name = name
    self.age = age
    
    

def unpickle(file):
 '''Load byte data from file'''
 with open(file, 'rb') as f:
     data = pickle.load(f, encoding='latin-1')
     return data


def download_data():
    url=''
    file_stream=urllib.request.urlopen(url)
    tar_file=tarfile.open(fileobj=file_stream,mode='r|gz')
    tar_file.extract(path='data/')
    tar_file.close()
    
    
def load_unpickledata():
    
    data_dir='../img/cifar-10-batches-py'
    train_data = None
    train_labels = []
    
    for i in range(1, 6):
        data_dic = unpickle(data_dir + "/data_batch_{}".format(i))
        if i == 1:
            train_data = data_dic['data']
        else:
            train_data = np.vstack((train_data, data_dic['data']))
        train_labels += data_dic['labels']
    
    test_data_dic = unpickle(data_dir + "/test_batch")
    test_data = test_data_dic['data']
    test_labels = test_data_dic['labels']
    

#    train_labels = np.array(train_labels)
#
#
#    test_labels = np.array(test_labels)
    
    return train_data,train_labels,test_data,test_labels


def visualisedata(data,i):
#    train_data = train_data.reshape((len(train_data), 3, 32, 32))
#    train_data = np.rollaxis(train_data, 1, 4)
##    test_data = test_data.reshape((len(test_data), 3, 32, 32))
##    test_data = np.rollaxis(test_data, 1, 4)
    
    data = data.reshape((len(data), 3, 32, 32))
    
    plt.imshow(data[i])
    plt.show()
    
    

train_data,train_labels,test_data,test_labels= load_unpickledata()

train_data=np.array(train_data)
train_labels=np.array(train_labels).reshape(-1,1)
test_data=np.array(test_data)
test_labels=np.array(test_labels).reshape(-1,1)



train_data=train_data[0:300]
train_labels=train_labels[0:300]

from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
knn_pipeline = make_pipeline(KNeighborsClassifier(n_neighbors=10,p=1))
##clf = svm.SVC(kernel='linear', C=1)
x=cross_val_score(knn_pipeline, train_data, train_labels.ravel(), cv=5,verbose=1,n_jobs=-1)








    
    


    
    






#data_dir='../img/cifar-10-batches-py'
#train_data = None
#train_labels = []
#
#for i in range(1, 6):
#    data_dic = unpickle(data_dir + "/data_batch_{}".format(i))
#    if i == 1:
#        train_data = data_dic['data']
#    else:
#        train_data = np.vstack((train_data, data_dic['data']))
#        train_labels += data_dic['labels']
#
#test_data_dic = unpickle(data_dir + "/test_batch")
#test_data = test_data_dic['data']
#test_labels = test_data_dic['labels']
#
#train_labels = np.array(train_labels)
#
#
#train_data = train_data.reshape((len(train_data), 3, 32, 32))
#train_data = np.rollaxis(train_data, 1, 4)
#train_labels = np.array(train_labels)
#
#test_data = test_data.reshape((len(test_data), 3, 32, 32))
#test_data = np.rollaxis(test_data, 1, 4)
#test_labels = np.array(test_labels)
#
#plt.imshow(train_data[12])
#plt.show()
    