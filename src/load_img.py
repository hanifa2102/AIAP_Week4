#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 15:08:31 2018

@author: hanifa
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
#import tarfile,urllib,os
#from sklearn.model_selection import cross_val_score
#from sklearn.neighbors import NearestNeighbors



class LoadImage:
    def __init__(self):
        print('')
        
    def download_data():
        '''Untested for now'''
        url=''
        file_stream=urllib.request.urlopen(url)
        tar_file=tarfile.open(fileobj=file_stream,mode='r|gz')
        tar_file.extract(path='data/')
        tar_file.close()

    def unpickle(self,file):
     '''Load byte data from file'''
     with open(file, 'rb') as f:
         data = pickle.load(f, encoding='latin-1')
         return data
     
    def visualisedata(self,i):
        my_dict={0:'airplane',1:'automobile',2:'bird',3:'cat',4:'deer',
                       5:'dog',6:'frog',7:'horse',8:'ship',9:'truck'}
        print('--------------------')
        print(my_dict[self.train_labels[i]])
        print('--------------------')
        plt.imshow(self.train_data_vis[i])
        plt.show()
        
    def load_unpickledata(self):
        
#        data_dir='../img/cifar-10-batches-py'
        data_dir='img/cifar-10-batches-py'
        train_data = None
        train_labels = []
        
        for i in range(1, 6):
            data_dic = self.unpickle(data_dir + "/data_batch_{}".format(i))
            if i == 1:
                train_data = data_dic['data']
            else:
                train_data = np.vstack((train_data, data_dic['data']))
            train_labels += data_dic['labels']
        
        test_data_dic = self.unpickle(data_dir + "/test_batch")
        test_data = test_data_dic['data']
        test_labels = test_data_dic['labels']
        
        self.train_data=train_data
        self.train_labels=train_labels
        self.test_data=test_data
        self.test_labels=test_labels
        
        train_data = train_data.reshape((len(train_data), 3, 32, 32))
        self.train_data_vis = np.rollaxis(train_data, 1, 4)

    def getData(self):
        self.train_data=np.array(self.train_data)
        self.train_labels=np.array(self.train_labels).reshape(-1,1)
        self.test_data=np.array(self.test_data)
        self.test_labels=np.array(self.test_labels).reshape(-1,1)
        
        return self.train_data,self.train_labels,self.test_data,self.test_labels
    

#
        
#one=LoadImage()
#one.load_unpickledata()
#one.visualisedata(33)
#train_data,train_labels,test_data,test_labels=one.getData()
#
#train_data=train_data[0:300]
#train_labels=train_labels[0:300]
#
#from sklearn.neighbors import KNeighborsClassifier
#from sklearn.pipeline import make_pipeline
#knn_pipeline = make_pipeline(KNeighborsClassifier(n_neighbors=10,p=1))
###clf = svm.SVC(kernel='linear', C=1)
#x=cross_val_score(knn_pipeline, train_data, train_labels.ravel(), cv=5,n_jobs=-1)








    
    


    
    






#