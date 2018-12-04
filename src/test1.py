#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 10:37:46 2018

@author: hanifa
"""

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict
 
   
if __name__ == "__main__":
    dict_1=unpickle("/home/hanifa/workspace/AIAP/AIAP_Week4/img/cifar-10-batches-py/data_batch_1")