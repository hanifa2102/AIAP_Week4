#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  6 17:17:48 2018

@author: hanifa
"""
import numpy as np


class MLPTwoLayers:
    def __init__(self,p,layer1,layer2):
        self.p=p
        self.layer1=layer1
        self.layer2=layer2
    def to_string(self):
        print("p=%d and layer1=%d and layer=%d" %(self.p,self.layer1,self.layer2))
    


if __name__ == "__main__":
#    one=MLPTwoLayers(5,100,50)
#    one.to_string()
#    5 points and x1 and x2
    X=np.random.randint(10,size=(5,2))
    y_true=2*X[:,0].reshape(-1,1)+3*X[:,1].reshape(-1,1)+10
    y=y_true+np.random.randint(low=1,high=3,size=(5,1))
    
    #Predict a
    w0=0.1
    w1=0.1
    b=0.1
    
    a=w0*X[:,0].reshape(-1,1)+w1*X[:,1].reshape(-1,1)+b
    
    L=a-y
    
    
    