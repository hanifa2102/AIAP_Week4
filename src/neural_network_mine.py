#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  7 14:04:31 2018

@author: hanifa
"""

import numpy as np

def sigmoid(z):
    return 1.0/(1+ np.exp(-z))

def sigmoid_derivative(z):
    return z * (1.0 - z)

class NeuralNetwork:
    def __init__(self,x,y):
        self.x=x
        self.y=y
#        X.shape (4,3)
        self.weights1=np.random.random((X.shape[1],4))
        self.layer1_z=np.zeros((4,4))
        self.layer1_a=np.zeros((4,4))
        
        self.weights2=np.random.random((4,1))
        self.layer2_z=np.zeros((4,1))
        self.layer2_z=np.zeros((4,1))
        
    def forward_prop(self):
        self.layer1_z=np.dot(self.x,self.weights1)
        self.layer1_a=sigmoid(self.layer1_z)
        
        self.layer2_z=np.dot(self.layer1_a,self.weights2)
        self.layer2_a=sigmoid(self.layer2_z)
        
                self.layer1 = sigmoid(np.dot(self.input, self.weights1))
        self.output = sigmoid(np.dot(self.layer1, self.weights2))
        
    def backward_prop(self):
        self.dweights2 = np.dot((2*(self.y - self.layer2_a)),sigmoid_derivative(self.layer2_z),self.layer1_a.T)
#        d_weights2 = np.dot(self.layer1.T, (2*(self.y - self.output) * sigmoid_derivative(self.output)))
        self.dweights1 = np.dot(
                    2*self.
                )
        
        d_weights2 = np.dot(self.layer1.T, (2*(self.y - self.output) * sigmoid_derivative(self.output)))
        d_weights1 = np.dot(self.input.T,  
                            (np.dot(2*(self.y - self.output) * sigmoid_derivative(self.output), self.weights2.T)
                                        * sigmoid_derivative(self.layer1))
                            )
        
        print(self.dweights2.shape)
        
        
if __name__ == "__main__":
    X = np.array([[0,0,1],
                  [0,1,1],
                  [1,0,1],
                  [1,1,1]])
    y = np.array([[0],[1],[1],[0]]) 
    
    
    a=NeuralNetwork(X,y)
    a.forward_prop()
    a.backward_prop()
    
    tt=a.layer2_a
    
    #2nd layer is output
    
    