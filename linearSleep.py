#Linear Regression for GPA
#Data structured based on sleep, study time, concentration, and test taking ability.
#Author: Luc Le

import csv
import numpy as np
import gradientf
import random

#Point at file and initiate empty list to parse data
file = 'sleep.csv'
data = []

#Loop through file to parse data
with open(file) as csv_file:
	csv_reader = csv.reader(csv_file, delimiter = ',')
	for A in csv_reader:
		data.append(A)
		
#Find the size of features and data.
size = np.shape(data)
m = size[0]-1
n = size[1]-1
#Create array of data to be worked with based off the size of data
x = np.array(data)[1:m+1,:n]
y = np.array(data)[1:m+1,n]
#Convert data into numeric value type
Y = y.astype(float)
X = x.astype(int)
#Create bias value for machine learning
theta = np.random.rand(n,1)
#Implement machine learning using data, output, bias, learning rate, iteration)
gradientf.gradient_descent(X,Y,theta,.001,10000)


# ~ gradientf.compute_cost(X,Y,theta)

