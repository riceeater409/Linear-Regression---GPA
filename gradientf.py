#Gradient Descent for Linear Regression
#Author: Luc Le
import numpy as np


#Create function	
def gradient_descent(x, y, theta, alpha, iterations):
	#Finding size of data for loop.
	m = len(y)
	i = 0
	cost_function = []
	b = []
	#Calculating the cost, then updating bias base off cost function
	for a in range(iterations):
		predicted_y = np.dot(x, theta)
		Y = np.transpose(predicted_y)-y	
		theta = theta - np.transpose(alpha/m*np.dot(Y,x))
		while i < m:
			square_errors = np.square(np.dot(x[i],theta)-y[i])
			b.append(square_errors)
			i += 1
		sum_of_square_errors = sum(b)
		cost = sum_of_square_errors/(2*m)
		cost_function.append(cost)	
	
	#Create user input to find out predicted GPA for user.
	features = []
	sleep = input("Enter how many hours you sleep on average a day..")
	study = input("Enter how many hours you study a week..")
	test = input("From a scale of 0 to 1 rate your test taking skill, 1 being good and 0 being poor.")
	concentration = input("From a scale of 1 to 5 rate your concentration ability, 1 being poor and 5 being great.")
	New_Sleep_Rate = float(sleep)
	New_Studying_Rate = float(study)
	New_Tests_Rate = float(test)
	New_Concentration_Rate = float(concentration)
	features.append(New_Sleep_Rate)
	features.append(New_Studying_Rate)
	features.append(New_Tests_Rate)
	features.append(New_Concentration_Rate)
	print(features)
	GPA =  np.dot(features,theta)
	#Value might exceed 4.0 so we cap it.
	if GPA >= 4.0:
		print("Your GPA is 4.0.")
	else:
		print("Your GPA is ",GPA)
	
	

