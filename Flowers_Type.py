#import the libraries
import pandas as pd
import sklearn


 #Read the data from the file
flower_data=pd.read_csv("flower.csv")
print("The dataset:\n\n",flower_data)


 #Split the class and features data
features_data=flower_data[["sepal_length","petal_length"]]
label_data=flower_data[["class"]]


#import the function to split the data
from sklearn.model_selection import train_test_split 

#Split the data into training and testing
x_train, x_test, y_train, y_test = train_test_split(features_data, label_data, test_size=0.30)

#Import the decision tree classifier
from sklearn.tree import DecisionTreeClassifier

#Set the decision tree attributes
my_decision_tree = DecisionTreeClassifier(max_depth=3)


#Train the model using training data
my_decision_tree.fit(x_train,y_train) 


#Use the Decision Tree trained model to predict the the flower type using  testing features
my_prediction=my_decision_tree.predict(x_test) 
print("\n\nThe actual flower type:\n",y_test.T)
print("\n\nThe predicted flower type:\n",my_prediction)


#Import the accuracy function
from sklearn.metrics import accuracy_score


#find the accuracy by comparing the predicted flower type and the actual type
my_accuracy=accuracy_score(y_test,my_prediction) 
print("\n\nMy decision tree accuracy =",my_accuracy)
