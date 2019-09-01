# Artificial Neural Network

# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
import tensorflow as tf

# Importing the dataset
dataset= pd.read_csv('Churn_Modelling.csv')
x= dataset.iloc[:,3:13].values
y= dataset.iloc[:, 13].values
dataframe=pd.DataFrame(x)

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder= LabelEncoder()
x[:,1]= labelencoder.fit_transform(x[:,1])
x[:,2]= labelencoder.fit_transform(x[:,2])
onehotencoder= OneHotEncoder(categorical_features= [1] )
x= onehotencoder.fit_transform(x).toarray()
x= x[:,1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test= train_test_split(x, y, test_size= 0.2, random_state= 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc= StandardScaler()
x_train= sc.fit_transform(x_train)
x_test= sc.fit_transform(x_test)

# Part 2 - Now let's make the ANN!

# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense

# Initialising the ANN
classifier= Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(units=6, kernel_initializer='uniform', activation= 'relu', input_dim=11))

# Adding the second hidden layer
classifier.add(Dense(units=6, kernel_initializer='uniform', activation= 'relu'))

# Adding the output layer
classifier.add(Dense(units= 1, kernel_initializer='uniform',activation='sigmoid'))


# Compiling the ANN
classifier.compile(optimizer= 'adam', loss='binary_crossentropy', metrics=['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(x_train, y_train, batch_size=10, epochs=20)

# Predicting the Test set results
y_pred= classifier.predict(x_test)
y_pred= (y_pred> 0.5)


# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm= confusion_matrix(y_test, y_pred)

#to calculate the accuracy
from sklearn.metrics import accuracy_score
accuracy= accuracy_score(y_test, y_pred)

#to get the weights(all independent variable or node associated with each node of hidden layer)
weights= classifier.get_weights()

#to get the summary of the code
summary= classifier.summary()


#import KerasRegressor but something is wrong
'''# Initialising the ANN
regressor = Sequential()
 
# Adding the input layer and the first hidden layer
regressor.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 11))
 
# Adding the second hidden layer
regressor.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))
 
# Adding the output layer
regressor.add(Dense(output_dim = 1, init = 'uniform'))
 
# Compiling the ANN
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')'''















  


