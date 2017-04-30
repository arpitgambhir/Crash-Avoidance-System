import os
import numpy as np
import pandas as pd
import glob
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from time import time

# Defining the 2 directories
dir1 = 'D:\Newdata\Files'  # Parent Directory containing all the folders with PER values 0 to 0.9
dir2 = 'D:\Newdata\Files\PER0'  #Directory for Ground Truth

# Defining the arrays so that we can append the value in them
directories = []
mean_arr = []
Grnd_Trth_df2 = []
camplinear = []
time_arr = []

os.chdir(dir2)    # Changing the directory to dir2, which contains files for PER = 0
for files in glob.glob("*.csv"):    # Walking through all the files in the directory "dir2"
    data1 = pd.read_csv(files, delimiter=',', header=None)    # Using pandas to read all the files in the directory "dir2"
    cars = pd.DataFrame(data1)    # Creating dataframes of all the files in the directory
    Groundtruth = cars.iloc[:, 4]    # Selecting only the 5th column of the files (I will treat this as the ground truth)
    Grnt_Trth_df = pd.DataFrame(Groundtruth)    # Creating a dataframe of only this 1 column for each file in the directory.
    Grnd_Trth_df2.append(Grnt_Trth_df)    # Appending all the dataframes of 823 files in this array

'''
I am defining a function whose input is a directory. This directory takes one directory at a time and passes it on the function as an input.
Which directory is to pass when is defined below this function. 
'''

def algorithm(directory):    # Defining a function named algorithm
    acc2 = []
    dataframe2 = []
    acc4 = []
    for files2 in glob.glob("*.csv"):   # Walking through all the files in directory, named "directory" (Function's input)
        data = pd.read_csv(files2, delimiter=',', header=None)    # Reading all the files from "directory"
        dataframe = pd.DataFrame(data)    # Creating a dataframe of all these files
        dataframe2.append(dataframe)    # Appending all these dataframes of 823 files to this array

    'Now I have to concatenate dataframes of all these 823 files with the dataframes of 823 files obtained earlier with only the ground truth value in them'

    for p in range(0, 823, 1):
        datafram3 = pd.concat([dataframe2[p], Grnd_Trth_df2[p]], axis=1, ignore_index=True)    # Concatenating 1st dataframe of both the arrays
                                                                                         # Concatenating 2nd dataframe of both the arrays
                                                                                         # And so on till the 823 dataframes are concatenated
        'So now we get 823 dataframes which have total 6 columns'

        features = datafram3.iloc[:, 0:4]    # First 4 columns are defined as input features
        output = datafram3.iloc[:, 4]    # 5th column is defined as the output
        grnd = datafram3.iloc[:, 5]    # 6th Column is the Ground Truth, with which we have to compare to get the accuracy

        'Now I have to split the data into training and testing set. Testing size is taken as 30% of the total data'

        features_train, features_test, output_train, output_test, grnd_train, grnd_test = train_test_split(features, output, grnd, test_size=0.3, random_state=5)

        acc = accuracy_score(grnd_test, output_test)    # Calculating the accuracies of CAMPLinear algorithm for each PER value
        acc2.append(acc)                                # Appending all the accuracies to this array

        a = len(np.unique(output_train))    # Checking the number of classes in output, i.e. 0 and 1 are two classes.
        if a == 1:                          # If there is only 1 class, we need to skip this file because for SVM we need at least 2 classes
            continue                        # and some files do have only 1 class, so it will produce an error
        else:                               # If there are more than 1 class, we can perform machine learning on those files

            ''' 
            Defining a classifier, just change this classifier to test different algorithms.
            For Support Vector Machine: clf = SVC()
            For Neural Networks: clf = MLPClassifier()
            For Random Forest: clf = RandomForestClassifier()
            For Naive Bayes: GaussianNB()
            '''

            clf = SVC()
            clf.fit(features_train, output_train)       # Fitting the algorithm
            predicted = clf.predict(features_test)      # Predicting the output for the test set
            acc3 = accuracy_score(grnd_test, predicted)      # Calculating the accuracies with respect to Ground Truth, i.e. PER = 0
            acc4.append(acc3)                            # Appending the accuracies to this array
    mean_arr.append(np.mean(acc4))                      # Appending the mean accuracies of my algorithm for each PER value to this array
    camplinear.append(np.mean(acc2))                    # Appending the mean accuracies of CAMPLinear algorithm for each PER value
    time_arr.append(round(time() - t1, 3))              # Calculating and appending the time taken for each PER value for a particular algorithm

'''
In the following lines of code, I am defining an absolute path to the parent directory, which contains all the sub directories of each PER value.
From PER 0 to PER 9. Program will list all the subdirectories in the parent directories and store them.
'''

directories = [os.path.abspath(x[0]) for x in os.walk(dir1)]
directories.remove(os.path.abspath(dir1))

'Here I used a for loop to loop through all the directories'

for i in directories:
    os.chdir(i)     # Changing the directory to ith directory in directories array
    t1 = time()     # Used to calculate the time
    algorithm(i)    # Calling the above defined function with i as the input to the function

print "Mean accuracies of CAMPLinear: ", camplinear     # Printing the mean accuracies of CAMPLinear Algorithm for each PER value
print "Mean accuracies of algorithm that I implemented: ", mean_arr     # Printing the mean accuracies of my algorithms for each PER value

PER = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]      # Defining the values of PER to plot the graphs

'Plotting the graph of various algorithms, and CAMPLinear algorithm Vs. PER '
plt.figure()
plt.plot(PER, mean_arr, label='My Algorithm', marker='8', color='red')
plt.plot(PER, camplinear, label='CAMPLinear', marker='8', color='blue')
plt.title('Accuracy Vs. PER Plot')
plt.xlabel('PER')
plt.ylabel('Accuracy')
plt.legend(loc=1)
plt.show()

'Plotting the graph of times taken for each algorithm for each PER value'
plt.figure()
plt.plot(PER, time_arr, label='My algorithm', marker='8', color='red')
plt.title('Time taken Vs. PER Plot')
plt.xlabel('PER')
plt.ylabel('Accuracy')
plt.legend(loc=1)
plt.show()

#=====================================================THE END==========================================================#

'''The code takes time because of the large dataset. Please be patient. Thank you.'''
