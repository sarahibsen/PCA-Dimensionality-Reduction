# author : Sarah I. 
# class : 488-01
# Assignment : Homework 2 pt 2 : )
# Resources : https://www.askpython.com/python/examples/mat-files-in-python
#            https://builtin.com/machine-learning/pca-in-python
#            https://www.kaggle.com/code/prashant111/naive-bayes-classifier-in-python for naive bayes
#            https://www.datacamp.com/tutorial/naive-bayes-scikit-learn # for the actual classification 
import matplotlib.pyplot as plt # for plotting 
import numpy as np
import pandas as pd

from matplotlib.figure import Figure
from scipy.io import loadmat
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA #to apply our pca 
from sklearn.utils import Bunch
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis 
# New imports for second project
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
)


iris = datasets.load_iris()
x_iris = iris.data
y_iris = iris.target
target_names_iris = iris.target_names

# load in the indian pines dataset
indian_pines = loadmat('indianR.mat') # has 'X'
x_raw = indian_pines['X']
# load in the ground truth 
groundtruth = loadmat('indian_gth.mat') # has 'gth' 
y_raw = groundtruth['gth']
# will have to structure it similarly to the iris dataset 
x_transposed = x_raw.T 
print(x_transposed.shape)
y_flat = y_raw.flatten() # flattening the y to be a 1D array of labels

mask = y_flat != 0 # we want to take out the zeros from both the y and x datasets
x_final = x_transposed[mask]
y_final = y_flat[mask]

# Going to manually define the target names 
target_names_ip_list = [
    'Alfalfa', 'Corn-notill', 'Corn-min', 'Corn', 'Grass-pasture',
    'Grass-trees', 'Grass-pasture-mowed', 'Hay-windrowed', 'Oats',
    'Soybean-notill', 'Soybean-min', 'Soybean-clean', 'Wheat', 'Woods',
    'Buildings-Grass-Trees-Drives', 'Stone-Steel-Towers'
]
# now converting the dict object to a Bunch object (what hte iris dataset is)
indian_pines = Bunch(
    data = x_final,
    target = y_final,
    target_names = np.array(target_names_ip_list),
    DESCR = "Indian Pines Dataset converted to a Bunch OBJ"
)

print(indian_pines.data.shape) # checking after the conversion to make sure it converted successfully
scaler = StandardScaler()
x_scaled_iris = scaler.fit_transform(x_iris)
x_scaled_ip = scaler.fit_transform(indian_pines.data)

# perform supervised classification on the iris and indian datasets using Naive Bayes and support vector machines (with RBF and Linear kernel)
# classifiers for training sizes = {10%, 20%, 30%, 40%, 50%} for each of the below cases // We have multiple features for both datasets 

# Indian Pines 
# going to iterate over the different training sizes that was asked of me : ' ) in a loop 
for train_fraction in training_sizes:
    
    # Calculate the corresponding test size fraction
    test_fraction = 1.0 - train_fraction
    x_train, x_test, y_train, y_test = train_test_split(
        x_final, 
        y_final, 
        train_size=train_fraction,  
        random_state=125
    )
    
    print(f"--- Running for Training Size: {train_fraction*100:.0f}% ---")
    print(f"x_train shape: {x_train.shape}")
    print(f"x_test shape: {x_test.shape}")
    print("-" * 30)

    # we will be testing three different models : RBF, Guaussian, and Linear
    gb_model = GaussianNB()
    gb_model.fit(x_train, y_train)


    # https://www.geeksforgeeks.org/python/creating-linear-kernel-svm-in-python/
    

