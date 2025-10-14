# author : Sarah I. 
# class : 488-01
# Assignment : Homework 2
# Resources : https://www.askpython.com/python/examples/mat-files-in-python
#            https://builtin.com/machine-learning/pca-in-python
import matplotlib as mp # for plotting 
import pandas as pd
from scipy.io import loadmat
#from sklearn.preprocessing import StandardScalar
from sklearn.decomposition import PCA #to apply our pca 


# load in the indian pines dataset
indian_pines = loadmat('indianR.mat') # has 'X'
x_raw = indian_pines['X']
# load in the ground truth 
groundtruth = loadmat('indian_gth.mat') # has 'gth' 
y_raw = groundtruth['gth']


# indian pines data contains zeros

x_transposed = x_raw.T 
print(x_transposed.shape)
y_flat = y_raw.flatten() # flattening the y to be a 1D array of labels

mask = y_flat != 0 # we want to take out the zeros from both the y and x datasets
x_final = x_transposed[mask]
y_final = y_flat[mask]


# for PCA, plot the explained variance for all of the PCs in the dataset
pca_full = PCA()
pca_full.fit(x_final)

# plotting the vaiance 


# Reduced data visualization using PCA to 2 dimensions--display the new transformmed data


# Reduce data visualization using LDA to 2 dimensions