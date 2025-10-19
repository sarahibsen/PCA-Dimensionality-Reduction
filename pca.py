# author : Sarah I. 
# class : 488-01
# Assignment : Homework 2
# Resources : https://www.askpython.com/python/examples/mat-files-in-python
#            https://builtin.com/machine-learning/pca-in-python
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
# for PCA, plot the explained variance for all of the PCs in the dataset
# plotting the vaiance 
# Iris Dataset
pca_iris_full = PCA()
pca_iris_full.fit(x_scaled_iris)
plt.figure(figsize=(8, 5))
plt.plot(np.arange(1, pca_iris_full.n_components_ + 1), np.cumsum(pca_iris_full.explained_variance_ratio_), marker='o')
plt.xlabel("Number of Principal Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("PCA Explained Variance for Iris Dataset")
plt.grid(True)
plt.savefig('IRS.png')
plt.show()

#  Indian Pines Dataset
pca_ip_full = PCA()
pca_ip_full.fit(x_scaled_ip)
plt.figure(figsize=(8, 5))
plt.plot(np.arange(1, pca_ip_full.n_components_ + 1), np.cumsum(pca_ip_full.explained_variance_ratio_), marker='.')
plt.xlabel("Number of Principal Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("PCA Explained Variance for Indian Pines Dataset")
plt.grid(True)
plt.savefig('IP.png')
plt.show()




# Reduced data visualization using PCA to 2 dimensions--display the new transformmed data
# PCA Visualization for Iris Dataset
pca_iris_2d = PCA(n_components=2)
x_pca_iris = pca_iris_2d.fit_transform(x_scaled_iris)
plt.figure(figsize=(8, 6))
scatter = plt.scatter(x_pca_iris[:, 0], x_pca_iris[:, 1], c=y_iris, cmap='viridis', edgecolor='k')
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA of Iris Dataset (2 Dimensions)")
plt.legend(handles=scatter.legend_elements()[0], labels=list(target_names_iris))
plt.savefig('PCA_IRS.png')
plt.show()

# PCA Visualization for Indian Pines Dataset
pca_ip_2d = PCA(n_components=2)
x_pca_ip = pca_ip_2d.fit_transform(x_scaled_ip)
plt.figure(figsize=(10, 8))
scatter = plt.scatter(x_pca_ip[:, 0], x_pca_ip[:, 1], c=indian_pines.target, cmap='jet', alpha=0.7)
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA of Indian Pines Dataset (2 Dimensions)")

handles, _ = scatter.legend_elements()
plt.legend(handles, indian_pines.target_names, title="Classes") 
plt.savefig('PCA_IP.png')
plt.show()

# Reduce data visualization using LDA to 2 dimensions
# LDA Visualization for Iris Dataset
# n_components for LDA cannot be larger than n_classes - 1
lda_iris = LinearDiscriminantAnalysis(n_components=2)
x_lda_iris = lda_iris.fit_transform(x_scaled_iris, y_iris)
plt.figure(figsize=(8, 6))
scatter = plt.scatter(x_lda_iris[:, 0], x_lda_iris[:, 1], c=y_iris, cmap='viridis', edgecolor='k')
plt.xlabel("Linear Discriminant 1")
plt.ylabel("Linear Discriminant 2")
plt.title("LDA of Iris Dataset (2 Dimensions)")
plt.legend(handles=scatter.legend_elements()[0], labels=list(target_names_iris))
plt.savefig('LDA_IRS.png')
plt.show()


# LDA Visualization for Indian Pines Dataset
lda_ip = LinearDiscriminantAnalysis(n_components=2)
x_lda_ip = lda_ip.fit_transform(x_scaled_ip, y_final) 
plt.figure(figsize=(10, 8)) 
scatter = plt.scatter(x_lda_ip[:, 0], x_lda_ip[:, 1], c=y_final, cmap='jet', edgecolor='none', alpha=0.7)
plt.xlabel("Linear Discriminant 1")
plt.ylabel("Linear Discriminant 2")
plt.title("LDA of Indian Pines Dataset (2 Dimensions)")

handles, _ = scatter.legend_elements()
plt.legend(handles, indian_pines.target_names, title="Classes")
plt.savefig('LDA_IP.png')
plt.show()
