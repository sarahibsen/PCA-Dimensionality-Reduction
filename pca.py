import matplotlib as mp # for plotting 
import pandas as pd
from scipy.io import loadmat
#from sklearn.preprocessing import StandardScalar
#from sklearn.decomposition import PCA #to apply our pca 


# load in the indian pines dataset
indian_pines = loadmat('indianR.mat') # has 'X'
#print(indian_pines) #sanity check

# load in the ground truth 
groundtruth = loadmat('indian_gth.mat') # has 'gth' 
#print(groundtruth) #sanity check 

# indian pines data contains zeros

pines_list = [[element for element in upperElement] for upperElement in indian_pines['X']]
g_list = [[element for element in upperElement] for upperElement in groundtruth['gth']]

pines_list =[i for i in pines_list if i!=0]

# converting the dataset into a pandas data frame
newData = list(zip(pines_list[0], pines_list[1])) # both x and y of the data set
columns = ['pines_x', 'pines_y']
df = pd.DataFrame(newData, columns=columns)
print(df.head())
# for PCA, plot the explained variance for all of the PCs in the dataset