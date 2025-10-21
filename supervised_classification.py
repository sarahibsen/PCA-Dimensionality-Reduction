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
from sklearn.svm import SVC


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

#print(indian_pines.data.shape) # checking after the conversion to make sure it converted successfully
scaler = StandardScaler()
# Scaled data to be used in the analysis
x_scaled_iris = scaler.fit_transform(x_iris)
x_scaled_ip = scaler.fit_transform(indian_pines.data)

# For the Confusion Matrix [This function is good and will be used]
def calculate_classwise_metrics(cm, class_names):
    metrics = []
    num_classes = cm.shape[0]
    for i in range(num_classes):
        tp = cm[i, i]
        fn = np.sum(cm[i, :]) - tp
        fp = np.sum(cm[:, i]) - tp
        tn = np.sum(cm) - (tp + fn + fp)
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        metrics.append({
            'Class': class_names[i],
            'Sensitivity': f"{sensitivity:.3f}",
            'Specificity': f"{specificity:.3f}"
        })
    return pd.DataFrame(metrics)

def plot_results(results, dataset_name, reduction_method, filename):
    training_sizes_percent = [10, 20, 30, 40, 50]
    plt.figure(figsize=(10, 6))
    for model_name, accuracies in results.items():
        plt.plot(training_sizes_percent, accuracies, marker='o', linestyle='-', label=model_name)
    title_part = f"with {reduction_method}" if reduction_method != "None" else "without Dimensionality Reduction"
    plt.title(f'Figure: Classification Accuracy on {dataset_name} Dataset {title_part}')
    plt.xlabel('Training Size (%)')
    plt.ylabel('Overall Classification Accuracy')
    plt.xticks(training_sizes_percent)
    plt.ylim(0, 1.05) 
    plt.grid(True)
    plt.legend()
    plt.savefig(filename, dpi=300)
    print(f"Plot saved as {filename}")
    plt.close()

def save_table_as_png(df, title, filename):
    fig, ax = plt.subplots(figsize=(12, 16)) # Increased figure size for larger tables
    ax.axis('off') 
    the_table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        loc='center',
        cellLoc='center',
        colWidths=[0.2, 0.2, 0.1, 0.1] # Adjust column widths
    )
    the_table.auto_set_font_size(False)
    the_table.set_fontsize(10)
    the_table.scale(1.2, 1.2)
    plt.title(title, fontsize=16, pad=20)
    fig.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight', pad_inches=0.1)
    print(f"Table saved as {filename}")
    plt.close(fig)

# 'global' declarations'
gnb_model = GaussianNB()
linear_svc = SVC(kernel='linear', random_state=125)
rbf_svc = SVC(kernel='rbf', random_state=125)
classifiers = { "Naive Bayes": gnb_model, "Linear SVM": linear_svc, "RBF SVM": rbf_svc }
train_sizes = [0.1, 0.2, 0.3, 0.4, 0.5]

# Result storage for plots
results_iris_pca, results_iris_lda, results_iris_none = ({name: [] for name in classifiers.keys()} for _ in range(3))
results_ip_pca, results_ip_lda, results_ip_none = ({name: [] for name in classifiers.keys()} for _ in range(3))

# Result storage for tables (list of dataframes)
metrics_list_pca, metrics_list_lda, metrics_list_none = ([], [], [])

# Component numbers
K_IRIS = 2
K_IP = 10 # Example value, replace with your justified choice

# ==============================================================================
# 2a.i) CLASSIFICATION WITH DIMENSIONALITY REDUCTION
# ==============================================================================
print("\n--- Running Experiments for Iris Dataset ---")
# --- Iris PCA ---
for train_fraction in train_sizes:
    x_train, x_test, y_train, y_test = train_test_split(x_scaled_iris, y_iris, train_size=train_fraction, random_state=125, stratify=y_iris)
    pca = PCA(n_components=K_IRIS)
    x_train_r, x_test_r = pca.fit_transform(x_train), pca.transform(x_test)
    for name, model in classifiers.items():
        model.fit(x_train_r, y_train)
        results_iris_pca[name].append(accuracy_score(y_test, model.predict(x_test_r)))

# --- Iris LDA ---
n_components_lda_iris = min(len(np.unique(y_iris)) - 1, x_scaled_iris.shape[1])
for train_fraction in train_sizes:
    x_train, x_test, y_train, y_test = train_test_split(x_scaled_iris, y_iris, train_size=train_fraction, random_state=125, stratify=y_iris)
    lda = LinearDiscriminantAnalysis(n_components=n_components_lda_iris)
    x_train_r, x_test_r = lda.fit_transform(x_train, y_train), lda.transform(x_test)
    for name, model in classifiers.items():
        model.fit(x_train_r, y_train)
        results_iris_lda[name].append(accuracy_score(y_test, model.predict(x_test_r)))

print("\n--- Running Experiments for Indian Pines Dataset ---")
# --- Indian Pines PCA ---
for train_fraction in train_sizes:
    x_train, x_test, y_train, y_test = train_test_split(x_scaled_ip, y_final, train_size=train_fraction, random_state=125, stratify=y_final)
    pca = PCA(n_components=K_IP)
    x_train_r, x_test_r = pca.fit_transform(x_train), pca.transform(x_test)
    for name, model in classifiers.items():
        model.fit(x_train_r, y_train)
        y_pred = model.predict(x_test_r)
        results_ip_pca[name].append(accuracy_score(y_test, y_pred))
        if train_fraction == 0.4:
            cm = confusion_matrix(y_test, y_pred, labels=np.unique(y_final))
            metrics_df = calculate_classwise_metrics(cm, indian_pines.target_names)
            metrics_df['Classifier'] = name
            metrics_list_pca.append(metrics_df)

# --- Indian Pines LDA ---
n_components_lda_ip = min(len(np.unique(y_final)) - 1, x_scaled_ip.shape[1])
for train_fraction in train_sizes:
    x_train, x_test, y_train, y_test = train_test_split(x_scaled_ip, y_final, train_size=train_fraction, random_state=125, stratify=y_final)
    lda = LinearDiscriminantAnalysis(n_components=n_components_lda_ip)
    x_train_r, x_test_r = lda.fit_transform(x_train, y_train), lda.transform(x_test)
    for name, model in classifiers.items():
        model.fit(x_train_r, y_train)
        y_pred = model.predict(x_test_r)
        results_ip_lda[name].append(accuracy_score(y_test, y_pred))
        if train_fraction == 0.4:
            cm = confusion_matrix(y_test, y_pred, labels=np.unique(y_final))
            metrics_df = calculate_classwise_metrics(cm, indian_pines.target_names)
            metrics_df['Classifier'] = name
            metrics_list_lda.append(metrics_df)

# ==============================================================================
# 2a.ii) CLASSIFICATION WITHOUT DIMENSIONALITY REDUCTION
# ==============================================================================
print("\n--- Running Experiments without Dimensionality Reduction ---")
# --- Iris No Reduction ---
for train_fraction in train_sizes:
    x_train, x_test, y_train, y_test = train_test_split(x_scaled_iris, y_iris, train_size=train_fraction, random_state=125, stratify=y_iris)
    for name, model in classifiers.items():
        model.fit(x_train, y_train)
        results_iris_none[name].append(accuracy_score(y_test, model.predict(x_test)))

# --- Indian Pines No Reduction ---
for train_fraction in train_sizes:
    x_train, x_test, y_train, y_test = train_test_split(x_scaled_ip, y_final, train_size=train_fraction, random_state=125, stratify=y_final)
    for name, model in classifiers.items():
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        results_ip_none[name].append(accuracy_score(y_test, y_pred))
        if train_fraction == 0.4:
            cm = confusion_matrix(y_test, y_pred, labels=np.unique(y_final))
            metrics_df = calculate_classwise_metrics(cm, indian_pines.target_names)
            metrics_df['Classifier'] = name
            metrics_list_none.append(metrics_df)

# ==============================================================================
# 2a.iii) PROVIDE PLOTS AND TABLES (SAVED AS PNG)
# ==============================================================================
print("\n--- Saving All Plots and Tables as PNG Files ---")
# Save all 6 plots
plot_results(results_iris_pca, 'Iris', 'PCA', 'figure1_iris_accuracy_pca.png')
plot_results(results_iris_lda, 'Iris', 'LDA', 'figure2_iris_accuracy_lda.png')
plot_results(results_iris_none, 'Iris', 'None', 'figure3_iris_accuracy_none.png')
plot_results(results_ip_pca, 'Indian Pines', 'PCA', 'figure4_ip_accuracy_pca.png')
plot_results(results_ip_lda, 'Indian Pines', 'LDA', 'figure5_ip_accuracy_lda.png')
plot_results(results_ip_none, 'Indian Pines', 'None', 'figure6_ip_accuracy_none.png')

# Combine and save the 3 tables
table_ip_pca = pd.concat(metrics_list_pca, ignore_index=True)
table_ip_pca = table_ip_pca[['Classifier', 'Class', 'Sensitivity', 'Specificity']]
save_table_as_png(
    table_ip_pca,
    'Table 1: Class-wise Metrics for Indian Pines (PCA)\n(40% Training Size)',
    'table1_ip_metrics_pca.png'
)

table_ip_lda = pd.concat(metrics_list_lda, ignore_index=True)
table_ip_lda = table_ip_lda[['Classifier', 'Class', 'Sensitivity', 'Specificity']]
save_table_as_png(
    table_ip_lda,
    'Table 2: Class-wise Metrics for Indian Pines (LDA)\n(40% Training Size)',
    'table2_ip_metrics_lda.png'
)

table_ip_none = pd.concat(metrics_list_none, ignore_index=True)
table_ip_none = table_ip_none[['Classifier', 'Class', 'Sensitivity', 'Specificity']]
save_table_as_png(
    table_ip_none,
    'Table 3: Class-wise Metrics (No Reduction)\n(40% Training Size)',
    'table3_ip_metrics_none.png'
)