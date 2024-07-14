import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.cluster.hierarchy import fcluster

# Load the Iris dataset
iris = sns.load_dataset('iris')

# Bar Plot
plt.figure(figsize=(10, 6))
sns.barplot(x='species', y='sepal_length', data=iris, ci=None)
plt.title('Bar Plot of Sepal Length by Species')
plt.xlabel('Species')
plt.ylabel('Sepal Length (cm)')
plt.show()

# Line Plot
plt.figure(figsize=(10, 6))
sns.lineplot(data=iris.drop(columns='species'))
plt.title('Line Plot of Iris Dataset Features')
plt.xlabel('Sample Index')
plt.ylabel('Measurement (cm)')
plt.legend(labels=iris.columns[:-1])
plt.show()

# Scatter Plot
plt.figure(figsize=(10, 6))
sns.scatterplot(x='sepal_length', y='sepal_width', hue='species', data=iris)
plt.title('Scatter Plot of Sepal Length vs Sepal Width')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Sepal Width (cm)')
plt.legend(title='Species')
plt.show()

# Histogram
plt.figure(figsize=(10, 6))
sns.histplot(iris['petal_length'], kde=True, bins=30)
plt.title('Histogram of Petal Length')
plt.xlabel('Petal Length (cm)')
plt.ylabel('Frequency')
plt.show()

# Dendrogram
# Perform hierarchical/agglomerative clustering
linked = linkage(iris.drop(columns='species'), method='ward')

# Create a color threshold to color the clusters
threshold = 7

plt.figure(figsize=(10, 6))
dendrogram(linked,
           orientation='top',
           labels=iris.index,
           distance_sort='descending',
           show_leaf_counts=True,
           color_threshold=threshold)
plt.title('Dendrogram of Iris Dataset')
plt.xlabel('Sample Index')
plt.ylabel('Euclidean Distance')
plt.show()
