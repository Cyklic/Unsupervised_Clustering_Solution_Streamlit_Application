
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np

def Cluster(df, income, spending):
    # Let' visualize these clusters
    fig, ax = plt.subplots(figsize=(8,5))
    sns.scatterplot(x='Annual_Income', y = 'Spending_Score', data=df, hue=df['Cluster'], palette='colorblind')
    plt.scatter(income, spending, color='red', s=200, edgecolors='black', label='User')
    plt.xlabel('Annual Income')
    plt.xlabel('Spending Score')
    plt.title('Mall Customer Grouping')
    plt.savefig("Clusters.png")
    plt.show()

def Silhouette_plot(wss):
    # Now, plot the silhouette plot
    plt.figure(figsize=(10, 6))
    wss.plot(x='cluster', y='Silhouette_Score')
    plt.xlabel('No. of clusters')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Plot')
    plt.legend()
    
    plt.savefig("Silhouette_Score.png")
    plt.show()
    

# def plot_confusion_matrix(y_true, y_pred, classes, normalize=False, title='Confusion Matrix'):
#     """
#     Plot the confusion matrix for the given true and predicted labels.
    
#     Args:
#         y_true (numpy.ndarray): Array of true labels.
#         y_pred (numpy.ndarray): Array of predicted labels.
#         classes (list): List of class labels.
#         normalize (bool, optional): Whether to normalize the confusion matrix. Default is False.
#         title (str, optional): Title for the plot. Default is 'Confusion Matrix'.
#     """
#     cm = confusion_matrix(y_true, y_pred)
#     if normalize:
#         cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

#     plt.figure(figsize=(8, 6))
#     sns.heatmap(cm, annot=True, cmap='Blues', xticklabels=classes, yticklabels=classes)
#     plt.xlabel('Predicted', fontsize=12)
#     plt.ylabel('Actual', fontsize=12)
#     plt.title(title, fontsize=16)
#     plt.show()