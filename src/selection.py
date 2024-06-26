import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt
from kneed import KneeLocator
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def Select_PCA(features, critical_value=0.9, plot=True, title=None, show=True):
    if plot and title is None:
        logger.error("Missing parameters for plotting PCA analysis, skipping...")
        plot = False

    pca = PCA()
    pca.fit(features)

    # 计算解释方差比率
    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_explained_variance = np.cumsum(explained_variance_ratio)
    n_components = np.argmax(cumulative_explained_variance >= critical_value) + 1

    if plot:
        # 可视化累积方差解释比例
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(cumulative_explained_variance) + 1), 
                 cumulative_explained_variance, marker='o', linestyle='--')
        plt.axvline(x=n_components, color='r', linestyle='--')
        plt.xlabel('Number of Principal Components')
        plt.ylabel('Cumulative Explained Variance')
        plt.title(title)
        plt.grid()
        if show:
            plt.show()
        else:
            plt.savefig(f'result/selection/{title.replace(" ", "_")}.png')

    # 打印解释方差比率和累积解释方差比率
    print("Explained variance ratio per principal component:")
    print(explained_variance_ratio)
    print("Cumulative explained variance ratio:")
    print(cumulative_explained_variance)
    
    return n_components

    
def Select_CA(features, method='linkage', plot=True, title=None, show=True):
    if plot and title is None:
        logger.error("Missing parameters for plotting CA analysis, skipping...")
        plot = False
        
    if method == 'linkage':
        # 使用层次聚类确定最佳簇数量
        Z = linkage(features, method='ward')
        plt.figure(figsize=(12, 8))
        dendrogram(Z)
        plt.title(title)
        plt.xlabel('Sample ID')
        plt.ylabel('Distance')
        plt.show()
        
    
    elif method == 'kmeans':
        # 使用肘部法则确定最佳K值
        inertia = []
        K_range = range(1, 15)
        for k in K_range:
            kmeans = KMeans(n_clusters=k, random_state=72)
            kmeans.fit(features)
            inertia.append(kmeans.inertia_)

        normalized_inertia = [i / len(features) for i in inertia]

        kneedle = KneeLocator(K_range, normalized_inertia, curve='convex', direction='decreasing')
        optimal_k = kneedle.elbow

        if plot:
            plt.figure(figsize=(8, 5))
            plt.plot(K_range, normalized_inertia, 'bo-')
            plt.xlabel('Number of clusters (K)')
            plt.ylabel('Inertia')
            plt.title(title)
            plt.axvline(x=optimal_k, color='r', linestyle='--')
            if show:
                plt.show()
            else:
                plt.savefig(f'result/selection/{title.replace(" ", "_")}.png')

        print(f'The optimal number of clusters is {optimal_k}')
        
        return optimal_k
