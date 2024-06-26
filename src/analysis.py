import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import CCA
from scipy.cluster.hierarchy import linkage, fcluster
import seaborn as sns
import matplotlib.pyplot as plt
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def Analysis_Corr(X, Y, plot=False, x_columns=None, y_columns=None, title=None, xlabel=None, ylabel=None, show=True):
    if plot and (x_columns is None or y_columns is None or title is None or xlabel is None or ylabel is None):
        logger.error("Missing parameters for plotting correlation analysis, skipping...")
        plot = False
        
    correlation_matrix = np.corrcoef(X.T, Y.T)
    correlation_matrix = correlation_matrix[X.shape[1]:, :X.shape[1]]

    if plot:
        # 绘制热力图
        plt.figure(figsize=(10, 6))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', xticklabels=x_columns, yticklabels=y_columns)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        if show:
            plt.show()
        else:
            plt.savefig(f'result/analysis/{title.replace(" ", "_")}.png')
        
    return correlation_matrix
    
    

def Analysis_PCA(features, n_components, plot=False, feature_columns=None, title=None, show=True):
    if plot and (feature_columns is None or title is None):
        logger.error("Missing parameters for plotting PCA analysis, skipping...")
        plot = False
    
    pca = PCA(n_components=n_components)
    pca_features = pca.fit_transform(features)

    if plot:
        if n_components == 2:
            # 可视化前两个主成分
            plt.figure(figsize=(12, 8))
            plt.scatter(pca_features[:, 0], pca_features[:, 1], marker='o')
            plt.xlabel('PCA Feature 1 (%.2f%% Variance)' % (pca.explained_variance_ratio_[0] * 100))
            plt.ylabel('PCA Feature 2 (%.2f%% Variance)' % (pca.explained_variance_ratio_[1] * 100))
            plt.title(title)

            # 可视化每个特征在 PCA 空间中的权重
            loading_vectors = pca.components_.T * np.sqrt(pca.explained_variance_)
            for i, feature in enumerate(feature_columns):
                plt.arrow(0, 0, loading_vectors[i, 0] * 8, loading_vectors[i, 1] * 8, 
                        color='red', alpha=0.5, head_width=0.05, head_length=0.1)
                plt.text(loading_vectors[i, 0] * 8.5 + 3, loading_vectors[i, 1] * 8.5, 
                        feature, color='black', ha='center', va='center', fontsize=12)
            if show:
                plt.show()
            else:
                plt.savefig(f'result/analysis/{title.replace(" ", "_")}.png')
            
        elif n_components > 2:
            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(111, projection='3d')
            
            ax.scatter(pca_features[:, 0], pca_features[:, 1], pca_features[:, 2], c='b', marker='o')
            ax.set_xlabel('PCA Feature 1 (%.2f%% Variance)' % (pca.explained_variance_ratio_[0] * 100))
            ax.set_ylabel('PCA Feature 2 (%.2f%% Variance)' % (pca.explained_variance_ratio_[1] * 100))
            ax.set_zlabel('PCA Feature 3 (%.2f%% Variance)' % (pca.explained_variance_ratio_[2] * 100))
            ax.set_title(title)

            k_arrow = 0.8
            k_text = 2.5
            loading_vectors = pca.components_.T * np.sqrt(pca.explained_variance_)
            for i, feature in enumerate(feature_columns):
                ax.quiver(0, 0, 0, loading_vectors[i, 0] * k_arrow, loading_vectors[i, 1] * k_arrow, loading_vectors[i, 2] * k_arrow, 
                        color='r', alpha=0.6, length=3)
                ax.text(loading_vectors[i, 0] * k_text, loading_vectors[i, 1] * k_text, loading_vectors[i, 2] * k_text, 
                        feature, color='black', ha='center', va='center', fontsize=10)
            if show:
                plt.show()
            else:
                plt.savefig(f'result/analysis/{title.replace(" ", "_")}.png')
        
        # 可视化所有主成分的系数
        plt.figure(figsize=(12, 8))
        bar_width = 0.2
        x = np.arange(len(feature_columns))

        for i in range(n_components):
            plt.bar(x + i * bar_width, pca.components_[i], bar_width, label=f'PCA Component {i + 1}')

        plt.xlabel('Feature')
        plt.ylabel('Coefficient')
        plt.title(f'Coefficients for {title}')
        plt.xticks(x + bar_width * (n_components - 1) / 2, feature_columns)

        for i in range(n_components):
            for j in range(len(feature_columns)):
                plt.text(x[j] + i * bar_width, pca.components_[i, j] + 0.02 if pca.components_[i, j] > 0 else pca.components_[i, j] - 0.05, 
                         f'{pca.components_[i, j]:.2f}', ha='center', va='bottom')
        plt.legend()
        if show:
            plt.show()
        else:
            plt.savefig(f'result/analysis/{title.replace(" ", "_")}_coefficients.png')        
    
    return pca, pca_features
    
    
def Analysis_CA(features, optimal_k, method='linkage', plot=False, pca=None, feature_columns=None, title=None, show=True):
    logger = logging.getLogger(__name__)
    if plot and (pca is None or feature_columns is None or title is None):
        logger.error("Missing parameters for plotting CA analysis, skipping...")
        plot = False
    
    if method == 'linkage':
        Z = linkage(features, method='ward')
        clusters = fcluster(Z, optimal_k, criterion='maxclust')
        labels = clusters - 1
    elif method == 'kmeans':  
        kmeans = KMeans(n_clusters=optimal_k, random_state=72)
        kmeans.fit(features)
        labels = kmeans.labels_
    
    if plot:
        pca_features = pca.fit_transform(features)
        if pca_features.shape[1] == 2:
            # 可视化聚类结果
            plt.figure(figsize=(10, 6))
            scatter = plt.scatter(pca_features[:, 0], pca_features[:, 1], c=labels, cmap='viridis', marker='o')
            plt.xlabel('PCA Feature 1 (%.2f%% Variance)' % (pca.explained_variance_ratio_[0] * 100))
            plt.ylabel('PCA Feature 2 (%.2f%% Variance)' % (pca.explained_variance_ratio_[1] * 100))
            plt.title(title)
            plt.colorbar(scatter, label='Cluster')

            # 可视化每个特征在 PCA 空间中的权重
            loading_vectors = pca.components_.T * np.sqrt(pca.explained_variance_)
            for i, feature in enumerate(feature_columns):
                plt.arrow(0, 0, loading_vectors[i, 0] * 12, loading_vectors[i, 1] * 12, 
                        color='red', alpha=0.5, head_width=0.05, head_length=0.1)
                plt.text(loading_vectors[i, 0] * 12.5 + 4, loading_vectors[i, 1] * 12.5, 
                        feature, color='black', ha='center', va='center', fontsize=12)
            if show:
                plt.show()
            else:
                plt.savefig(f'result/analysis/{title.replace(" ", "_")}.png')
        
        elif pca_features.shape[1] > 2:
            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(111, projection='3d')
            
            scatter = ax.scatter(pca_features[:, 0], pca_features[:, 1], pca_features[:, 2], c=labels, cmap='viridis', marker='o')
            ax.set_xlabel('PCA Feature 1 (%.2f%% Variance)' % (pca.explained_variance_ratio_[0] * 100))
            ax.set_ylabel('PCA Feature 2 (%.2f%% Variance)' % (pca.explained_variance_ratio_[1] * 100))
            ax.set_zlabel('PCA Feature 3 (%.2f%% Variance)' % (pca.explained_variance_ratio_[2] * 100))
            ax.set_title(title)
            fig.colorbar(scatter, label='Cluster')

            k_arrow = 0.8
            k_text = 2
            loading_vectors = pca.components_.T * np.sqrt(pca.explained_variance_)
            for i, feature in enumerate(feature_columns):
                ax.quiver(0, 0, 0, loading_vectors[i, 0] * k_arrow, loading_vectors[i, 1] * k_arrow, loading_vectors[i, 2] * k_arrow, 
                        color='r', alpha=0.6, length=3)
                ax.text(loading_vectors[i, 0] * k_text, loading_vectors[i, 1] * k_text, loading_vectors[i, 2] * k_text, 
                        feature, color='black', ha='center', va='center', fontsize=10)

            if show:
                plt.show()
            else:
                plt.savefig(f'result/analysis/{title.replace(" ", "_")}.png')
        
    return kmeans


def Analysis_CCA(X, Y, n_components, plot=False, x_columns=None, y_columns=None, title=None, show=True):
    if plot and (x_columns is None or y_columns is None or title is None):
        logger.error("Missing parameters for plotting CCA analysis, skipping...")
        plot = False
        
    cca = CCA(n_components=n_components)
    cca.fit(X, Y)
    X_c, Y_c = cca.transform(X, Y)
    
    # 打印所有典型载荷
    for i in range(n_components):
        print(f"Canonical component {i + 1} X loadings: {cca.x_loadings_[:, i]}")
        print(f"Canonical component {i + 1} Y loadings: {cca.y_loadings_[:, i]}")
        print("First canonical correlation:", np.corrcoef(X_c[:, i], Y_c[:, i])[0, 1])
    
    if plot:
        # # 提取典型载荷
        # U_coefficients = cca.x_loadings_[:, 0]
        # V_coefficients = cca.y_loadings_[:, 0]
        
        # # 可视化典型载荷
        # plt.figure(figsize=(12, 6))

        # plt.subplot(1, 2, 1)
        # plt.barh(x_columns, U_coefficients, color='b')
        # plt.xlabel('Coefficient Value')
        # plt.title(f'{title} loadings for X')

        # plt.subplot(1, 2, 2)
        # plt.barh(y_columns, V_coefficients, color='r')
        # plt.xlabel('Coefficient Value')
        # plt.title(f'{title} loadings for Y')

        # plt.tight_layout()
        # if show:
        #     plt.show()
        # else:
        #     plt.savefig(f'result/analysis/{title.replace(" ", "_")}_loadings.png')
            
        # 可视化典型相关变量关系
        plt.figure(figsize=(10, 6))
        color = ['b', 'r', 'g', 'y', 'c', 'm', 'k', 'w']
        for i in range(n_components):
            plt.scatter(X_c[:, i], Y_c[:, i], color=color[i], label=f'Canonical Component {i + 1}')
        plt.xlabel('Canonical Component 1')
        plt.ylabel('Canonical Component 2')
        plt.title(title)
        plt.legend()
        
        # 过原点线性拟合
        for i in range(n_components):
            a, b = np.polyfit(X_c[:, i], Y_c[:, i], 1)
            plt.plot(X_c[:, i], a * X_c[:, i] + b, color=color[i], linestyle='--', label=f'Linear Fit {i + 1}')
        
        if show:
            plt.show()
        else:
            plt.savefig(f'result/analysis/{title.replace(" ", "_")}.png')

    
    return cca, X_c, Y_c