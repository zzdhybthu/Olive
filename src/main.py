import pandas as pd
from selection import Select_PCA, Select_CA
from analysis import Analysis_PCA, Analysis_CA, Analysis_CCA, Analysis_Corr
from preprocess import preprocess_data
from eda import eda
import datatype as dt
import os


data = pd.read_csv('dataset/oliveoil.csv')

os.makedirs('result/eda',exist_ok=True)
os.makedirs('result/selection',exist_ok=True)
os.makedirs('result/analysis',exist_ok=True)


def main(x_columns, y_columns, x_name, y_name, critical_value=0.9, plot=True, show=True):
    global data
    x_features = data[x_columns]
    y_features = data[y_columns]
    xy_features = data[x_columns + y_columns]


    correlation_matrix = Analysis_Corr(
        x_features, 
        y_features, 
        plot=plot, 
        x_columns=x_columns, 
        y_columns=y_columns, 
        title=f'Correlation Matrix between X ({x_name} Features) and Y ({y_name} Features)',
        xlabel=f'{x_name} Features',
        ylabel=f'{y_name} Features',
        show=show
    )


    x_n_components = Select_PCA(
        x_features, 
        critical_value=critical_value,
        plot=plot,
        title=f'PCA Selection of Olive {x_name} Data',
        show=show
    )

    x_pca, x_pca_features = Analysis_PCA(
        x_features, 
        x_n_components, 
        plot=plot, 
        feature_columns=x_columns, 
        title=f'PCA Analysis of Olive {x_name} Data',
        show=show
    )

    # 将 PCA 结果添加回原数据集
    for i in range(x_n_components):
        data[f'{x_name}_PCA_{i + 1}'] = x_pca_features[:, i]
    x_pca_columns = [f'{x_name}_PCA_{i}' for i in range(1, x_n_components + 1)]
    x_pca_features = data[x_pca_columns]


    y_n_components = Select_PCA(
        y_features, 
        critical_value=critical_value,
        plot=plot,
        title=f'PCA Selection of Olive {y_name} Data',
        show=show
    )

    y_pca, y_pca_features = Analysis_PCA(
        y_features, 
        y_n_components, 
        plot=plot, 
        feature_columns=y_columns, 
        title=f'PCA Analysis of Olive {y_name} Data',
        show=show
    )


    # 将 PCA 结果添加回原数据集
    for i in range(y_n_components):
        data[f'{y_name}_PCA_{i + 1}'] = y_pca_features[:, i]
    y_pca_columns = [f'{y_name}_PCA_{i}' for i in range(1, y_n_components + 1)]
    y_pca_features = data[y_pca_columns]
    
    
    x_optimal_k = Select_CA(
        x_features,
        method="kmeans",
        plot=plot,
        title=f'K-Means Clustering Selection of Olive {x_name} Data',
        show=show
    )

    x_kmeans = Analysis_CA(
        x_features,
        x_optimal_k,
        method="kmeans",
        plot=plot,
        pca=x_pca,
        feature_columns=x_columns,
        title=f'K-Means Clustering of Olive {x_name} Data',
        show=show
    )
    
    
    y_optimal_k = Select_CA(
        y_features,
        method="kmeans",
        plot=plot,
        title=f'K-Means Clustering Selection of Olive {y_name} Data',
        show=show
    )
    
    y_kmeans = Analysis_CA(
        y_features,
        y_optimal_k,
        method="kmeans",
        plot=plot,
        pca=y_pca,
        feature_columns=y_columns,
        title=f'K-Means Clustering of Olive {y_name} Data',
        show=show
    )


    correlation_matrix_pca = Analysis_Corr(
        x_pca_features, 
        y_pca_features, 
        plot=plot, 
        x_columns=x_pca_columns, 
        y_columns=y_pca_columns, 
        title=f'Correlation Matrix between {x_name} PCA Features and {y_name} PCA Features',
        xlabel=f'{x_name} PCA Features',
        ylabel=f'{y_name} PCA Features',
        show=show
    )


    cca, X_c, Y_c = Analysis_CCA(
        x_pca_features, 
        y_pca_features,
        n_components=min(x_n_components, y_n_components), 
        plot=plot, 
        x_columns=x_pca_columns,
        y_columns=y_pca_columns,
        title=f'CCA Analysis between Olive {x_name} PCA Features and Olive {y_name} PCA Features',
        show=show
    )



eda(data, show=True)
data = preprocess_data(data)
# 氧化程度与酸度的关系
main(dt.oxidation_columns, dt.acid_columns, 'Oxidation', 'Acid', critical_value=0.9, plot=True, show=False)
# 视觉与浓稠度的关系
main(dt.visual_columns, dt.viscosity_columns, 'Visual', 'Viscosity', critical_value=0.9, plot=True, show=False)
# 物理性质与化学性质的关系
main(dt.physics_columns, dt.chemistry_columns, 'Pyhsical', 'Chemical', critical_value=0.9, plot=True, show=True)