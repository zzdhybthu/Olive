import seaborn as sns
import matplotlib.pyplot as plt


def eda(data, show=True):
    
    # 数据信息
    print("Data Info:")
    print(data.info())

    print("\nData Description:")
    print(data.describe())

    print("\nMissing Values:")
    print(data.isnull().sum())

    with open('result/eda/data_info.txt', 'w') as f:
        f.write("Data Info:\n")
        data.info(buf=f)
        
        f.write("\nData Description:\n")
        f.write(data.describe().to_string())
        
        f.write("\n\nMissing Values:\n")
        f.write(data.isnull().sum().to_string())

    # 分布
    plt.figure(figsize=(12, 8))
    for i, column in enumerate(data.columns[1:], 1):
        plt.subplot(3, 4, i)
        sns.histplot(data[column], kde=True)
        plt.title(f'Distribution of {column}')
    plt.tight_layout()
    if show:
        plt.show()
    else:
        plt.savefig('result/eda/distribution.png')

    # pairplot
    plt.figure(figsize=(24, 16))
    sns.pairplot(data.drop(columns=['ID']))
    if show:
        plt.show()
    else:
        plt.savefig('result/eda/pairplot.png')

    # 相关矩阵
    corr_matrix = data.drop(columns=['ID']).corr()
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
    plt.title('Correlation Matrix')
    if show:
        plt.show()
    else:
        plt.savefig('result/eda/corr_matrix.png')