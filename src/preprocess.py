import pandas as pd
from sklearn.preprocessing import StandardScaler


def preprocess_data(data):    
    # 标准化
    data = data.drop(columns=['ID'])
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    data_scaled = pd.DataFrame(data_scaled, columns=data.columns)

    return data_scaled
