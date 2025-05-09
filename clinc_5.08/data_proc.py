import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# 加载CSV数据
df = pd.read_csv('dataset.csv')


# 数据预处理
def preprocess_data(df):
    print(f"原始数据形状: {df.shape}")

    # 添加额外特征
    df['glucose_cv'] = df['std_glucose'] / df['mean_glucose']
    df['glucose_fluctuation_index'] = df['glucose_range'] / np.sqrt(df['glucose_measurement_count'])

    # 高/低血糖严重程度
    df['severe_hyperglycemia_ratio'] = df.apply(
        lambda x: x['hyperglycemia_ratio'] * (x['max_glucose'] > 250), axis=1
    )

    # 创建年龄组的独热编码
    df['age_group'] = pd.cut(
        df['age_at_admission'],
        bins=[0, 18, 31, 51, 71, 120],
        labels=['0-17', '18-30', '31-50', '51-70', '71+']
    )
    age_dummies = pd.get_dummies(df['age_group'], prefix='age')
    df = pd.concat([df, age_dummies], axis=1)

    return df


# 预处理数据
processed_df = preprocess_data(df)
processed_df.to_csv('processed_dataset.csv', index=False)