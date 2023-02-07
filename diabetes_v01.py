import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv('diabetes.csv')
df.dropna(axis=1, inplace=True)
df.head(9)


def norm_x(item):
    X = np.array(df[item]).reshape(-1, 1)
    scaler = MinMaxScaler()
    scaler.fit(X)
    X_scaled = scaler.transform(X)
    df[item] = X_scaled.reshape(1, -1)[0]


norm_x('preg')
norm_x('plas')
norm_x('pres')
norm_x('skin')
norm_x('insu')
norm_x('mass')
norm_x('pedi')
norm_x('age')


# one-hot-encoder
encoder = OneHotEncoder(handle_unknown='ignore')
encoder_df = pd.DataFrame(encoder.fit_transform(df[['class']]).toarray())

df = df.join(encoder_df)
df.drop('class', axis=1, inplace=True)
df.drop(0, axis=1, inplace=True)
df.to_csv('processed_data.csv', index=False)
