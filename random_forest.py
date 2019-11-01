from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
df = pd.read_csv('./train.csv', header=0)
df.replace('male', 1, inplace=True)
df.replace('female', 0, inplace=True)
labels = df['Survived'].values
datas = df.drop(['Survived'], axis=1).values
np.set_printoptions(threshold=np.inf, linewidth=np.inf)
print(datas)
model = RandomForestClassifier(n_estimators=100, bootstrap=True, max_features='sqrt')
