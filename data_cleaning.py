import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
np.set_printoptions(threshold=np.inf, linewidth=np.inf)


# 合并训练/测试集
train = pd.read_csv('./train.csv', header=0)
test = pd.read_csv('./test.csv', header=0)
test['Survived'] = 0
train_test = train.append(test, sort=False)

# 合并兄弟姐妹配偶/父母子女标签
train_test['SibSp_Parch'] = train_test['SibSp'] + train_test['Parch']

# one hot encoding
train_test = pd.get_dummies(train_test, columns=['SibSp', 'Parch', 'SibSp_Parch', 'Pclass', 'Sex', 'Embarked'])


# 处理name标签
pattern1 = '.*,(.*)'
pattern2 = '^(.*?)\.'
train_test['Name1'] = train_test['Name'].str.extract(pattern1, expand=False).str.extract(pattern2, expand=False).str.strip()
train_test['Name1'].replace(['Capt', 'Col', 'Major', 'Dr', 'Rev'], 'Officer', inplace=True)
train_test['Name1'].replace(['Jonkheer', 'Don', 'Sir', 'the Countess', 'Dona', 'Lady'], 'Royalty', inplace=True)
train_test['Name1'].replace(['Mme', 'Ms', 'Mrs'], 'Mrs', inplace=True)
train_test['Name1'].replace(['Mlle', 'Miss'], 'Miss', inplace=True)
train_test['Name1'].replace(['Mr'], 'Mr', inplace=True)
train_test['Name1'].replace(['Master'], 'Master', inplace=True)

print(train_test.columns)


'''
粗略可视化，观察Survived Embarked之间的互相影响，并以age的直方图来呈现
grid = sns.FacetGrid(train, col='Survived', row='Embarked', height=2.2, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
grid.add_legend()
plt.show()

'''
