import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
pd.set_option('display.max_columns', 1000, 'max_rows', 100000, 'expand_frame_repr', False)


class Dealer():
    def __init__(self, train_path, test_path):
        self.train_path = train_path
        self.test_path = test_path

    def load_clean(self):
        # 合并训练/测试集
        train = pd.read_csv(self.train_path, header=0)
        test = pd.read_csv(self.test_path, header=0)
        test['Survived'] = 0
        train_test = train.append(test, sort=False)

        # 合并兄弟姐妹配偶/父母子女标签
        train_test['SibSp_Parch'] = train_test['SibSp'] + train_test['Parch']

        # one hot encoding
        # train_test = pd.get_dummies(train_test, columns=['SibSp', 'Parch', 'SibSp_Parch', 'Pclass', 'Sex', 'Embarked'])

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
        train_test.drop(['Name', 'Cabin', 'SibSp', 'Parch', 'Ticket', 'PassengerId'], axis=1, inplace=True)
        train_test.rename(columns={'Name1': 'Title'}, inplace=True)
        train_test = train_test[['Survived', 'Title', 'Sex', 'Age', 'Embarked', 'Pclass', 'Fare', 'SibSp_Parch']]
        train_test['Embarked'].replace(['S', 'C', 'Q', 'a'], [0, 1, 2, 1], inplace=True)
        train_test['Sex'].replace(['male', 'female'], [0, 1], inplace=True)
        train_test['Pclass'].replace([1, 2, 3], [0, 1, 2], inplace=True)
        train_test['Title'].replace(['Officer', 'Royalty', 'Mrs', 'Miss', 'Mr', 'Master'], [0, 1, 2, 3, 4, 5], inplace=True)

        # 暂时用均值/众数的办法填充nan
        train_test['Age'].fillna(train_test['Age'].mean(), inplace=True)
        train_test['Fare'].fillna(train_test['Fare'].mean(), inplace=True)
        train_test['Embarked'].fillna(train_test['Embarked'].mode()[0], inplace=True)

        # 分组
        age_bins = [0., 18., 30., 40., 50., 100.]
        res = pd.cut(train_test['Age'], age_bins, labels=[0, 1, 2, 3, 4], right=False)
        train_test['Age'] = res

        fare_bins = [-0.1, 50, 100, 150, 200, 250, 300, 521]
        res = pd.cut(train_test['Fare'], fare_bins, labels=[0, 1, 2, 3, 4, 5, 6])
        train_test['Fare'] = res

        train = train_test[:891].copy()
        test = train_test[891:].copy()
        test.drop(['Survived'], axis=1, inplace=True)

        return train.values, test.values
        # np.set_printoptions(threshold=np.inf, linewidth=np.inf)
        # corr = train_test.corr()
        # print(corr['Survived'].sort_values(ascending=False))


        # # 粗略可视化，观察Survived Embarked之间的互相影响，并以age的直方图来呈现
        # grid = sns.FacetGrid(train_test, col='Survived', row='Pclass', height=2.2, aspect=1.6)
        # grid.map(plt.hist, 'Fare', alpha=.5, bins=5)
        # grid.add_legend()
        # plt.show()
        # sns.barplot(y=train_test['Survived'], x=train_test['Pclass'])
        # plt.show()