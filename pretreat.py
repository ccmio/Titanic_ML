import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from discret import Binning
from model import MyModel
import pickle
pd.set_option('display.max_columns', 1000, 'max_rows', 100000, 'expand_frame_repr', False)

# 是否装载制作好的训练集
cleaned_data_path = './data/cleaned_data'
load_data = False


class Dealer:
    def __init__(self, train_path, test_path):
        self.train_path = train_path
        self.test_path = test_path

    def load_clean(self):
        if load_data:
            with open(cleaned_data_path, 'rb') as file:
                train_test = pickle.load(file)
        else:
            print('\n==================== data loading... ================')
            # 合并训练/测试集
            train = pd.read_csv(self.train_path, header=0)
            test = pd.read_csv(self.test_path, header=0)
            train_test = pd.concat([train, test], sort=True)
            # 缺失值统计
            for col in train_test.columns.tolist():
                missing_num = train_test[col].isnull().sum()
                if missing_num:
                    print('Missing value:{:10s} : {:5}, ({:.1f}%) '.format(col, missing_num, missing_num*100/len(train_test)))
            print('==================== data loaded. ===================\n')

            print('\n==================== feature engineering... ===================\n')
            # 生成一个feature，用来标示数据缺失程度——显然数据缺失的越多越可能是个死人
            train_test['DataLacker'] = None
            train_test.loc[(train_test['Age'].isnull()) & (train_test['Cabin'].isnull()), 'DataLacker'] = 3
            train_test.loc[(train_test['Age'].notnull()) & (train_test['Cabin'].isnull()), 'DataLacker'] = 2
            train_test.loc[(train_test['Age'].isnull()) & (train_test['Cabin'].notnull()), 'DataLacker'] = 1
            train_test.loc[(train_test['Age'].notnull()) & (train_test['Cabin'].notnull()), 'DataLacker'] = 0

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
            train_test.drop(['Name', 'Ticket', 'PassengerId'], axis=1, inplace=True)
            train_test.rename(columns={'Name1': 'Title'}, inplace=True)

            # 处理Cabin标签
            pattern3 = '.*([A-Z])[0-9]*'
            train_test['Deck'] = train_test['Cabin'].str.extract(pattern3, expand=False).str.strip()
            train_test.drop('Cabin', axis=1, inplace=True)

            # 异常值处理
            train_test['Embarked'].replace('a', 'S', inplace=True)
            train_test['SibSp'].replace(8, 6, inplace=True)
            train_test['Parch'].replace(9, 7, inplace=True)
            train_test['Deck'].replace('T', 'A', inplace=True)

            # 处理fare缺失值
            fare_list = []
            for row in train_test[(train_test['Fare'] == 0) | (train_test['Fare'].isnull())].iterrows():
                if isinstance(row[1]['Deck'], str):
                    fare = train_test.groupby(['Deck', 'Pclass'])['Fare'].mean()[row[1]['Deck']][row[1]['Pclass']]
                    fare_list.append(fare)
                else:
                    fare = train_test.groupby(['Title', 'Pclass'])['Fare'].mean()[row[1]['Title']][row[1]['Pclass']]
                    fare_list.append(fare)
            train_test.loc[(train_test['Fare'] == 0) | (train_test['Fare'].isnull()), 'Fare'] = fare_list

            # 处理Embarked缺失值
            train_test['Embarked'].fillna(train_test['Embarked'].mode()[0], inplace=True)

            # label encoding
            train_test['Embarked'].replace(['S', 'C', 'Q'], range(3), inplace=True)
            train_test['Sex'].replace(['male', 'female'], range(2), inplace=True)
            train_test['Pclass'].replace([1, 2, 3], range(3), inplace=True)
            train_test['Deck'].replace(['A', 'B', 'C', 'D', 'E', 'F', 'G'], range(7), inplace=True)
            train_test['Title'].replace(['Officer', 'Royalty', 'Mrs', 'Miss', 'Mr', 'Master'], range(6), inplace=True)

            # 数据相关性分析
            for corr_type in ['pearson', 'spearman', 'kendall']:
                fig, m_ax = plt.subplots(figsize=(10, 8))
                plt.title(corr_type.capitalize() + '  Correlation')
                ax = sns.heatmap(train_test.corr(corr_type), linewidths=0.1, square=True, ax=m_ax, linecolor='white',
                                 annot=True)
                ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
                ax.xaxis.set_ticks_position('top')
                bottom, top = ax.get_ylim()  # matplotlib3.1.1的bug，会导致heatmap图上下边缘显示不全，只好通过手动设定轴的范围来解决
                ax.set_ylim(bottom + 0.5, top - 0.5)
                # plt.show()

            # 连续数据离散化 - 卡方分箱
            age_binning = Binning(train_test[['Pclass', 'Age']].dropna(axis=0))
            age_bins = age_binning.chimerge(limit=7)
            res = pd.cut(train_test['Age'], age_bins, labels=range(7))
            train_test['Age'] = res
            fare_binning = Binning(train_test[['Pclass', 'Fare']].dropna(axis=0))
            fare_bins = fare_binning.chimerge(limit=11)
            res = pd.cut(train_test['Fare'], fare_bins, labels=range(10, -1, -1))
            train_test['Fare'] = res

            # Age缺失值处理
            pre_ages = MyModel.my_bayes(train_test[['Age', 'Title', 'SibSp', 'Parch']])
            train_test.loc[(train_test['Age'].isnull()), 'Age'] = pre_ages

            # Deck缺失值处理
            pre_decks = MyModel.my_bayes(train_test[['Deck', 'Pclass', 'Fare']])
            train_test.loc[(train_test['Deck'].isnull()), 'Deck'] = pre_decks

            train_test['Deck'] = train_test['Deck'].astype(int)
            print(len(train_test[train_test.duplicated()]))
            print(len(train_test))
            with open(cleaned_data_path, 'wb') as file:
                pickle.dump(train_test, file)
        train_test = train_test[['Survived', 'Title', 'Sex', 'Age', 'SibSp', 'Parch', 'Embarked', 'Pclass', 'Fare', 'Deck', 'DataLacker']]

        result = MyModel.my_ranforest(train_test)
        print('=================== feature engineering Done. =================\n')
        return train_test

        # 粗略可视化，观察Survived Embarked之间的互相影响，并以age的直方图来呈现
        # grid = sns.FacetGrid(train_test, col='Survived', row='Pclass', height=2.2, aspect=1.6)
        # grid.map(plt.hist, 'Fare', alpha=.5, bins=5)
        # grid.add_legend()
        # plt.show()
        # sns.barplot(y=train_test['Survived'], x=train_test['Pclass'])
        # plt.show()
