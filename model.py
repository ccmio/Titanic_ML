from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Activation
import pandas as pd
import numpy as np
pd.set_option('display.max_columns', 1000, 'max_rows', 100000, 'expand_frame_repr', False)
np.set_printoptions(threshold=np.inf, linewidth=np.inf)


class MyModel:
    def __init__(self):
        pass

    @staticmethod
    def my_mlp():
        model = Sequential()
        # 不写input_shape会报无法载入weights的错！
        model.add(Dense(input_shape=(9,), units=16))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(0.5))

        model.add(Dense(units=2))
        model.add(BatchNormalization())
        model.add(Activation('softmax'))

        return model

    def my_ranforest(self):
        pass

    @staticmethod
    def my_bayes(dataframe, lamb=1):  # lamb 拉普拉斯平滑系数

        '''
        df['Age', 'Title', 'SibSp', 'Parch', 'Pclass', 'Fare']
        '''
        # 计算所有feature的可取值个数 cates_num
        pre_target = dataframe.columns.values[0]
        print('{:>10s}: Bayes classifying...\n'.format(pre_target))
        cates_num = [len(set(feature[1])) for feature in dataframe.iteritems()]
        pre_cates = len(set(dataframe.loc[(dataframe[pre_target].notnull()), pre_target].values))
        cates_num.pop(0)
        cates_num.insert(0, pre_cates)

        to_pre = dataframe.loc[dataframe[pre_target].isnull()].values[:, 1:]

        df = dataframe.dropna(axis=0)
        # 计算分类结果的先验概率y_prob[]
        y_num = []
        y_prob = []
        total_age_num = len(df[pre_target])
        for age in range(cates_num[0]):
            age_num = sum(df[pre_target] == age)
            y_num.append(age_num)
            y_prob.append(age_num/total_age_num)

        # 计算条件概率con_prob (年龄，feature种类， feature取值）
        con_prob = []
        for age in range(cates_num[0]):
            temp_age = []
            for idx, x_i in enumerate(df.columns.values[1:]):
                temp_x_i = []
                # cates_num[idx+1]为第idx个feature可以取值的个数
                sj = cates_num[idx+1]
                for x_ij in range(sj):
                    temp_x_i.append((len(df.loc[(df[x_i] == x_ij) & (df[pre_target] == age)])+lamb)/(y_num[age] + sj * lamb))
                temp_age.append(np.array(temp_x_i))
            con_prob.append(np.array(temp_age))
        con_prob = np.array(con_prob)

        # 贝叶斯分类
        pre_result = []
        for person in to_pre:
            person_prob = []
            for age in range(cates_num[0]):
                prob = y_prob[age]
                for idx, feature in enumerate(person):
                    try:
                        prob *= con_prob[age][idx][feature]
                    except:
                        print(age, idx, feature)
                person_prob.append(prob)
            pre_result.append(np.argmax(person_prob))
        print('{:>10s}: Bayes classify DONE.\n'.format(pre_target))
        return pre_result




