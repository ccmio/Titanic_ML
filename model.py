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
        model.add(Dense(input_shape=(10,), units=16))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(0.5))

        model.add(Dense(units=2))
        model.add(BatchNormalization())
        model.add(Activation('softmax'))

        return model

    @staticmethod
    def my_ranforest(dataframe):

        def gini_calc(dataset):
            length = dataset.shape[0]
            feature_cates = [list(sorted(set(dataset[:, i]))) for i in range(dataset.shape[1])]
            # print(feature_cates)
            gini_matrix = []
            min_gini = []
            for idx, feature in enumerate(feature_cates[1:]):
                feature_ginis = []

                for feature_value in feature:
                    equal_dataset = dataset[dataset[:, idx+1] == feature_value]
                    equal_len = len(equal_dataset)
                    equal_prob = equal_len/length
                    equal_con_prob_sum = 0

                    neq_dataset = dataset[dataset[:, idx+1] != feature_value]
                    neq_len = length - equal_len
                    neq_prob = 1 - equal_prob
                    neq_con_prob_sum = 0

                    for label in feature_cates[0]:
                        if equal_len:
                            equal_con_dataset = equal_dataset[equal_dataset[:, 0] == label]
                            equal_con_prob = len(equal_con_dataset)/equal_len
                        else:
                            equal_con_prob = 0
                        equal_con_prob_sum += equal_con_prob * (1 - equal_con_prob)

                        if neq_len:
                            neq_con_dataset = neq_dataset[neq_dataset[:, 0] == label]
                            neq_con_prob = len(neq_con_dataset)/neq_len
                        else:
                            neq_con_prob = 0
                        neq_con_prob_sum += neq_con_prob * (1 - neq_con_prob)

                    feature_ginis.append(equal_prob*equal_con_prob_sum + neq_prob*neq_con_prob_sum)
                min_gini.append(min(feature_ginis))
                gini_matrix.append(feature_ginis)

            # print('min gini = ', min(min_gini))
            feature_idx = min_gini.index(min(min_gini))
            value_idx = gini_matrix[feature_idx].index(min(gini_matrix[feature_idx]))
            feature = feature_idx + 1
            value = feature_cates[feature][value_idx]
            # for i in gini_matrix:
            #     print(i)
            return feature, value, min(min_gini), gini_matrix

        def node_divide(parent, dataset, depth):
            if depth < 100:
                feature, value, min_gini, gini_matrix = gini_calc(dataset)
                left_dataset = dataset[dataset[:, feature] == value]
                right_dataset = dataset[dataset[:, feature] != value]
                left_len = len(left_dataset)
                right_len = len(right_dataset)

                survived_num = len(dataset[dataset[:, 0] == 1])
                total_num = len(dataset)
                survived_prob = survived_num/total_num
                loss = survived_prob if survived_prob < 0.5 else 1 - survived_prob
                parent.loss = loss

                if left_len > 0 and right_len > 0:
                    parent.feature = feature
                    parent.value = value
                    parent.left_len = left_len
                    parent.right_len = right_len

                    left_node = DeTreeNode()
                    right_node = DeTreeNode()
                    parent.left = node_divide(left_node, left_dataset, depth+1)
                    parent.right = node_divide(right_node, right_dataset, depth+1)
                else:
                    parent.survived = 1 if survived_prob > 0.5 else 0
                    # print('========叶子结点========人数：{} 生存率：{:.1f}%'.format(total_num, parent.survived*100))
                    # print(dataset)
            return parent

        def predict(parent, dataset):
            feature_dict = {idx: value for idx, value in enumerate(dataset.columns.values)}
            feature = parent.feature
            value = parent.value
            if parent.left and parent.right:
                left_dataset = dataset[dataset[feature_dict[feature]] == value]
                right_dataset = dataset[dataset[feature_dict[feature]] != value]
                predict(parent.left, left_dataset)
                predict(parent.right, right_dataset)
            elif len(dataset):
                index = dataset.index.values
                result[index] = parent.survived
            return result

        def acc_cul(parent, dataset):
            feature_dict = {idx: value for idx, value in enumerate(dataset.columns.values)}
            feature = parent.feature
            value = parent.value
            survived_num = len(dataset[dataset[:, 0] == 1])
            total_num = len(dataset)
            survived_prob = survived_num / total_num
            loss = survived_prob if survived_prob < 0.5 else 1 - survived_prob
            parent.loss = loss
            if parent.left and parent.right:
                left_dataset = dataset[dataset[feature_dict[feature]] == value]
                right_dataset = dataset[dataset[feature_dict[feature]] != value]
                acc_cul(parent.left, left_dataset)
                acc_cul(parent.right, right_dataset)
            else:
                parent.survived = 1 if survived_prob > 0.5 else 0
            return

        def pruning(tree):
            t = tree
            alpha = np.inf
            g = []
            ctt = 0
            tt = 0

            def postorder(tree, ctt, tt, depth):
                if tree.left and tree.right:
                    print(depth)
                    print(tree.loss)
                    ctt_l, tt_l = postorder(tree.left, ctt, tt, depth+1)
                    ctt_r, tt_r = postorder(tree.right, ctt, tt, depth+1)
                    ctt = ctt_l + ctt_r
                    tt = tt_l + tt_r
                    g.append((tree.loss - ctt)/(tt - 1))
                    return ctt, tt
                else:
                    ctt += tree.loss
                    tt += 1
                    return ctt, tt
            postorder(tree, ctt, tt, 0)
            print(g, ctt, tt)
            return

        class DeTreeNode:
            def __init__(self):
                self.feature = None
                self.value = None
                self.left = None
                self.left_len = None
                self.right = None
                self.right_len = None
                self.loss = None

        length = len(dataframe[dataframe['Survived'].notnull()])
        dataset = dataframe.values[:length]
        np.random.seed(6)
        np.random.shuffle(dataset)
        root = DeTreeNode()
        decision_tree = node_divide(root, dataset, 0)
        result = np.zeros(len(dataframe) - length)
        result = predict(decision_tree, dataframe[length:])
        # pruning(decision_tree)

        return result


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




