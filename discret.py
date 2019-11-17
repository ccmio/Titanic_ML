import numpy as np
import pandas as pd

load_txt = False  # 文件读取开关，分箱计算量不小，没必要每次都算
base_path = './data/'


class Binning:

    def __init__(self, data):
        self.data = data
        self.label = self.data.columns.values[0]  # 分类标准为列标题第一个，也就是survived
        self.feature = self.data.columns.values[1].lower()  # 待分箱特征为输入的列标题的第二个（Age，Fare）

    # 卡方分箱方法
    def chimerge(self, limit):
        # 分类指标为Pclass（与age）相关性最大
        def initial_dataset(dataset):
            data_list = []
            for client in dataset.values:
                temp = {self.feature: client[0]}
                data_list.append((temp, client[1:]))
            data_list = sorted(data_list, key=lambda x: x[0][self.feature])
            return data_list

        def chi_square(bin1, bin2):
            stat1 = np.zeros(len(bin1[0][1]))
            stat2 = np.zeros(len(bin2[0][1]))
            for person in bin1:
                stat1 += person[1]
            for person in bin2:
                stat2 += person[1]
            stat = np.vstack([stat1, stat2])
            n = np.sum(stat)
            for i in range(2):
                for j in range(len(bin1[0][1])):
                    eij = np.sum(stat[:, j]) * np.sum(stat[i, :]) / n
                    stat[i][j] = np.square(stat[i][j] - eij) / (eij + 1e-10)
            return np.sum(stat)

        def merge(data_list, path):
            bins = [[person] for person in data_list]
            while len(bins) > limit:
                chi2s = []
                min_idxs = []
                merge_point = []
                temp = []
                gap = 0
                merged_bins = []
                # 记录全部区间对的卡方值
                for idx in range(len(bins) - 1):
                    chi2 = chi_square(bins[idx], bins[idx + 1])
                    chi2s.append(chi2)
                # 记录并列最小的卡方值的区间对的融合起点（区间对的前者的索引）
                # ！万一出现需要融合的个体连续，比如融合起点为1，2，3，那么应该把1，2，3，4区间全部融合起来
                # 所以存储融合起点的时候可以判断，如果下一个起点比上一个起点大1，那么不append，直接取代
                chi2_min = min(chi2s)

                # 确定最小值点
                for idx, chi2 in enumerate(chi2s):
                    if chi2 == chi2_min:
                        min_idxs.append(idx)
                min_idxs.append(len(bins) - 1)

                # 划分聚类集合
                for i, idx in enumerate(min_idxs):
                    if idx == i + gap:
                        temp.append(idx)
                    else:
                        temp.append(i + gap)
                        merge_point.append(temp)
                        j = i + gap + 1
                        while j < idx:
                            merge_point.append([j])
                            j = j + 1
                        temp = [idx]
                        gap = idx - i
                    if i == len(min_idxs) - 1:
                        merge_point.append(temp)

                # 聚类
                for point in merge_point:
                    temp_bin = []
                    for idx in point:
                        temp_bin += bins[idx]
                    merged_bins.append(temp_bin)
                bins = merged_bins

            points = [-1.]
            for obj in bins:
                points.append(obj[-1][0][self.feature])
            with open(path, 'w+') as file:
                for point in points:
                    file.write(str(point)+'\n')
            return points

        path = base_path + self.feature + '_bins.txt'
        print('{:>10s}: Chimerging...\n'.format(self.feature))
        if not load_txt:
            dataset = pd.get_dummies(self.data, columns=[self.label])
            data_list = initial_dataset(dataset)
            bins = merge(data_list, path)
        else:
            with open(path) as file:
                bins = [float(point[:-1]) for point in file.readlines()]
        print('{:>10s}: ChiMerge DONE.\n'.format(self.feature))
        return bins


