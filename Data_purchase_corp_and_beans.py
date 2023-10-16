import copy
from sklearnex import patch_sklearn
patch_sklearn()
import random
from sklearn.linear_model import RidgeCV,LassoCV
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from scipy.stats import entropy
import os
from sklearn.metrics import accuracy_score
from sklearn import preprocessing



class iter_purchase():
    def __init__(self, select_method,budget_per_round,iter_time,init_size,dataset='crop'):

        # ------------ hyper parameters-----------------
        self.budget = budget_per_round
        self.iter_time = iter_time
        self.select_method = select_method
        self.dataset=dataset
        self.init_size = init_size
        if dataset == 'crop':
            data = np.loadtxt("./Data/WinnipegDataset.txt", delimiter=',', skiprows=1)
        elif dataset == 'poisonedcrop':
            data = np.loadtxt("./Data/PoisonedWinnipegDataset.txt", delimiter=',', skiprows=1)
        elif dataset == 'roadnet':
            data = np.loadtxt("./Data/3D_spatial_network.txt", delimiter=',')
        elif dataset == 'beans':
            file_path = "./Data/Dry_Bean_Dataset.xlsx"
            data = pd.read_excel(file_path, header=0)
            last_column_name = data.columns[-1]
            data[last_column_name] = pd.factorize(data[last_column_name])[0]
            data = data.to_numpy()
            data = np.delete(data, 6, axis=1)
        else:
            print("no dataset")
        np.random.seed(43)
        if dataset != 'poisonedcrop':
            np.random.shuffle(data)

        if self.dataset == 'crop':
            X = data[:, 1:]
            y = data[:, 0]
            y = np.array([int(v - 1) for v in y])
            min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
            X = min_max_scaler.fit_transform(X)
        elif self.dataset == 'poisonedcrop':
            X = data[:, 1:]
            y = data[:, 0]
            y = np.array([int(v - 1) for v in y])
            min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
            X = min_max_scaler.fit_transform(X)
        elif self.dataset == 'roadnet':
            X = data[:, 1:3]
            y = data[:, 3]
            min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
            X = min_max_scaler.fit_transform(X)
            y = min_max_scaler.fit_transform(y.reshape(-1, 1))
            y = y.flatten()
        elif self.dataset == 'beans':
            X = data[:,:-1]
            y = data[:,-1]

        sp = int(len(data)*0.8)

        self.X_pool = X[:sp]
        self.y_pool = y[:sp]

        if self.dataset == 'poisonedcrop':
            a = list(zip(self.X_pool,self.y_pool))
            random.shuffle(a)
            self.X_pool, self.y_pool = zip(*a)
            self.X_pool = np.array(self.X_pool)
            self.y_pool = np.array(self.y_pool)

        self.X_test = X[sp:]
        self.y_test = y[sp:]

        self.data_ebd = copy.deepcopy(self.X_pool)

        print(f'the size of the dataset is{len(self.data_ebd)}')

        self.predicted_forget_time = []
        self.acc_record = []
        self.train_idx = []
        self.pool_idx = list(range(len(self.data_ebd)))
        self.predicted_forget_time = []


    def get_etropy(self,data_indices):

        X = self.X_pool[data_indices]
        y = self.y_pool[data_indices]

        num_splits = 5
        split_size = len(X) // num_splits

        confidences = []
        for i in range(num_splits):
            test_start = i * split_size
            if i == num_splits - 1:
                test_end = len(X)
            else:
                test_end = (i + 1) * split_size
            test_indices = list(range(test_start, test_end))
            train_indices = list(range(0, test_start)) + list(range(test_end, len(X)))

            X_train, y_train = X[train_indices], y[train_indices]
            X_test, y_test = X[test_indices], y[test_indices]

            clf = RandomForestClassifier(n_estimators=100, random_state=42)
            # print(y_train)
            clf.fit(X_train, y_train)

            confidence = clf.predict_proba(X_test)
            print(confidence)
            if len(confidences) == 0:
                confidences = confidence
            else:
                confidences = np.concatenate((confidences, confidence), axis=0)

        # print(confidences)
        # calculate the entropy
        data_entropy = []
        for c in confidences:
            data_entropy.append(entropy(c))
        data_entropy = np.array(data_entropy)
        # print(data_entropy)
        return data_entropy

    def get_acc(self,data_indices):

        X = self.X_pool[data_indices]
        y = self.y_pool[data_indices]
        X_test = self.X_test
        y_test = self.y_test

        clf = RandomForestClassifier(n_estimators=100,random_state=42)
        clf.fit(X,y)
        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        return accuracy

    def select_data(self):
        if len(self.train_idx) == 0:
            selected_idx = random.sample(self.pool_idx,self.init_size)

        else:
            forget_time = np.array(copy.deepcopy(self.predicted_forget_time),dtype='int')
            for i in range(len(forget_time)):
                forget_time[i] = max(0,forget_time[i])
            data_weight = forget_time
            idx = copy.deepcopy(self.pool_idx)
            select_method = self.select_method
            if select_method == 'weight':
                df = pd.DataFrame({
                    'data_weight' : data_weight,
                    'idx' : idx
                })
                selected_idx = df.sample(n=self.budget, weights='data_weight')['idx']
            elif select_method == 'order':
                group = zip(idx,data_weight)
                groupvalue = sorted(group, key=(lambda x: float(x[1])))
                idx = list(zip(*groupvalue))[0]
                selected_idx = idx[-self.budget:]

        return list(selected_idx)

    def UAS(self):
        # update idx groups
        selected_idx = self.select_data()
        self.train_idx = sorted(self.train_idx + selected_idx)
        self.pool_idx = list(set(self.pool_idx)-set(self.train_idx))
        print(f'totally selected {len(self.train_idx)} data !')
        idx_embedding = []
        pool_embedding = []
        for idx in self.train_idx:
            idx_embedding.append(self.data_ebd[idx].tolist())
        for idx in self.pool_idx:
            pool_embedding.append(self.data_ebd[idx].tolist())

        # acc_lastround,forget_times = run_cifar.train_Model(self.train_idx,dataset=self.dataset,cal_forget=True,model=self.model)
        forget_times = self.get_etropy(self.train_idx)
        acc_lastround = self.get_acc(self.train_idx)
        # print(forget_times)
        # print(idx_embedding)
        # print(forget_times)
        # print(np.array(idx_embedding).isnull().any())
        # print(np.isnan(idx_embedding).any())
        # print(np.isfinite(idx_embedding).any())
        # # print(np.array(forget_times).isnull().any())
        # print(np.isnan(forget_times).any())
        # print(np.isfinite(idx_embedding).any())

        linermodel = RidgeCV().fit(idx_embedding, forget_times)
        self.predicted_forget_time = linermodel.predict(pool_embedding)
        self.acc_record.append(acc_lastround)


    def run(self):
        for i in range(self.iter_time):
            print(f'round{i} started!')
            self.UAS()
            print(f'acc：{self.acc_record}')
            print(f'round{i} finished!')
        print(f"final acc：{self.acc_record}")
        return self.acc_record


def get_acc(data_indices,dataset = 'crop'):
    print(len(data_indices))
    if dataset == 'crop':
        data = np.loadtxt("./Data/WinnipegDataset.txt", delimiter=',', skiprows=1)
    elif dataset == 'roadnet':
        data = np.loadtxt("./datasets/WinnipegDataset.txt", delimiter=',')
    elif dataset == 'beans':
        file_path = "./Data/Dry_Bean_Dataset.xlsx"
        data = pd.read_excel(file_path, header=0)
        last_column_name = data.columns[-1]
        data[last_column_name] = pd.factorize(data[last_column_name])[0]
        data = data.to_numpy()
        data = np.delete(data, 6, axis=1)
    else:
        print("no dataset")

    np.random.seed(43)
    np.random.shuffle(data)
    sp = int(len(data) * 0.8)


    data_pool = data[:sp]
    test_data = data[sp:]

    sdata_ebd = data_pool[:, 1:]

    if dataset == 'crop':
        X = data_pool[:, 1:][data_indices]
        y = data_pool[:, 0][data_indices]
        y = np.array([int(v - 1) for v in y])

        X_test = test_data[:, 1:]
        y_test = test_data[:, 0]
        y_test = np.array([int(v - 1) for v in y_test])
    elif dataset == 'roadnet':
        X = data[:, 1:3][data_indices]
        y = data[:, 3][data_indices]

        X_test = test_data[:, 1:3]
        y_test = test_data[:, 3]
        y_test = np.array([int(v - 1) for v in y_test])
    elif dataset == 'beans':
        X = data_pool[:, :-1][data_indices]
        y = data_pool[:, -1][data_indices]
        y = np.array([int(v) for v in y])

        X_test = test_data[:, :-1]
        y_test = test_data[:, -1]
        y_test = np.array([int(v) for v in y_test])
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X, y)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    # print(accuracy)
    return accuracy


if __name__ == '__main__':
    env = iter_purchase(select_method='order', budget_per_round=2500, iter_time=20,init_size=4500,
                        dataset='poisonedcrop')
    acc_list = env.run()

    # env = iter_purchase(select_method='order', budget_per_round=500, iter_time=10, init_size=200,
    #                     dataset='beans')
    # acc_list = env.run()
    # acc_list = []
    # for i in list(range(200,5200,500)):
    #     acc_list.append(get_acc(random.sample(list(range(10000)), i),dataset = "beans"))
    # print(acc_list)

    # print(get_acc(random.sample(list(range(260000)),40000)))
    # print(get_acc(random.sample(list(range(260000)), 30000)))
    # print(get_acc(random.sample(list(range(260000)), 20000)))
    # print(get_acc(random.sample(list(range(260000)), 10000)))


# np.random.seed(42)
# data = np.loadtxt("./Data/WinnipegDataset.txt", delimiter=',', skiprows=1)
# np.random.shuffle(data)
#
# X = data[:,1:]
# y = data[:,0]
# y = np.array([int(v-1) for v in y])
#
# num_splits = 5
# split_size = len(X) // num_splits
#
#
# confidences = []
# if not os.path.exists('./confidents_crop.pth'):
#     for i in range(num_splits):
#         test_start = i * split_size
#         if i==num_splits-1:
#             test_end = len(X)
#         else:
#             test_end = (i + 1) * split_size
#         test_indices = list(range(test_start, test_end))
#         train_indices = list(range(0, test_start)) + list(range(test_end, len(X)))
#
#         X_train, y_train = X[train_indices], y[train_indices]
#         X_test, y_test = X[test_indices], y[test_indices]
#
#         clf = RandomForestClassifier(n_estimators=100,random_state=42)
#         print(y_train)
#         clf.fit(X_train, y_train)
#
#
#         confidence = clf.predict_proba(X_test)
#         print(confidence)
#         if len(confidences) == 0:
#             confidences = confidence
#         else:
#             confidences = np.concatenate((confidences,confidence),axis = 0)
#     torch.save(confidences, './confidents_crop.pth')
# confidences = torch.load('./confidents_crop.pth')
# print(confidences)
# #calculate the entropy
# data_entropy = []
# for c in confidences:
#     data_entropy.append(entropy(c))
# data_entropy = np.array(data_entropy)
# print(data_entropy)
#
# data_index = np.array(list(range(len(X))))
# train_idx, test_idx = train_test_split(data_index, test_size=0.1, random_state=42)
#
# train_data = zip(train_idx,data_entropy[train_idx])
# train_data = sorted(train_data, key=(lambda x: float(x[1])))
# sorted_train_index = list(zip(*train_data))[0]
#
#
# test_indices = test_idx
# # low_entropy
# mid = 10000
# train_indices = list(sorted_train_index[:mid])
# X_train, y_train = X[train_indices], y[train_indices]
# X_test, y_test = X[test_indices], y[test_indices]
# clf = RandomForestClassifier(n_estimators=100,random_state=42)
# clf.fit(X_train, y_train)
# y_pred = clf.predict(X_test)
# accuracy = accuracy_score(y_test, y_pred)
# print(accuracy)
#
# # high_entropy
# train_indices = list(sorted_train_index[mid:])
# X_train, y_train = X[train_indices], y[train_indices]
# X_test, y_test = X[test_indices], y[test_indices]
# clf = RandomForestClassifier(n_estimators=100,random_state=42)
# clf.fit(X_train, y_train)
# y_pred = clf.predict(X_test)
# accuracy = accuracy_score(y_test, y_pred)
# print(accuracy)
