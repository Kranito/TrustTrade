'''
买家初始化一个model，计算效用函数给卖家
卖家返回数据后更新训练集重新获取效用函数
通过ACS+Fscore的方式来选择数据
'''
import copy
import random
import sys
import time

import joblib
from sklearn.decomposition import PCA

sys.path.append('example_forgetting_master')
import os
from sklearn.linear_model import RidgeCV,LassoCV
import torch
from example_forgetting_master import run_cifar
import os
import numpy as np
import pandas as pd



def same_seeds(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_hms(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)

    return h, m, s
# same_seeds(2)

class iter_purchase():
    def __init__(self, select_method,budget_per_round,iter_time):

        # ------------ hyper parameters-----------------
        self.budget = budget_per_round
        self.iter_time = iter_time
        self.select_method = select_method

        # origin Datamodel embedding
        origin_data_ebd = torch.load('./train_50pct_old.pt', map_location=torch.device('cpu'))
        '''
        Use the following code to transfer new vision into old vision torch pack.
        data_ebd = torch.load('./train_50pct.pt', map_location=torch.device('cpu'))
        torch.save(data_ebd, './train_50pct_old.pt', _use_new_zipfile_serialization=False)
        '''
        retrain = not (os.path.exists(f"./Models/PCA_4000.pkl"))
        print(retrain)
        self.PCA_num = 4000
        if retrain:
            PCA_model = PCA(n_components=self.PCA_num)
            if os.path.exists(f"./Models/PCA_{self.PCA_num}.pkl"):
                PCA_model = joblib.load(f"./Models/PCA_{self.PCA_num}.pkl")
            else:
                PCA_model.fit(origin_data_ebd)
                joblib.dump(PCA_model, f"./Models/PCA_{self.PCA_num}.pkl")


        else:
            PCA_model = PCA(n_components=4000)
            PCA_model = joblib.load(f"./Models/PCA_4000.pkl")

        print("PCA model initialized successfully")
        self.data_ebd = np.array(PCA_model.transform(origin_data_ebd).tolist())

        self.train_idx = []
        self.pool_idx = list(range(len(self.data_ebd)))
        print(len(self.data_ebd))
        self.predicted_forget_time = []
        self.acc_record = []

    def select_data(self):
        '按照难度给权重抽样'

        if len(self.train_idx) == 0:
            selected_idx = random.sample(self.pool_idx,self.budget)

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
            elif select_method == 'rand':
                random.shuffle(idx)
                selected_idx = idx[-self.budget:]
        # print(selected_idx)
        # print(len(selected_idx))
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

        acc_lastround,forget_times = run_cifar.run_cifar(self.train_idx)
        print(forget_times)
        print(len(idx_embedding))

        linermodel = RidgeCV().fit(idx_embedding, forget_times)
        self.predicted_forget_time = linermodel.predict(pool_embedding)
        self.acc_record.append(acc_lastround)


    def run(self):
        for i in range(self.iter_time):
            print(f'round{i} started!')
            self.UAS()
            print(f'目前准确度：{self.acc_record}')
            print(f'round{i} finished!')
        print(f"最终准确度为：{self.acc_record}")
        return self.acc_record



def EA(data_num = 30000):
    filepath = './baselines/EA and SPS/Data-acquisition-for-ML-main/CIFAR10/1/EA-Squareroot-0.05001-' + str(
            data_num)
    with open(filepath)as f:
        reader = f.readline()
        train_idx = [int(x) for x in reader.split(' ')]
    acc_lastround, forget_times = run_cifar.run_cifar(train_idx)
    print(acc_lastround)
    return acc_lastround

def ACA(data_num = 30000):
    filepath = './baselines/EA and SPS/Data-acquisition-for-ML-main/CIFAR10/1/EA-Squareroot-0.05001-' + str(
            data_num)
    with open(filepath)as f:
        reader = f.readline()
        train_idx = [int(x) for x in reader.split(' ')]
    acc_lastround, forget_times = run_cifar.run_cifar(train_idx)
    print(acc_lastround)
    return acc_lastround

if __name__ == '__main__':
    acc_all = []
    exp_time = 1

    run_type = 'EA'
    if run_type == 'UAS':
        for i in range(exp_time):
            print(run_type)
            env = iter_purchase(select_method='order', budget_per_round=3000, iter_time=16)
            all_list = env.run(run_type)
            acc_all.append(all_list)
        print(f"最终{exp_time}轮准确度为：{acc_all}")
    elif run_type == 'EA':
        EA(data_num=30000)
    elif run_type == 'ACS':
        ACS(data_num=30000)

