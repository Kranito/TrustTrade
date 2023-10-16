
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"
import argparse
import copy
import random
import sys
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from scipy.stats import entropy
import joblib
from sklearn.decomposition import PCA

sys.path.append('example_forgetting_master')
import os
from sklearn.linear_model import RidgeCV,LassoCV
import torch
import run_cifar
import run_model

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


class iter_purchase():
    def __init__(self, select_method,budget_per_round,iter_time,dataset='cifar10',ebd_method = 'datamodel',checkpoint=None,model = 'resnet18'):

        # ------------ hyper parameters-----------------
        self.budget = budget_per_round
        self.iter_time = iter_time
        self.select_method = select_method
        self.checkpoint = checkpoint
        self.dataset=dataset
        self.ebd_method = ebd_method
        self.model = model

        # origin Datamodel embedding

        self.data_ebd = self.get_embedding(dataset = dataset,ebd_method=ebd_method)

        print(f'the size of the dataset is{len(self.data_ebd)}')

        self.predicted_forget_time = []
        self.acc_record = []
        if checkpoint:
            if model == 'resnet18':
                self.train_idx = torch.load(f'./Checkpoints/{self.dataset}-{self.budget}/train_idx-{self.ebd_method}-{checkpoint}.pth')
                self.pool_idx = torch.load(f'./Checkpoints/{self.dataset}-{self.budget}/pool_idx-{self.ebd_method}-{checkpoint}.pth')
                self.predicted_forget_time = torch.load(
                    f'./Checkpoints/{self.dataset}-{self.budget}/forgettime-{self.ebd_method}-{checkpoint}.pth')
            else :
                self.train_idx = torch.load(f'./Checkpoints/{self.dataset}-{self.budget}/{self.model}-train_idx-{self.ebd_method}-{checkpoint}.pth')
                self.pool_idx = torch.load(f'./Checkpoints/{self.dataset}-{self.budget}/{self.model}-pool_idx-{self.ebd_method}-{checkpoint}.pth')
                self.predicted_forget_time = torch.load(
                    f'./Checkpoints/{self.dataset}-{self.budget}/{self.model}-forgettime-{self.ebd_method}-{checkpoint}.pth')
        else:
            self.train_idx = []
            self.pool_idx = list(range(len(self.data_ebd)))
            self.predicted_forget_time = []


    def get_embedding(self,dataset,ebd_method):
        if ebd_method =='datamodel':
            if dataset == 'cifar10':
                origin_data_ebd = torch.load('./train_50pct_old.pt', map_location=torch.device('cpu'))
                '''
                           Use the following code to transfer new vision into old vision torch pack.
                           data_ebd = torch.load('./train_50pct.pt', map_location=torch.device('cpu'))
                           torch.save(data_ebd, './train_50pct_old.pt', _use_new_zipfile_serialization=False)
                           '''
                retrain = not (os.path.exists(f"./Models/PCA_4000.pkl"))
                print(retrain)
                PCA_num = 4000
                if retrain:
                    PCA_model = PCA(n_components=PCA_num)
                    if os.path.exists(f"./Models/PCA_{PCA_num}.pkl"):
                        PCA_model = joblib.load(f"./Models/PCA_{PCA_num}.pkl")
                    else:
                        PCA_model.fit(origin_data_ebd)
                        joblib.dump(PCA_model, f"./Models/PCA_{PCA_num}.pkl")
                else:
                    PCA_model = PCA(n_components=4000)
                    PCA_model = joblib.load(f"./Models/PCA_4000.pkl")

                print("PCA model initialized successfully")
                return np.array(PCA_model.transform(origin_data_ebd).tolist())

        elif ebd_method == 'clip':
            embedding = torch.load(f'./{dataset}-clip.pt')
            return embedding
        elif ebd_method == 'resnet':
            embedding = torch.load(f'./{dataset}-resnet.pt')
            return embedding
        elif ebd_method == 'resnet50':
            embedding = torch.load(f'./{dataset}-resnet50.pt')
            return embedding
        elif ebd_method == 'vgg16':
            embedding = torch.load(f'./{dataset}-vgg16.pt')
            return embedding

    def select_data(self):


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
                    'data_weight' : data_weight+0.001,#prevent from too much 0
                    'idx' : idx
                })
                selected_idx = df.sample(n=self.budget, weights='data_weight')['idx']
            elif select_method == 'order':
                group = zip(idx,data_weight)
                groupvalue = sorted(group, key=(lambda x: float(x[1])))
                idx = list(zip(*groupvalue))[0]
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

        acc_lastround,forget_times = run_cifar.train_Model(self.train_idx,dataset=self.dataset,cal_forget=True,model=self.model)

        # print(forget_times)
        print(len(idx_embedding))
        print(len(forget_times))

        linermodel = RidgeCV().fit(idx_embedding, forget_times)
        self.predicted_forget_time = linermodel.predict(pool_embedding)
        self.acc_record.append(acc_lastround)

        if self.model == 'resnet18':
            torch.save(self.pool_idx, f'./Checkpoints/{self.dataset}-{self.budget}/pool_idx-{self.ebd_method}-{len(self.train_idx)}.pth')
            torch.save(self.train_idx, f'./Checkpoints/{self.dataset}-{self.budget}/train_idx-{self.ebd_method}-{len(self.train_idx)}.pth')
            torch.save(self.predicted_forget_time, f'./Checkpoints/{self.dataset}-{self.budget}/forgettime-{self.ebd_method}-{len(self.train_idx)}.pth')
        elif self.model == 'resnet9':
            torch.save(self.pool_idx,
                       f'./Checkpoints/{self.dataset}-{self.budget}/resnet9-pool_idx-{self.ebd_method}-{len(self.train_idx)}.pth')
            torch.save(self.train_idx,
                       f'./Checkpoints/{self.dataset}-{self.budget}/resnet9-train_idx-{self.ebd_method}-{len(self.train_idx)}.pth')
            torch.save(self.predicted_forget_time,
                       f'./Checkpoints/{self.dataset}-{self.budget}/resnet9-forgettime-{self.ebd_method}-{len(self.train_idx)}.pth')

    def run(self):
        for i in range(self.iter_time):
            print(f'round{i} started!')
            self.UAS()
            print(f'acc：{self.acc_record}')
            print(f'round{i} finished!')
        print(f"final acc：{self.acc_record}")
        return self.acc_record

def UAS_checkpoint(dataset,budget,ebd_method,data_num=3000):

    if dataset in ['cifar10','cifar100']:
        filepath = f'./Checkpoints/{dataset}-{budget}/train_idx-{ebd_method}-{data_num}.pth'
    elif dataset =='fmow':
        filepath = f'./Checkpoints/FMoW-{budget}/train_idx-{data_num}.pth'

    train_idx = torch.load(filepath,map_location=torch.device('cpu'))
    acc_lastround, forget_times = run_cifar.train_Model(train_idx,dataset=dataset)
    print(acc_lastround)
    return acc_lastround

def EA(dataset,data_num = 30000):
    if dataset == 'cifar10':
        filepath = './baselines/EA and SPS/CIFAR10/1/EA-Squareroot-0.05001-' + str(
            data_num)
        init_image_ids = np.loadtxt('./baselines/EA and SPS/CIFAR10/1/' + "init.csv")
        init_image_ids = [int(v) for v in init_image_ids]
    elif dataset == 'cifar100':
        filepath = './baselines/EA and SPS/CIFAR100/1/EA-Squareroot-0.05001-' + str(
            data_num)
        init_image_ids = np.loadtxt('./baselines/EA and SPS/CIFAR100/1/' + "init.csv")
        init_image_ids = [int(v) for v in init_image_ids]
    elif dataset == 'fmow':
        filepath = './baselines/EA and SPS/FMOW/1/EA-Squareroot-0.05001-' + str(
            data_num)
        init_image_ids = np.loadtxt('./baselines/EA and SPS/FMOW/1/' + "init.csv")
        init_image_ids = [int(v) for v in init_image_ids]
    with open(filepath)as f:
        reader = f.readline()
        train_idx = [int(x) for x in reader.split(' ')]
        train_idx += init_image_ids
    acc_lastround, forget_times = run_cifar.train_Model(train_idx,dataset=dataset)
    print(acc_lastround)
    return acc_lastround

def SPS(dataset, data_num = 30000):
    if dataset == 'cifar10':
        filepath = './baselines/EA and SPS/CIFAR10/1/SPS-301-' + str(
            data_num)
        init_image_ids = np.loadtxt('./baselines/EA and SPS/CIFAR10/1/' + "init.csv")
        init_image_ids = [int(v) for v in init_image_ids]
    elif dataset == 'cifar100':
        filepath = './baselines/EA and SPS/CIFAR100/1/SPS-301-' + str(
            data_num)
        init_image_ids = np.loadtxt('./baselines/EA and SPS/CIFAR100/1/' + "init.csv")
        init_image_ids = [int(v) for v in init_image_ids]
    elif dataset == 'fmow':
        filepath = './baselines/EA and SPS/FMOW/1/SPS-301-' + str(
            data_num)
        init_image_ids = np.loadtxt('./baselines/EA and SPS/FMOW/1/' + "init.csv")
        init_image_ids = [int(v) for v in init_image_ids]

    with open(filepath)as f:
        reader = f.readline()
        train_idx = [int(x) for x in reader.split(' ')]
        train_idx+=init_image_ids

    acc_lastround, forget_times = run_cifar.train_Model(train_idx,dataset=dataset)
    print(acc_lastround)
    return acc_lastround

def ACS_U(dataset,data_num = 30000):
    # if dataset == 'cifar10':
    #     filepath = './baselines/EA and SPS/CIFAR10/1/uniform-' + str(
    #         data_num)
    # if dataset == 'cifar100':
    #     filepath = './baselines/EA and SPS/CIFAR100/1/uniform-' + str(
    #         data_num)
    # elif dataset == 'fmow':
    #     filepath = './baselines/EA and SPS/FMOW/1/uniform-' + str(
    #         data_num)

    # with open(filepath)as f:
    #     reader = f.readline()
    #     train_idx = [int(x) for x in reader.split(' ')]
    train_idx = random.sample(range(50000),data_num)
    acc_lastround, forget_times = run_cifar.train_Model(train_idx,dataset=dataset)
    print(acc_lastround)
    return acc_lastround

if __name__ == '__main__':


    parser = argparse.ArgumentParser(description='PyTorch local error training')
    parser.add_argument('--method',default='UAS',help='UAS,SPS,EA,ACS_RD,ACS_Uniform')
    parser.add_argument('--budget_per_round', type=int,default=3000, help='select amount of UAS per round')

    parser.add_argument('--checkpoint', default=None, help='the checkpoint file (data amount) of error stop'
                                                           'to prevent the unexcept error')
    parser.add_argument('--embedding_method',default='datamodel',help='the embedding method of data')
    parser.add_argument('--dataset',default='cifar10',help='dataset')
    parser.add_argument('--model',default='resnet18')
    parser.add_argument('--rerun', default=False,type=bool, help="true means to recalculate the data points. False means to"
                                                       "use the checkpoint in './Checkpoints/[dataset]-[budget]/'")

    args = parser.parse_args()
    run_type = args.method

    if args.dataset in ['cifar10']:
        iter_time = int(50000/args.budget_per_round)
        datanum = [4955, 10955, 16955, 19955, 22955, 25955, 28955, 32955]#datanum+10045=true num
    elif args.dataset in [ 'cifar100']:
        iter_time = int(50000 / args.budget_per_round)
        datanum = [37955]  #4955, 10955, 16955, 19955, 22955, 25955, 28955, 32955 datanum+10045=true num
    elif args.dataset == 'fmow':
        iter_time = int(21404 / args.budget_per_round)
        datanum =[9000,12000,15000]#datanum + 5000 = true num

    acc_all = []
    exp_time = 3
    for i in range(exp_time):
        acc_list = []
        if run_type == 'UAS':
            if not args.rerun:
                datanum = [15000,18000,21000,27000,30000,33000,36000,39000,42000,48000]
                for n in datanum:
                    acc_list.append(UAS_checkpoint(args.dataset,args.budget_per_round,args.embedding_method,n))
            else:
                print(run_type)
                env = iter_purchase(select_method='weight', budget_per_round=args.budget_per_round, iter_time=iter_time,
                                dataset=args.dataset,checkpoint = args.checkpoint,ebd_method = args.embedding_method,model=args.model)
                acc_list = env.run()
        elif run_type == 'EA':
            for n in datanum:
                acc_list.append(EA(args.dataset,data_num=n))
        elif run_type == 'SPS':
            for n in datanum:
                acc_list.append(SPS(args.dataset,data_num=n))
        elif run_type == 'ACS_Uniform':
            if args.dataset == 'cifar100':
                datanum = [36000,39000,43000,48000]
            if args.dataset == 'cifar10':
                datanum = [15000,21000,27000,30000,33000,36000,39000,43000,48000]
            for n in datanum:
                acc_list.append(ACS_U(args.dataset,data_num=n))
        elif run_type == 'all':
            acc_list.append(run_cifar.train_Model(dataset=args.dataset,data_idx=range(50000)))
        else:
            print("method error!")
        acc_all.append(acc_list)
        print(acc_all)

    print(f"method:{run_type}，accuracy:{acc_all}")

