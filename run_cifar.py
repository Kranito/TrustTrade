import pdb
import argparse
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1,3,4"
import numpy as np
import numpy.random as npr
import time

import sys
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models
from torch.autograd import Variable
from torch.nn import CrossEntropyLoss
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import Subset

from torchvision import datasets, transforms
from sklearn.metrics import roc_auc_score
import Models.Resnet9
from order_examples_by_forgetting import order_examples
from Resnet18 import ResNet18
from example_forgetting_master.Cutout.model.wide_resnet import WideResNet
import example_forgetting_master.Cutout.model.resnet as cifarResNet


def train_Model(data_idx, cal_forget =False,output_dir = '../Data/forget_scores' , dataset = 'cifar10',model='resnet18'):
    '''
    :input: selected data index, output dir
    :return: data index and  forget_times
    '''
    model_options = ['resnet18', 'wideresnet']
    dataset_options = ['cifar10', 'fmow','cifar100']

    # Format time for printing purposes
    def get_hms(seconds):
        m, s = divmod(seconds, 60)
        h, m = divmod(m, 60)

        return h, m, s

    # Train model for one epoch
    #
    # example_stats: dictionary containing statistics accumulated over every presentation of example
    #
    def train(args, model, device, trainset, model_optimizer, epoch,
              example_stats):
        train_loss = 0.
        correct = 0.
        total = 0.

        model.train()

        # Get permutation to shuffle trainset
        trainset_permutation_inds = npr.permutation(
            np.arange(len(trainset)))

        # print('\n=> Training Epoch #%d' % (epoch))

        batch_size = args.batch_size
        for batch_idx, batch_start_ind in enumerate(
                range(0, len(trainset), batch_size)):

            # Get trainset indices for batch
            batch_inds = trainset_permutation_inds[batch_start_ind:
                                                   batch_start_ind + batch_size]

            # Get batch inputs and targets, transform them appropriately
            transformed_trainset = []
            targets = []
            for ind in batch_inds:
                transformed_trainset.append(trainset.__getitem__(ind)[0])
                targets.append(trainset.__getitem__(ind)[1])
            inputs = torch.stack(transformed_trainset)
            targets = torch.LongTensor(targets)

            # Map to available device
            inputs, targets = inputs.to(device), targets.to(device)

            # Forward propagation, compute loss, get predictions
            model_optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            _, predicted = torch.max(outputs.data, 1)

            # Update statistics and loss
            acc = predicted == targets
            for j, index in enumerate(batch_inds):

                # Get index in original dataset (not sorted by forgetting)
                index_in_original_dataset = train_indx[index]

                # Compute missclassification margin
                output_correct_class = outputs.data[j, targets[j].item()]
                sorted_output, _ = torch.sort(outputs.data[j, :])
                if acc[j]:
                    # Example classified correctly, highest incorrect class is 2nd largest output
                    output_highest_incorrect_class = sorted_output[-2]
                else:
                    # Example misclassified, highest incorrect class is max output
                    output_highest_incorrect_class = sorted_output[-1]
                margin = output_correct_class.item(
                ) - output_highest_incorrect_class.item()

                # Add the statistics of the current training example to dictionary
                index_stats = example_stats.get(index_in_original_dataset,
                                                [[], [], []])
                index_stats[0].append(loss[j].item())
                index_stats[1].append(acc[j].sum().item())
                index_stats[2].append(margin)
                example_stats[index_in_original_dataset] = index_stats

            # Update loss, backward propagate, update optimizer
            loss = loss.mean()
            train_loss += loss.item()
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()
            loss.backward()
            model_optimizer.step()

            sys.stdout.write('\r')
            sys.stdout.write(
                '| Epoch [%3d/%3d] Iter[%3d/%3d]\t\tLoss: %.4f Acc@1: %.3f%%' %
                (epoch, args.epochs, batch_idx + 1,
                 (len(trainset) // batch_size) + 1, loss.item(),
                 100. * correct.item() / total))
            sys.stdout.flush()
            sys.stdout.write('\r')

            # Add training accuracy to dict
            index_stats = example_stats.get('train', [[], []])
            index_stats[1].append(100. * correct.item() / float(total))
            example_stats['train'] = index_stats
    def train_fmow(args, model, device, trainset, model_optimizer, epoch,scaler,
              example_stats):
        train_loss = 0.
        correct = 0.
        total = 0.

        model.train()

        # Get permutation to shuffle trainset
        trainset_permutation_inds = npr.permutation(
            np.arange(len(trainset)))

        # print('\n=> Training Epoch #%d' % (epoch))

        batch_size = args.batch_size
        for batch_idx, batch_start_ind in enumerate(
                range(0, len(trainset), batch_size)):

            # Get trainset indices for batch
            batch_inds = trainset_permutation_inds[batch_start_ind:
                                                   batch_start_ind + batch_size]

            # Get batch inputs and targets, transform them appropriately
            transformed_trainset = []
            targets = []
            for ind in batch_inds:
                transformed_trainset.append(trainset.__getitem__(ind)[0])
                targets.append(trainset.__getitem__(ind)[1])
            inputs = torch.stack(transformed_trainset)
            targets = torch.LongTensor(targets)

            # Map to available device
            inputs, targets = inputs.to(device), targets.to(device)

            # Forward propagation, compute loss, get predictions
            model_optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            _, predicted = torch.max(outputs.data, 1)

            # Update statistics and loss
            acc = predicted == targets
            for j, index in enumerate(batch_inds):

                # Get index in original dataset (not sorted by forgetting)
                index_in_original_dataset = train_indx[index]

                # Compute missclassification margin
                output_correct_class = outputs.data[j, targets[j].item()]
                sorted_output, _ = torch.sort(outputs.data[j, :])
                if acc[j]:
                    # Example classified correctly, highest incorrect class is 2nd largest output
                    output_highest_incorrect_class = sorted_output[-2]
                else:
                    # Example misclassified, highest incorrect class is max output
                    output_highest_incorrect_class = sorted_output[-1]
                margin = output_correct_class.item(
                ) - output_highest_incorrect_class.item()

                # Add the statistics of the current training example to dictionary
                index_stats = example_stats.get(index_in_original_dataset,
                                                [[], [], []])
                index_stats[0].append(loss[j].item())
                index_stats[1].append(acc[j].sum().item())
                index_stats[2].append(margin)
                example_stats[index_in_original_dataset] = index_stats

            # Update loss, backward propagate, update optimizer
            loss = loss.mean()
            train_loss += loss.item()
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()
            scaler.scale(loss).backward()
            scaler.step(model_optimizer)
            scaler.update()
            scheduler.step()
            # loss.backward()
            # model_optimizer.step()

            sys.stdout.write('\r')
            sys.stdout.write(
                '| Epoch [%3d/%3d] Iter[%3d/%3d]\t\tLoss: %.4f Acc@1: %.3f%%' %
                (epoch, args.epochs, batch_idx + 1,
                 (len(trainset) // batch_size) + 1, loss.item(),
                 100. * correct.item() / total))
            sys.stdout.flush()
            sys.stdout.write('\r')

            # Add training accuracy to dict
            index_stats = example_stats.get('train', [[], []])
            index_stats[1].append(100. * correct.item() / float(total))
            example_stats['train'] = index_stats
    # Evaluate model predictions on heldout test data
    #
    # example_stats: dictionary containing statistics accumulated over every presentation of example
    #
    def test(epoch, model, device, example_stats):
        nonlocal best_acc
        test_loss = 0.
        correct = 0.
        total = 0.
        test_batch_size = 32
        y=[]
        y_pred_prob = torch.tensor([]).to(device)

        model.eval()

        for batch_idx, batch_start_ind in enumerate(
                range(0, len(test_dataset), test_batch_size)):

            # Get batch inputs and targets
            transformed_testset = []
            targets=[]
            for ind in range(
                    batch_start_ind,
                    min(
                        len(test_dataset),
                        batch_start_ind + test_batch_size)):
                transformed_testset.append(test_dataset.__getitem__(ind)[0])
                targets.append(test_dataset.__getitem__(ind)[1])
            inputs = torch.stack(transformed_testset)
            targets = torch.LongTensor(
                targets)

            # if dataset =='fmow':
            #     transformed_testset = []
            #     targets = []
            #     for ind in batch_inds:
            #         transformed_trainset.append(trainset.__getitem__(ind)[0])
            #         targets.append(trainset.__getitem__(ind)[1])
            #     inputs = torch.stack(transformed_trainset)
            #     targets = torch.LongTensor(targets)


            # Map to available device
            inputs, targets = inputs.to(device), targets.to(device)

            # Forward propagation, compute loss, get predictions
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss = loss.mean()
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()
            y+=targets.tolist()
            y_pred_prob = torch.cat((y_pred_prob,outputs.data),dim=0)

        # Add test accuracy to dict
        y_pred_prob = torch.nn.functional.softmax(y_pred_prob.cpu(), dim=1)
        auc = roc_auc_score(y, y_pred_prob, multi_class='ovr', average='macro')
        acc = 100. * correct.item() / total
        index_stats = example_stats.get('test', [[], []])
        index_stats[1].append(100. * correct.item() / float(total))
        example_stats['test'] = index_stats
        if epoch ==199 or epoch == 200:
            print("\n| Validation Epoch #%d\t\t\tLoss: %.4f Acc@1: %.2f%%" %
                  (epoch, loss.item(), acc))
            print(f"AUC:{auc}")

        # Save checkpoint when best model
        if acc > best_acc:
            # print('| Saving Best model...\t\t\tTop1 = %.2f%%' % (acc))
            state = {
                'acc': acc,
                'epoch': epoch,
            }
            save_point = os.path.join(args.output_dir, 'checkpoint', args.dataset)
            os.makedirs(save_point, exist_ok=True)
            torch.save(state, os.path.join(save_point, save_fname + '.t7'))
            best_acc = acc

    class set_args():
        def __init__(self,dataset,output_dir,model):
            self.dataset= dataset
            if dataset =='cifar10':
                self.batch_size = 128
                self.epochs = 200
                self.learning_rate = 0.1
                self.data_augmentation = True
            elif dataset =='cifar100':
                self.batch_size = 128
                self.epochs = 200
                self.learning_rate = 0.1
                self.data_augmentation = True
            elif dataset == 'fmow':
                self.batch_size = 256
                self.epochs = 15
                self.learning_rate = 0.4
                self.data_augmentation = False
                self.lr_peak_epoch = 6
            self.model=model
            self.use_cuda = True
            self.cuda = None
            self.optimizer = 'sgd'
            self.input_dir = dataset+'_results/'
            self.output_dir = output_dir

        def vars(self):
            return {'dataset':self.dataset, 'data_augmentation':self.data_augmentation}

    # Enter all arguments that you want to be in the filename of the saved output
    ordered_args = [
        'dataset', 'data_augmentation'
    ]

    # Parse arguments and setup name of output file with forgetting stats
    args = set_args(dataset,output_dir,model)
    args_dict = args.vars()
    print(args_dict)
    save_fname = '__'.join(
        '{}_{}'.format(arg, args_dict[arg]) for arg in ordered_args)

    # Set appropriate devices
    args.cuda = args.use_cuda and torch.cuda.is_available()
    use_cuda = args.cuda
    device = torch.device("cuda" if use_cuda else "cpu")
    cudnn.benchmark = False  # Should make training go faster for large models


    # Image Preprocessing
    normalize = transforms.Normalize(
        mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
        std=[x / 255.0 for x in [63.0, 62.1, 66.7]])

    # Setup train transforms
    train_transform = transforms.Compose([])
    if args.data_augmentation:
        train_transform.transforms.append(transforms.RandomCrop(32, padding=4))
        train_transform.transforms.append(transforms.RandomHorizontalFlip())
    train_transform.transforms.append(transforms.ToTensor())
    train_transform.transforms.append(normalize)

    # Setup test transforms
    test_transform = transforms.Compose([transforms.ToTensor(), normalize])

    os.makedirs(args.output_dir, exist_ok=True)

    # Load the appropriate train and test datasets
    device_id = [0,1]
    if args.dataset == 'cifar10':
        num_classes = 10
        train_dataset = datasets.CIFAR10(
            root='/tmp/data/',
            train=True,
            transform=train_transform,
            download=True)
        train_dataset = Subset(train_dataset,data_idx)
        print(len(train_dataset))

        test_dataset = datasets.CIFAR10(
            root='/tmp/data/',
            train=False,
            transform=test_transform,
            download=True)
    elif args.dataset == 'cifar100':
        num_classes = 100
        train_dataset = datasets.CIFAR100(
            root='/tmp/data/',
            train=True,
            transform=train_transform,
            download=True)
        train_dataset = Subset(train_dataset, data_idx)
        print(len(train_dataset))

        test_dataset = datasets.CIFAR100(
            root='/tmp/data/',
            train=False,
            transform=test_transform,
            download=True)
    elif args.dataset == 'fmow':
        device_id = [0, 1, 2, 3, 4, 5, 6, 7]
        import fmow
        num_classes = 62
        train_dataset,test_dataset = fmow.get_dataset()
        train_dataset = Subset(train_dataset, data_idx)

    train_indx = np.array(range(len(train_dataset)))

    print('Training on ' + str(len(train_dataset)) + ' examples')

    # Setup model
    if args.model == 'resnet18':
        if args.dataset=='fmow':
            model = ResNet18(num_classes=num_classes,dataset=args.dataset)
        elif args.dataset in ['cifar10','cifar100']:
            model = cifarResNet.ResNet18(num_classes=num_classes)

    elif args.model == 'wideresnet':
        if args.dataset == 'svhn':
            model = WideResNet(
                depth=16, num_classes=num_classes, widen_factor=8, dropRate=0.4)
        else:
            model = WideResNet(
                depth=28, num_classes=num_classes, widen_factor=10, dropRate=0.3)
    elif args.model == 'resnet9':
        model = Models.Resnet9.resnet(num_class=num_classes)
    else:
        print(
            'Specified model not recognized. Options are: resnet18 and wideresnet')

    # Setup loss

    model=torch.nn.DataParallel(model,device_ids=device_id)
    model = model.cuda(device=device_id[0])
    criterion = nn.CrossEntropyLoss().cuda()
    criterion.__init__(reduction='none')

    # Setup optimizer
    if args.dataset in {'cifar10','cifar100'} :
        if args.optimizer == 'adam':
            model_optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        elif args.optimizer == 'sgd':
            model_optimizer = torch.optim.SGD(
                model.parameters(),
                lr=args.learning_rate,
                momentum=0.9,
                nesterov=True,
                weight_decay=5e-4)
            scheduler = MultiStepLR(
                model_optimizer, milestones=[60, 120, 160], gamma=0.2)
        else:
            print('Specified optimizer not recognized. Options are: adam and sgd')
    elif args.dataset == 'fmow':
        if args.optimizer == 'adam':
            model_optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        elif args.optimizer == 'sgd':
            model_optimizer = torch.optim.SGD(
                model.parameters(),
                lr=args.learning_rate,
                momentum=0.9,
                nesterov=True,
                weight_decay=1e-3)
            lr_peak_epoch = args.lr_peak_epoch
            iters_per_epoch = int(21404/args.batch_size)
            # Cyclic LR with single triangle
            lr_schedule = np.interp(np.arange((args.epochs + 1) * iters_per_epoch),
                                    [0, lr_peak_epoch * iters_per_epoch, args.epochs * iters_per_epoch],
                                    [0, 1, 0])
            scheduler = torch.optim.lr_scheduler.LambdaLR(model_optimizer, lr_schedule.__getitem__)
            scaler = torch.cuda.amp.GradScaler()

        else:
            print('Specified optimizer not recognized. Options are: adam and sgd')

    # Initialize dictionary to save statistics for every example presentation
    example_stats = {}

    best_acc = 0
    elapsed_time = 0
    for epoch in range(args.epochs):
        start_time = time.time()

        if args.dataset in ['cifar10','cifar100']:
            train(args, model, device, train_dataset, model_optimizer, epoch,
                example_stats)
        elif args.dataset == 'fmow':
            train_fmow(args, model, device, train_dataset, model_optimizer, epoch,scaler,
                  example_stats)
        test(epoch, model, device, example_stats)

        epoch_time = time.time() - start_time
        elapsed_time += epoch_time
        # print('| Elapsed time : %d:%02d:%02d' % (get_hms(elapsed_time)))

        # Update optimizer step
        if args.optimizer == 'sgd':
            scheduler.step(epoch)

        # Save the stats dictionary
        fname = os.path.join(args.output_dir, save_fname)
        with open(fname + "__stats_dict.pkl", "wb") as f:
            pickle.dump(example_stats, f)

        # Log the best train and test accuracy so far
        with open(fname + "__best_acc.txt", "w") as f:
            f.write('train test \n')
            f.write(str(max(example_stats['train'][1])))
            f.write(' ')
            f.write(str(max(example_stats['test'][1])))


    input_dir = output_dir ='../Data/forget_scores'
    output_name = f'{args.dataset}_sorted'
    input_fname_args = ['dataset',args.vars()['dataset'] ,'data_augmentation',args.vars()['data_augmentation'] ]

    if cal_forget == True:
        data_idx, forget_times = order_examples(data_idx,input_dir,output_dir,output_name,input_fname_args)
    else:
        forget_times = None

    #delete processing files
    os.remove(fname + "__stats_dict.pkl")
    os.remove(fname + "__best_acc.txt")

    return best_acc,forget_times

if __name__ == '__main__':


    print(train_Model(range(20000),dataset='fmow'))