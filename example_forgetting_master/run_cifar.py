import pdb
import argparse
import numpy as np
import numpy.random as npr
import time
import os
import sys
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import Subset

from torchvision import datasets, transforms

from order_examples_by_forgetting import order_examples
from Cutout.model.resnet import ResNet18
from Cutout.model.wide_resnet import WideResNet



def run_cifar(data_idx, output_dir = '../Data/forget_scores' , dataset = 'cifar10'):
    '''
    :input: selected data index, output dir
    :return: data index and  forget_times
    '''
    model_options = ['resnet18', 'wideresnet']
    dataset_options = ['cifar10', 'cifar100']

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

        print('\n=> Training Epoch #%d' % (epoch))

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

        model.eval()

        for batch_idx, batch_start_ind in enumerate(
                range(0, len(test_dataset.test_labels), test_batch_size)):

            # Get batch inputs and targets
            transformed_testset = []
            for ind in range(
                    batch_start_ind,
                    min(
                        len(test_dataset.test_labels),
                        batch_start_ind + test_batch_size)):
                transformed_testset.append(test_dataset.__getitem__(ind)[0])
            inputs = torch.stack(transformed_testset)
            targets = torch.LongTensor(
                np.array(
                    test_dataset.test_labels)[batch_start_ind:batch_start_ind +
                                                              test_batch_size].tolist())

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

        # Add test accuracy to dict
        acc = 100. * correct.item() / total
        index_stats = example_stats.get('test', [[], []])
        index_stats[1].append(100. * correct.item() / float(total))
        example_stats['test'] = index_stats
        print("\n| Validation Epoch #%d\t\t\tLoss: %.4f Acc@1: %.2f%%" %
              (epoch, loss.item(), acc))

        # Save checkpoint when best model
        if acc > best_acc:
            print('| Saving Best model...\t\t\tTop1 = %.2f%%' % (acc))
            state = {
                'acc': acc,
                'epoch': epoch,
            }
            save_point = os.path.join(args.output_dir, 'checkpoint', args.dataset)
            os.makedirs(save_point, exist_ok=True)
            torch.save(state, os.path.join(save_point, save_fname + '.t7'))
            best_acc = acc

    class set_args():
        def __init__(self,dataset,output_dir):
            self.dataset= dataset
            self.model='resnet18'
            self.batch_size=128
            self.epochs=200
            self.learning_rate=0.1
            self.data_augmentation = True
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
    args = set_args(dataset,output_dir)
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

        test_dataset = datasets.CIFAR100(
            root='/tmp/data/',
            train=False,
            transform=test_transform,
            download=True)

    train_indx = np.array(range(len(train_dataset)))

    print('Training on ' + str(len(train_dataset)) + ' examples')

    # Setup model
    if args.model == 'resnet18':
        model = ResNet18(num_classes=num_classes)
    elif args.model == 'wideresnet':
        if args.dataset == 'svhn':
            model = WideResNet(
                depth=16, num_classes=num_classes, widen_factor=8, dropRate=0.4)
        else:
            model = WideResNet(
                depth=28, num_classes=num_classes, widen_factor=10, dropRate=0.3)
    else:
        print(
            'Specified model not recognized. Options are: resnet18 and wideresnet')

    # Setup loss
    model = model.cuda()
    criterion = nn.CrossEntropyLoss().cuda()
    criterion.__init__(reduction='none')

    # Setup optimizer
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

    # Initialize dictionary to save statistics for every example presentation
    example_stats = {}

    best_acc = 0
    elapsed_time = 0
    for epoch in range(args.epochs):
        start_time = time.time()

        train(args, model, device, train_dataset, model_optimizer, epoch,
              example_stats)
        test(epoch, model, device, example_stats)

        epoch_time = time.time() - start_time
        elapsed_time += epoch_time
        print('| Elapsed time : %d:%02d:%02d' % (get_hms(elapsed_time)))

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
    output_name = 'cifar10_sorted'
    input_fname_args = ['dataset',args.vars()['dataset'] ,'data_augmentation',args.vars()['data_augmentation'] ]

    data_idx, forget_times = order_examples(data_idx,input_dir,output_dir,output_name,input_fname_args)

    #delete processing files
    os.remove(fname + "__stats_dict.pkl")
    os.remove(fname + "__best_acc.txt")

    return best_acc,forget_times

if __name__ == '__main__':
    print(run_cifar(range(1000)))