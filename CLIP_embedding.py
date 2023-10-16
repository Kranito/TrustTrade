import torch

import clip
from tqdm import tqdm

from torch.utils.data import DataLoader
import torchvision



device = 'cuda:0'
model,preprocess = clip.load("ViT-B/32", device=device)


# TRAIN_DATASET = torchvision.datasets.CIFAR10('./Data/cifar10', train=True, download=False,transform=preprocess)
# TEST_DATASET = torchvision.datasets.CIFAR10('./Data/cifar10', train=False, download=False,transform=preprocess)

TRAIN_DATASET = torchvision.datasets.CIFAR100('./Data/cifar100', train=True, download=True,transform=preprocess)
TEST_DATASET = torchvision.datasets.CIFAR100('./Data/cifar100', train=False, download=True,transform=preprocess)



test_embedding_clip = torch.tensor([])





def get_features(Dtype = 'train',data_set = 'cifar10'):
    '''
    return correspond embedding
    :param dataset:
    :return:
    '''
    all_features = []
    all_labels = []
    print(type(Dtype))
    if data_set == 'cifar10':
        TRAIN_DATASET = torchvision.datasets.CIFAR10('./Data/cifar10', train=True, download=False, transform=preprocess)
        TEST_DATASET = torchvision.datasets.CIFAR10('./Data/cifar10', train=False, download=False, transform=preprocess)
    elif data_set == 'cifar100':
        TRAIN_DATASET = torchvision.datasets.CIFAR100('./Data/cifar100', train=True, download=True,
                                                      transform=preprocess)
        TEST_DATASET = torchvision.datasets.CIFAR100('./Data/cifar100', train=False, download=True,
                                                     transform=preprocess)

    elif data_set == 'fmow':
        import fmow
        TRAIN_DATASET,TEST_DATASET = fmow.get_dataset(transform =preprocess)


    if Dtype == 'train':
        dataset = TRAIN_DATASET
    elif Dtype == 'test':
        dataset = TEST_DATASET
    else:
        print(Dtype,'please input the datatype')

    with torch.no_grad():
        for images, labels in tqdm(DataLoader(dataset, batch_size=100)):
            features = model.encode_image(images.to(device))

            all_features.append(features)
            all_labels.append(labels)

    return torch.cat(all_features).cpu().numpy(), torch.cat(all_labels).cpu().numpy()


a,b=get_features('train','cifar100')
torch.save(a, './cifar100-clip.pt',_use_new_zipfile_serialization=False)

# def get_decision_value(dataset='test'):
#     if dataset == 'test':
#         X, lables = get_features(TEST_DATASET)
#     elif dataset == 'train':
#         X, lables = get_features(TRAIN_DATASET)
#     else:
#         print("wrong dataset")
#
#     import csv
#     if not os.path.exists("decision_value_test.csv"):
#         acc, badpoints, weight = ResNet1.train_and_return_error_points(range(20000), range(10000))
#         Y = torch.cat(badpoints, 0).cpu().tolist()
#         X = np.array(X, dtype='float')
#         # feature normalize
#         print(pstdev(X[2]))
#         for i in range(len(X)):
#             X[i] -= mean(X[i])
#             X[i] /= pstdev(X[i])
#         Y = np.array(Y)
#
#         import svm
#         clf, best_c = svm.find_best_svm(X, Y)
#
#         decision_value = clf.decision_function(X)
#         with open("decision_value_test.csv", 'a', newline='') as f:
#             writer = csv.writer(f)
#             writer.writerow(decision_value)
#         f.close()
#     else:
#         decision_value = pd.read_csv("decision_value_test.csv",header=None)
#         decision_value = np.array(decision_value)
#
#     print(len(decision_value))
#     return decision_value


# decision_value = get_decision_value()
# # torch.save("./decision_value_test.pt",decision_value)
#
#
#
# test_embedding_PCA = pd.read_csv("PCA_embedding_test.csv",header=None,prefix="PC")
# test_embedding_PCA = np.array(test_embedding_PCA)
# # print(test_embedding_PCA)
# # print(test_embedding_PCA[0][1:])
#
# print(len(test_embedding_PCA))
#
# embedding_mean = []
# embedding_dev = []

# print(test_embedding_PCA[0])
# for testpoint_idx in range(0,len(test_embedding_PCA)):
#     embedding_mean.append(mean(test_embedding_PCA[testpoint_idx][1:]))
#     embedding_dev.append(pstdev(test_embedding_PCA[testpoint_idx][1:]))
#
# print(len(embedding_mean),len(embedding_dev))
#
#
#
# import matplotlib.pyplot as plt
# plt.scatter(embedding_mean,decision_value)
# plt.xlabel("mean")
# plt.ylabel("decisionValue")
#
# plt.savefig("mean-decisionvalue.png")
# plt.show()
# plt.clf()
#
# plt.scatter(embedding_dev,decision_value)
# plt.xlabel("dev")
# plt.ylabel("decisionValue")
#
# plt.savefig("dev-decisionvalue.png")
# plt.show()
