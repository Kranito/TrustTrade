import torch


from tqdm import tqdm
from torchvision import models
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms


device = 'cuda:0'
model=models.vgg16(pretrained=True)
model.to(device)
model.eval()
preprocess =  transforms.Compose([
    transforms.Resize((224, 224)), 
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), 
])


# TRAIN_DATASET = torchvision.datasets.CIFAR10('./Data/cifar10', train=True, download=False,transform=preprocess)
# TEST_DATASET = torchvision.datasets.CIFAR10('./Data/cifar10', train=False, download=False,transform=preprocess)

TRAIN_DATASET = torchvision.datasets.CIFAR100('./Data/cifar100', train=True, download=True,transform=preprocess)
TEST_DATASET = torchvision.datasets.CIFAR100('./Data/cifar100', train=False, download=True,transform=preprocess)



test_embedding_clip = torch.tensor([])





def get_features(Dtype = 'train',data_set = 'cifar10'):

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
            images = images.to(device) 
            features = model(images)

            all_features.append(features)
            all_labels.append(labels)

    return torch.cat(all_features).cpu().numpy(), torch.cat(all_labels).cpu().numpy()


a,b=get_features('train','cifar10')
torch.save(a, './cifar10-vgg16.pt',_use_new_zipfile_serialization=False)

