import torch
import torchvision
import torchvision.transforms as transforms
import sys
from albumentation_transforms import albumentations_train_transforms
import numpy as np

## Get Train and Test data

def get_train_test(classes):
    mean = np.array([0.4914, 0.4822, 0.4465])
    std = np.array([0.2023, 0.1994, 0.2010])
	
    train_transform = albumentations_train_transforms(mean, std, p=1.0)

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=mean,
            std=std
        )])
    

    SEED = 1

    # CUDA?
    cuda = torch.cuda.is_available()
    print("CUDA Available?", cuda)

    # For reproducibility
    torch.manual_seed(SEED)

    if cuda:
        torch.cuda.manual_seed(SEED)

    # dataloader arguments - something you'll fetch these from cmdprmt
    dataloader_args = dict(shuffle=True, batch_size=128, num_workers=4, pin_memory=True) if cuda else dict(shuffle=True,
                                                                                                           batch_size=64)

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=train_transform)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=test_transform)

    trainloader = torch.utils.data.DataLoader(trainset, **dataloader_args)
    testloader = torch.utils.data.DataLoader(testset, **dataloader_args)

    return trainloader, testloader, test_transform