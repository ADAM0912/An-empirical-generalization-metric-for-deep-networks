import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, ConcatDataset, Subset
import copy
import argparse
import torchvision
from torchvision import transforms, datasets
import pandas as pd
import os
import shutil
from utils import EarlyStopping, largest_unstructured
import torch.nn.utils.prune as prune
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
from utils import CustomDataset
from scipy.special import softmax
from torch.utils.data import DataLoader, TensorDataset
import torchvision.transforms.functional as TF




def prepare_dataset(dataset,transform):


    augmented_images = []
    augmented_labels = []
    for data, target in dataset:
        augmented_image = transform(data)
        augmented_images.append(augmented_image)
        augmented_labels.append(target)

    images_tensor = torch.stack(augmented_images)
    labels_tensor = torch.tensor(augmented_labels)
    return images_tensor, labels_tensor

def save_dataset_as_png(dataset, ssim, zeroshot):
    directory = f'augmented_dataset_halfs/cifar100_{ssim:.2f}_{zeroshot}'
    if not os.path.exists(directory):
        os.makedirs(directory)  # Create the directory if it does not exist

    for i, (image, label) in enumerate(dataset):
        # Convert tensor to PIL Image
        if label>=50:
            image = TF.to_pil_image(image)
            label = label.numpy()
        # Construct file path

        class_directory = os.path.join(directory, str(label))
        os.makedirs(class_directory, exist_ok=True)
        output_filename = os.path.join(class_directory, f'{i}.png')
        # Save the image
        image.save(output_filename, format='PNG')
        # breakpoint()
        # torchvision.utils.save_image(image, output_filename)

ssim_list = [0.60,0.65, 0.70, 0.75, 0.80,0.85, 0.90,0.95,1.0]
n_zeroshot = [0,10, 20, 30, 40, 50]
# ssim_list = [1.0]
# n_zeroshot = [100]

transform = transforms.Compose([  # Convert tensor to PIL Image to use more complex transforms
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
    transforms.RandomResizedCrop(32, scale=(0.8, 1.0), ratio=(0.75, 1.33)),
    transforms.ToTensor()  # Convert back to tensor
])

for i, ssim in enumerate(ssim_list):
    for zeroshot in n_zeroshot:
        if zeroshot==0:
            if ssim == 1.0:
                cifar_dataset = datasets.CIFAR100(root='./data', train=False, download=True)
                valid_indices = [i for i, (_, label) in enumerate(cifar_dataset) if label < 50]
                valid_dataset = Subset(cifar_dataset, valid_indices)
                save_dataset_as_png(valid_dataset, ssim, zeroshot)
                print('ssim=', ssim, 'zeroshot=', zeroshot)
            else:
                cifar_dataset = CustomDataset(root_dir=f'generated_datasets/cifar100_{ssim:.2f}')
                valid_indices = [i for i, (_, label) in enumerate(cifar_dataset) if label < 50]
                valid_dataset = Subset(cifar_dataset, valid_indices)
                save_dataset_as_png(valid_dataset, ssim, zeroshot)
                print('ssim=', ssim, 'zeroshot=', zeroshot)
        else:
            if ssim == 1.0:
                cifar_dataset = datasets.CIFAR100(root='./data', train=False, download=True)
                aug_indices = [i for i, (_, label) in enumerate(cifar_dataset) if 50 <= label < zeroshot+50]
                aug_dataset = Subset(cifar_dataset, aug_indices)
                x_augmented, y_augmented = prepare_dataset(aug_dataset,transform)
                aug_dataset = TensorDataset(x_augmented, y_augmented)
                valid_indices = [i for i, (_, label) in enumerate(cifar_dataset) if label < 50]
                valid_dataset =Subset(cifar_dataset, valid_indices)
                dataset = ConcatDataset([valid_dataset, aug_dataset])
                save_dataset_as_png(dataset,ssim,zeroshot)
                print('ssim=',ssim,'zeroshot=',zeroshot)
            else:
                cifar_dataset = CustomDataset(root_dir=f'generated_datasets/cifar100_{ssim:.2f}')
                aug_indices = [i for i, (_, label) in enumerate(cifar_dataset) if 50 <= label < zeroshot+50]
                aug_dataset = Subset(cifar_dataset, aug_indices)
                x_augmented, y_augmented = prepare_dataset(aug_dataset,transform)
                aug_dataset = TensorDataset(x_augmented, y_augmented)
                valid_indices = [i for i, (_, label) in enumerate(cifar_dataset) if label < 50]
                valid_dataset =Subset(cifar_dataset, valid_indices)
                dataset = ConcatDataset([valid_dataset, aug_dataset])
                save_dataset_as_png(dataset,ssim,zeroshot)
                print('ssim=', ssim, 'zeroshot=', zeroshot)
