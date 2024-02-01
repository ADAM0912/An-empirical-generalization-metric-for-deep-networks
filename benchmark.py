import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, ConcatDataset, Subset
import copy
import argparse
from torchvision import transforms, datasets
import pandas as pd
import os
import shutil
from utils import EarlyStopping, Classifier, save_checkpoint
import torch.nn.utils.prune as prune
import h5py
import torch.nn.functional as F




def train(args, model, model_classifier, device, train_loader):
    sum_loss, sum_correct = 0, 0
    optimizerCls = optim.SGD(model_classifier.parameters(), args.learningrate, momentum=args.momentum)
    model_classifier.train()

    for i, (data, target) in enumerate(train_loader):
        if args.fc:
            data, target = data.to(device).view(data.size(0), -1), target.to(device)
        else:
            data, target = data.to(device), target.to(device)
        feature_output = model(data)

        output = model_classifier(feature_output.detach())
        criterion = nn.CrossEntropyLoss()
        loss = criterion(output, target)
        pred = output.max(1)[1]
        sum_correct += pred.eq(target).sum().item()
        error = 1 - (sum_correct / len(train_loader.dataset))
        sum_loss += len(data) * loss.item()

        optimizerCls.zero_grad()
        loss.backward()
        optimizerCls.step()
    return sum_loss / len(train_loader.dataset), error

def load_data(split, dataset_name, datadir, nchannels,zeroshot_number):
    if dataset_name == 'MNIST':
        normalize = transforms.Normalize(mean=[0.131], std=[0.289])
    else:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    tr_transform = transforms.Compose([transforms.Resize(32), transforms.ToTensor(), normalize])

    # for CLIP
    # tr_transform = transforms.Compose([
    #     transforms.Resize((224, 224)),  # CLIP expects 224x224 images
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    # ])
    val_transform = tr_transform

    get_dataset = getattr(datasets, dataset_name)
    if dataset_name == 'SVHN':
        if split == 'train':
            dataset = get_dataset(root=datadir, split='train', download=True, transform=tr_transform)
        else:
            dataset = get_dataset(root=datadir, split='test', download=True, transform=val_transform)
    else:
        if split == 'train':
            train_dataset = get_dataset(root=datadir, train=True, download=True, transform=tr_transform)
            train_indices = [i for i, (_, label) in enumerate(train_dataset) if label < 50+zeroshot_number]
            dataset = Subset(train_dataset, train_indices)
            # breakpoint()
        else:
            # dataset1 = get_dataset(root=datadir, train=True, download=True, transform=tr_transform)
            test_dataset = get_dataset(root=datadir, train=False, download=True, transform=val_transform)
            # dataset = ConcatDataset([dataset1, dataset2])
            # dataset = test_dataset
            train_indices = [i for i, (_, label) in enumerate(test_dataset) if label < 50+zeroshot_number]
            dataset = Subset(test_dataset, train_indices)


    return dataset

def validate(args, model, classifier, device, val_loader, criterion):
    # num_classes = len(val_loader.dataset.classes)  # Assuming your dataset provides a 'classes' attribute
    num_classes = args.zeroshot+50
    class_correct = torch.zeros(num_classes, dtype=torch.float32).to(device)
    class_total = torch.zeros(num_classes, dtype=torch.float32).to(device)

    sum_loss, sum_correct = 0, 0

    # switch to evaluation mode
    model.eval()
    classifier.eval()
    with torch.no_grad():
        for i, (data, target) in enumerate(val_loader):

            if args.fc:
                data, target = data.to(device).view(data.size(0), -1), target.to(device)
            else:
                data, target = data.to(device), target.to(device)

            # compute the output
            feature_output = model(data)
            output = classifier(feature_output)
            # breakpoint()

            # compute the classification error and loss
            pred = output.max(1)[1]
            sum_correct += pred.eq(target).sum().item()
            sum_loss += len(data) * criterion(output, target).item()

            # Update class-wise correct and total counts
            for t, p in zip(target.view(-1), pred.view(-1)):
                class_correct[t] += (t == p).item()
                class_total[t] += 1
            # breakpoint()
    # Calculate error rate for each class
    class_error_rates = 1 - (class_correct / class_total)

    overall_error_rate = 1 - (sum_correct / len(val_loader.dataset))
    # breakpoint()
    return overall_error_rate, sum_loss / len(val_loader.dataset), class_error_rates


def main():

    # settings
    parser = argparse.ArgumentParser(description='Training a fully connected NN with one hidden layer')
    parser.add_argument('--no-cuda', default=False, action='store_true',
                        help='disables CUDA training')
    parser.add_argument('--datadir', default='datasets', type=str,
                        help='path to the directory that contains the datasets (default: datasets)')
    parser.add_argument('--dataset', default='CIFAR100', type=str,
                        help='name of the dataset (options: MNIST | CIFAR10 | CIFAR100 | SVHN, default: CIFAR10)')
    parser.add_argument('--nunits', default=1024, type=int,
                        help='number of hidden units (default: 1024)')
    parser.add_argument('--epochs', default=1000, type=int,
                        help='number of epochs to train (default: 1000)')
    parser.add_argument('--stopcond', default=0.001, type=float,
                        help='stopping condtion based on the cross-entropy loss (default: 0.01)')
    parser.add_argument('--batchsize', default=64, type=int,
                        help='input batch size (default: 64)')
    parser.add_argument('--learningrate', default=0.0001, type=float,
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='momentum (default: 0.9)')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight decay')
    parser.add_argument('--hid_dim', default=512, type=int,
                        help='hid_dim (default: 512)')
    parser.add_argument('--out_dim', default=512, type=int,
                        help='out_dim (default: 512)')
    parser.add_argument('--latent_dim', default=512, type=int,
                        help='latent_dim (default: 512)')
    parser.add_argument('--test_epoch', default='40', type=str)
    parser.add_argument('--window', default=0.1, type=float)
    parser.add_argument('--zeroshot', default=20, type=int)
    parser.add_argument('--fc', action='store_false')
    args = parser.parse_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    # breakpoint()
    kwargs = {'num_workers': 20, 'pin_memory': True} if use_cuda else {}
    nchannels, nclasses = 3, 10
    if args.dataset == 'MNIST': nchannels = 1
    if args.dataset == 'CIFAR100': nclasses = 50+args.zeroshot

    # create an initial model


    # size = [256, 512, 1024, 2048, 4096, 8192, 16384]
    window = [0.0]
    # num_zeroshot = [0]
    num_zeroshot = [0,10, 20, 30, 40, 50]

    size = [64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]
    # window = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
    # num_zeroshot = [0,10]
    for zeroshot in num_zeroshot:

        train_dataset = load_data('train', args.dataset, args.datadir, nchannels,zeroshot_number=zeroshot)
        val_dataset = load_data('val', args.dataset, args.datadir, nchannels,zeroshot_number=zeroshot)

        train_loader = DataLoader(train_dataset, batch_size=args.batchsize, shuffle=True, **kwargs)
        val_loader = DataLoader(val_dataset, batch_size=args.batchsize, shuffle=False, **kwargs)
        for num_unit in size:

            args.zeroshot = zeroshot
            args.nunits = num_unit

            nclasses = 50 + zeroshot
            model = nn.Sequential(nn.Linear(32 * 32 * nchannels, num_unit), nn.ReLU(),
                                      nn.Linear(num_unit, args.latent_dim), nn.ReLU())

            best_model_file_feature = os.path.join('Checkpoints_benchmark', 'feature',
                                                   'checkpoint.pth')
            checkpoint = torch.load(best_model_file_feature)
            model.load_state_dict(checkpoint)
            model = model.to(device)
            model_classifier = Classifier(n_feature=args.latent_dim, n_hidden=512, n_output=nclasses).to(device)
            criterion = nn.CrossEntropyLoss().to(device)
            early_stopping = EarlyStopping(patience=10, verbose=False)

            early_stopping.path = os.path.join('Checkpoints_benchmark', str(num_unit), str(zeroshot))


            # training the model
            for epoch in range(0, args.epochs):

                tr_loss, error = train(args, model, model_classifier, device, train_loader)

                val_err, val_loss, cls_err = validate(args, model, model_classifier, device, val_loader, criterion)

                print(f'Epoch: {epoch + 1}/{args.epochs}\t Training loss: {val_loss:.5f}\t val error rate: {val_err:.3f}\t train error rate: {error:.3f}\t')

                early_stopping(val_err, model_classifier, cls_err, args)
                if error < args.stopcond:
                    print('nunit=', num_unit, 'zeroshot=', zeroshot, 'best_error',
                          early_stopping.val_loss_min)
                    break

                if early_stopping.early_stop:
                    print('nunit=', num_unit, 'zeroshot=', zeroshot, 'best_error',  early_stopping.val_loss_min)
                    break




if __name__ == '__main__':
    main()
