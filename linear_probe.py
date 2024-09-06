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
from utils import EarlyStopping, largest_unstructured
import torch.nn.utils.prune as prune
import h5py
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet
from utils import CustomDataset

class Classifier(nn.Module):
    def __init__(self, n_feature=512, n_hidden=1024, n_output=250):
        super(Classifier, self).__init__()
        self.out = torch.nn.Linear(n_feature, n_output)

    def forward(self, x_layer):
        feature = self.out(x_layer)
        return feature


def save_checkpoint(state, directory):

    if not os.path.isdir(directory):
        os.makedirs(directory)
    checkpoint_file = os.path.join(directory, 'checkpoint.pth')
    best_model_file = os.path.join(directory, 'model_best.pth')
    torch.save(state, checkpoint_file)
    shutil.copyfile(checkpoint_file, best_model_file)

def save_checkpoint_epoch(state, directory,epoch):
    directory = os.path.join(directory, str(epoch))
    if not os.path.isdir(directory):
        os.makedirs(directory)
    checkpoint_file = os.path.join(directory, 'checkpoint.pth')
    best_model_file = os.path.join(directory, 'model_best.pth')
    torch.save(state, checkpoint_file)
    shutil.copyfile(checkpoint_file, best_model_file)

def train(args, model, model_classifier, device, train_loader, mean_dict, epoch):
    sum_loss, sum_correct = 0, 0
    optimizerCls = optim.AdamW(model_classifier.parameters(), args.learningrate)
    model.eval()
    model_classifier.train()

    for i, (data, target) in enumerate(train_loader):

        data, target = data.to(device), target.to(device)
        feature_output= model(data)

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

def load_data(split, dataset_name, datadir, preprocess,zeroshot_number):
    # for CLIP
    get_dataset = getattr(datasets, dataset_name)

    if dataset_name == 'CIFAR100':
        if split == 'train':
            train_dataset = get_dataset(root=datadir, train=True, download=True, transform=preprocess)
            train_indices = [i for i, (_, label) in enumerate(train_dataset) if label < 50+zeroshot_number]
            dataset = Subset(train_dataset, train_indices)
            # dataset = train_dataset

        else:
            test_dataset = get_dataset(root=datadir, train=False, download=True, transform=preprocess)
            test_indices = [i for i, (_, label) in enumerate(test_dataset) if label < 50+zeroshot_number]
            dataset = Subset(test_dataset, test_indices)


    return dataset

def validate(args, model, classifier, device, val_loader, criterion):
    num_classes = args.zeroshot+50
    class_correct = torch.zeros(num_classes, dtype=torch.float32).to(device)
    class_total = torch.zeros(num_classes, dtype=torch.float32).to(device)

    sum_loss, sum_correct = 0, 0

    # switch to evaluation mode
    model.eval()
    classifier.eval()
    with torch.no_grad():
        for i, (data, target) in enumerate(val_loader):

            data, target = data.to(device), target.to(device)
            feature_output= model(data)
            output = classifier(feature_output)
            # compute the classification error and loss
            pred = output.max(1)[1]
            sum_correct += pred.eq(target).sum().item()
            sum_loss += len(data) * criterion(output, target).item()

            # Update class-wise correct and total counts
            for t, p in zip(target.view(-1), pred.view(-1)):
                class_correct[t] += (t == p).item()
                class_total[t] += 1
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
    parser.add_argument('--epochs', default=10, type=int,
                        help='number of epochs to train (default: 1000)')
    parser.add_argument('--stopcond', default=0.005, type=float,
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
    parser.add_argument('--checkpoints_feature', default='Checkpoints eff/feature', type=str)
    parser.add_argument('--test_epoch', default='40', type=str)
    parser.add_argument('--prune', default=0.1, type=float)
    parser.add_argument('--no_prune',action= 'store_true')
    parser.add_argument('--zeroshot', default=20, type=int)
    parser.add_argument('--fc', action='store_false')
    args = parser.parse_args()
    args = parser.parse_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda:1" if use_cuda else "cpu")
    # breakpoint()
    kwargs = {'num_workers': 20, 'pin_memory': True} if use_cuda else {}
    if args.dataset == 'CIFAR100': nclasses = 50+args.zeroshot

    num_zeroshot = [0,10, 20, 30, 40, 50]
    available_models = ['efficientnet-b0', 'efficientnet-b1', 'efficientnet-b2', 'efficientnet-b3', 'efficientnet-b4', 'efficientnet-b5', 'efficientnet-b6', 'efficientnet-b7']
    batchsize = [32, 32, 32, 16, 16,8,8,4]

    for model_index, model_type in enumerate(available_models):
        for zeroshot in num_zeroshot:

            nclasses = 50 + zeroshot
            args.zeroshot = zeroshot
            args.prune = prune

            model = EfficientNet.from_pretrained(model_type)
            model_classifier = Classifier(n_feature=1000, n_hidden=512, n_output=nclasses)

            model = model.to(device)
            preprocess = transforms.Compose([transforms.Resize(224), transforms.ToTensor(),
                                             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), ])
            model_classifier = model_classifier.to(device)
            file_location = os.path.join(args.checkpoints_feature,
                                         str(model_type))
            best_model_file_feature = os.path.join(file_location, 'model_best.pth')
            checkpoint = torch.load(best_model_file_feature)
            model.load_state_dict(checkpoint)

            model = model.to(device)
            train_dataset = load_data('train', args.dataset, args.datadir, preprocess, zeroshot)
            val_dataset = load_data('val', args.dataset, args.datadir, preprocess, zeroshot)

            train_loader = DataLoader(train_dataset, batch_size=batchsize[model_index], shuffle=True, **kwargs)
            val_loader = DataLoader(val_dataset, batch_size=batchsize[model_index], shuffle=False, **kwargs)
            criterion = nn.CrossEntropyLoss().to(device)
            mean_dict = {}
            early_stopping = EarlyStopping(patience=5, verbose=False)

            early_stopping.path = os.path.join('cls_checkpoints_eff', str(model_type), str(zeroshot))

            # training the model
            for epoch in range(0, args.epochs):

                tr_loss, error = train(args, model, model_classifier, device, train_loader, mean_dict, epoch)

                val_err, val_loss, cls_err = validate(args, model, model_classifier, device, val_loader, criterion)

                print(
                    f'Epoch: {epoch + 1}/{args.epochs}\t Training loss: {val_loss:.5f}\t val error rate: {val_err:.3f}\t train error rate: {error:.3f}\t')

                early_stopping(val_err, model_classifier, cls_err, args)
                if error < args.stopcond:
                    print('model=', model_type, 'zeroshot=', zeroshot, 'best_error',
                          early_stopping.val_loss_min)
                    break

                if early_stopping.early_stop:
                    print('model=', model_type, 'zeroshot=', zeroshot, 'best_error', early_stopping.val_loss_min)
                    break




if __name__ == '__main__':
    main()
