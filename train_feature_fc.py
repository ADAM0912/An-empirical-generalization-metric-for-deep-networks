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


class Classifier(nn.Module):
    def __init__(self, n_feature=512, n_hidden=1024, n_output=250):
        super(Classifier, self).__init__()
        # self.n_hidden = torch.nn.Linear(n_feature, n_hidden)
        # self.out = torch.nn.Linear(n_hidden, n_output)
        self.out = torch.nn.Linear(n_feature, n_output)

    def forward(self, x_layer):
        # feature = torch.relu(self.n_hidden(x_layer))
        feature = self.out(x_layer)
        # x_layer = torch.nn.functional.softmax(feature)
        # output = torch.sigmoid(feature)

        return feature

def calculate_mean(output, label, mean_dict,epoch):
    mean_output = torch.zeros((output.shape))
    if epoch == 0 :
        for i in range(len(label)):
            # breakpoint()
            index = int(label[i])
            if index not in mean_dict.keys():
                mean_dict[index]= output[i]
                mean_output[i] = mean_dict[index]
            else:
                mean_dict[index] = (mean_dict[index]+output[i])/2
                mean_output[i] = mean_dict[index]
    else:
        for i in range(len(label)):
            index = int(label[i])
            mean_output[i] = mean_dict[index]


    return mean_dict, mean_output

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

def train(args, model, model_classifier, device, train_loader, mean_dict, optimizer, epoch):
    sum_loss, sum_correct = 0, 0
    optimizerfeature = optim.SGD(list(model.parameters()), args.learningrate, momentum=args.momentum)
    # optimizerfeature = torch.optim.Adam(list(model.parameters()), lr=args.learningrate)

    optimizerCls = optim.SGD(model_classifier.parameters(), args.learningrate, momentum=args.momentum)
    # switch to train mode
    model.train()
    model_classifier.train()
    count = 1

    for i, (data, target) in enumerate(train_loader):
        # breakpoint()
        # assert not torch.isnan(data).any()
        # assert not torch.isinf(data).any()
        # assert not torch.isnan(target).any()
        data, target = data.to(device).view(data.size(0),-1), target.to(device)
        feature_output = model(data)
        output = model_classifier(feature_output)
        criterion = nn.CrossEntropyLoss()
        loss = criterion(output, target)
        # breakpoint()
        pred = output.max(1)[1]
        sum_correct += pred.eq(target).sum().item()
        error = 1 - (sum_correct / len(train_loader.dataset))
        sum_loss += len(data) * loss.item()

        # compute the gradient and do an SGD step
        optimizerfeature.zero_grad()
        optimizerCls.zero_grad()
        loss.backward()
        optimizerfeature.step()
        optimizerCls.step()

        # if count == 1:
        #     out_cls = output.cpu().detach().numpy()
        #     count += 1
        # else:
        #     out_cls = np.concatenate((out_cls, output.cpu().data.numpy()), axis=0)
    return sum_loss / len(train_loader.dataset), error

def load_data(split, dataset_name, datadir, nchannels):

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

    if dataset_name == 'CIFAR100':
        if split == 'train':
            train_dataset = get_dataset(root=datadir, train=True, download=True, transform=tr_transform)
            train_indices = [i for i, (_, label) in enumerate(train_dataset) if label < 50]
            dataset = Subset(train_dataset, train_indices)
            # dataset = train_dataset

        else:
            test_dataset = get_dataset(root=datadir, train=True, download=True, transform=val_transform)
            test_indices = [i for i, (_, label) in enumerate(test_dataset) if label >= 50]
            dataset = Subset(test_dataset, test_indices)

    if dataset_name == 'CIFAR10':
        if split == 'train':
            train_dataset = get_dataset(root=datadir, train=True, download=True, transform=tr_transform)
            train_indices = [i for i, (_, label) in enumerate(train_dataset) if label < 8]
            dataset = Subset(train_dataset, train_indices)

        else:
            test_dataset = get_dataset(root=datadir, train=True, download=True, transform=val_transform)
            test_indices = [i for i, (_, label) in enumerate(test_dataset) if label >= 8]
            dataset = Subset(test_dataset, test_indices)

    return dataset

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
    parser.add_argument('--learningrate', default=0.01, type=float,
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='momentum (default: 0.9)')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight decay')
    parser.add_argument('--hid_dim', default=512, type=float,
                        help='hid_dim (default: 512)')
    parser.add_argument('--out_dim', default=512, type=float,
                        help='out_dim (default: 512)')
    parser.add_argument('--latent_dim', default=512, type=float,
                        help='latent_dim (default: 512)')
    parser.add_argument('--checkpoints_feature', default='Checkpoints/feature', type=str)
    parser.add_argument('--checkpoints_mean', default='Checkpoints/mean', type=str)
    parser.add_argument('--checkpoints_clsf', default='Checkpoints/clsf', type=str)
    parser.add_argument('--checkpoints_clsm', default='Checkpoints/clsm', type=str)
    parser.add_argument('--n_train', default=1, type=int)
    args = parser.parse_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    # breakpoint()
    kwargs = {'num_workers': 20, 'pin_memory': True} if use_cuda else {}
    nchannels, nclasses = 3, 10
    if args.dataset == 'MNIST': nchannels = 1
    if args.dataset == 'CIFAR100': nclasses = 100

    # create an initial model
    model = nn.Sequential(nn.Linear(32 * 32 * nchannels, args.nunits), nn.ReLU(), nn.Linear(args.nunits, args.latent_dim), nn.ReLU())
    model_classifier = Classifier(n_feature=args.latent_dim, n_hidden=512, n_output=nclasses)
    # best_model_file_cls = os.path.join(args.checkpoints_feature, 'model_best.pth')
    # checkpoint = torch.load(best_model_file_cls)
    # model_feature.load_state_dict(checkpoint)
    # if torch.cuda.is_available():
    #     if torch.cuda.device_count() > 1:
    #         model = torch.nn.DataParallel(model)
    #         model_classifier = torch.nn.DataParallel(model_classifier)
    model = model.to(device)
    model_classifier = model_classifier.to(device)

    # breakpoint()
    # model, _ = clip.load("ViT-B/32", device=device)

    # model = model.float()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().to(device)
    # optimizer = optim.SGD(model_mean.parameters(), args.learningrate, momentum=args.momentum)
    optimizer = torch.optim.AdamW(list(model.parameters())+list(model_classifier.parameters()), lr= args.learningrate)


    # loading data
    train_dataset = load_data('train', args.dataset, args.datadir, nchannels)
    val_dataset = load_data('val', args.dataset, args.datadir, nchannels)

    train_loader = DataLoader(train_dataset, batch_size=args.batchsize, shuffle=True, **kwargs)
    val_loader = DataLoader(val_dataset, batch_size=args.batchsize, shuffle=False, **kwargs)
    data={}
    mean_dict = {}

    # training the model
    for i in range(args.n_train):
        model = nn.Sequential(nn.Linear(32 * 32 * nchannels, args.nunits), nn.ReLU(),
                              nn.Linear(args.nunits, args.latent_dim), nn.ReLU()).to(device)
        model_classifier = Classifier(n_feature=512, n_hidden=512, n_output=nclasses).to(device)



        for epoch in range(0, args.epochs):
            # train for one epoch
            tr_loss, error = train(args, model, model_classifier, device, train_loader, mean_dict, optimizer, epoch)

            # val_err, val_loss, val_margin = validate(args, model, device, val_loader, criterion)

            print(f'n_train: {i + 1}/{args.n_train}\t Epoch: {epoch + 1}/{args.epochs}\t Training loss: {tr_loss:.5f}\t error rate: {error:.3f}\t')

            # stop training if the cross-entropy loss is less than the stopping condition
            if error < args.stopcond: break
            # if epoch % 10 == 9:
            #     # calculate the training error and margin of the learned model
            #     tr_err, tr_loss, tr_margin = validate(args, model, device, train_loader, criterion)
            #     print(f'\nFinal: Training loss: {tr_loss:.3f}\t Training margin {tr_margin:.3f}\t ',
            #             f'Training error: {tr_err:.3f}\t Validation error: {val_err:.3f}\n')
            #     k_test_all = np.zeros((nclasses,1))
            #     for i in range(nclasses):
            #         # breakpoint()
            #         k_test_all[i] = measures.calculate_kappa(out_cls, i, thres=10)
            #         k_test_mean = np.mean(k_test_all)
            #     # measure = measures.calculate(model, init_model, device, train_loader, tr_margin)
            #     # for key, value in measure.items():
            #     #     print(f'{key:s}:\t {float(value):3.3}')
            #     data[epoch+1] = [k_test_mean, tr_loss,tr_err,val_err]
            # # breakpoint()
            # df = pd.DataFrame(data)
            # df.to_excel('output_kappa.xlsx',index_label= 'epoch')
        feature_checkpoint = model.state_dict()

        save_checkpoint_epoch(feature_checkpoint, directory=args.checkpoints_feature, epoch = i)

        clsf_checkpoint = model_classifier.state_dict()

        save_checkpoint_epoch(clsf_checkpoint, directory=args.checkpoints_clsf, epoch = i)

if __name__ == '__main__':
    main()
