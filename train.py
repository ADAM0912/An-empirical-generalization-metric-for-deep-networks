import torch
from torch import nn, optim
from torch.utils.data import DataLoader, ConcatDataset, Subset
import argparse
from torchvision import transforms, datasets
import os

import shutil
from efficientnet_pytorch import EfficientNet
from torch.optim.lr_scheduler import LambdaLR


class Classifier(nn.Module):
    def __init__(self, n_feature=512, n_hidden=1024, n_output=250):
        super(Classifier, self).__init__()
        self.out = torch.nn.Linear(n_feature, n_output)

    def forward(self, x_layer):
        feature = self.out(x_layer)
        return feature

def save_checkpoint_epoch(state, directory):
    if not os.path.isdir(directory):
        os.makedirs(directory)
    checkpoint_file = os.path.join(directory, 'checkpoint.pth')
    best_model_file = os.path.join(directory, 'model_best.pth')
    torch.save(state, checkpoint_file)
    shutil.copyfile(checkpoint_file, best_model_file)

def train(args, model, model_classifier, device, train_loader, mean_dict, optimizer, epoch):
    sum_loss, sum_correct = 0, 0
    model.train()
    model_classifier.train()
    for i, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        feature_output = model(data)
        output = model_classifier(feature_output)
        criterion = nn.CrossEntropyLoss()
        loss = criterion(output, target)
        pred = output.max(1)[1]
        sum_correct += pred.eq(target).sum().item()
        error = 1 - (sum_correct / len(train_loader.dataset))
        sum_loss += len(data) * loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return sum_loss / len(train_loader.dataset), error

def load_data(split, dataset_name, datadir, preprocess):

    get_dataset = getattr(datasets, dataset_name)
    if dataset_name == 'CIFAR100':
        if split == 'train':
            train_dataset = get_dataset(root=datadir, train=True, download=True, transform=preprocess)
            train_indices = [i for i, (_, label) in enumerate(train_dataset) if label < 50]
            dataset = Subset(train_dataset, train_indices)
            # dataset = train_dataset

        else:
            test_dataset = get_dataset(root=datadir, train=True, download=True, transform=preprocess)
            test_indices = [i for i, (_, label) in enumerate(test_dataset) if label >= 50]
            dataset = Subset(test_dataset, test_indices)

    return dataset

def lr_decay(epoch):
    return 0.97 ** (epoch / 2.4)

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
    parser.add_argument('--learningrate', default=0.256, type=float,
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
    parser.add_argument('--checkpoints_feature', default='Checkpoints eff/feature', type=str)
    parser.add_argument('--n_train', default=1000, type=int)
    args = parser.parse_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    # breakpoint()
    kwargs = {'num_workers': 20, 'pin_memory': True} if use_cuda else {}
    if args.dataset == 'CIFAR100': nclasses = 50


    available_models = ['efficientnet-b0', 'efficientnet-b1', 'efficientnet-b2', 'efficientnet-b3', 'efficientnet-b4', 'efficientnet-b5', 'efficientnet-b6', 'efficientnet-b7']
    batchsize = [64, 64, 64, 32, 32,16,16,16]

    for model_index in range(8):
        model_type = available_models[model_index]
        model = EfficientNet.from_pretrained(model_type)
        model_classifier = Classifier(n_feature=1000, n_hidden=512, n_output=nclasses)

        model = model.to(device)
        preprocess = transforms.Compose([transforms.Resize(224), transforms.ToTensor(),
                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),])
        model_classifier = model_classifier.to(device)
        optimizer = optim.RMSprop(list(model.parameters()) + list(model_classifier.parameters()), args.learningrate,
                              weight_decay=1e-5)

        # loading data
        train_dataset = load_data('train', args.dataset, args.datadir, preprocess)

        train_loader = DataLoader(train_dataset, batch_size=batchsize[model_index], shuffle=True, **kwargs)
        mean_dict = {}
        scheduler = LambdaLR(optimizer, lr_lambda=lr_decay)

        # training the model

        for epoch in range(0, args.epochs):
            # train for one epoch
            tr_loss, error = train(args, model, model_classifier,  device, train_loader, mean_dict, optimizer, epoch)
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
            print(f'Epoch: {epoch + 1}/{args.epochs}\t Training loss: {tr_loss:.5f}\t error rate: {error:.3f}\t learning rate: {current_lr:.3f}\t')
            if error < args.stopcond: break

        file_location = os.path.join(args.checkpoints_feature,
                                     str(model_type))

        feature_checkpoint = model.state_dict()

        save_checkpoint_epoch(feature_checkpoint, directory=file_location)


if __name__ == '__main__':
    main()