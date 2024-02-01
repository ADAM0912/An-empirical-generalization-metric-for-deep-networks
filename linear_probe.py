
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, ConcatDataset, Subset
import copy
import argparse
from torchvision import transforms, datasets
import os
import shutil
from utils import EarlyStopping
import torch.nn.utils.prune as prune
import h5py
import torch.nn.functional as F

class Classifier(nn.Module):
    def __init__(self, n_feature=512, n_hidden=1024, n_output=250):
        super(Classifier, self).__init__()
        # self.n_hidden = torch.nn.Linear(n_feature, n_hidden)
        # self.out = torch.nn.Linear(n_hidden, n_output)
        self.out = torch.nn.Linear(n_feature, n_output)

    def forward(self, x_layer):
        # feature = self.n_hidden(x_layer)
        feature = self.out(x_layer)
        # x_layer = torch.nn.functional.softmax(feature)
        # output = torch.sigmoid(feature)

        return feature

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=512, conv_channels=(64, 128, 256), fc_units=512):
        super(SimpleCNN, self).__init__()
        # Unpack the number of channels for each convolutional layer
        ch1, ch2, ch3 = conv_channels

        # Define the convolutional layers
        self.conv1 = nn.Conv2d(3, ch1, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(ch1, ch2, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(ch2, ch3, kernel_size=3, stride=1, padding=1)

        # Calculate the size of the flattened features after the last convolutional layer
        self.flat_features = ch3 * 4 * 4

        # Define the fully connected layers
        self.fc1 = nn.Linear(self.flat_features, fc_units)
        self.fc2 = nn.Linear(fc_units, num_classes)

        # Define batchnorm and dropout layers
        self.batchnorm1 = nn.BatchNorm2d(ch1)
        self.batchnorm2 = nn.BatchNorm2d(ch2)
        self.batchnorm3 = nn.BatchNorm2d(ch3)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # Apply the convolutional layers with batchnorm and maxpool
        x = F.max_pool2d(F.relu(self.batchnorm1(self.conv1(x))), kernel_size=2, stride=2)
        x = F.max_pool2d(F.relu(self.batchnorm2(self.conv2(x))), kernel_size=2, stride=2)
        x = F.max_pool2d(F.relu(self.batchnorm3(self.conv3(x))), kernel_size=2, stride=2)

        # Flatten the output of the last convolutional layer
        x = x.view(x.size(0), -1)

        # Apply the fully connected layers with dropout
        x = F.relu(self.dropout(self.fc1(x)))
        x = self.fc2(x)
        return x

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

    optimizerCls = optim.SGD(model_classifier.parameters(), args.learningrate, momentum=args.momentum)
    # optimizerCls = optim.SGD(model_classifier.parameters(), args.learningrate, momentum=args.momentum)
    # switch to train mode
    # model.train()
    model_classifier.train()
    count = 1

    for i, (data, target) in enumerate(train_loader):
        # breakpoint()
        # assert not torch.isnan(data).any()
        # assert not torch.isinf(data).any()
        # assert not torch.isnan(target).any()
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

        # compute the gradient and do an SGD step
        optimizerCls.zero_grad()
        loss.backward()
        optimizerCls.step()

        # if count == 1:
        #     out_cls = output.cpu().detach().numpy()
        #     count += 1
        # else:
        #     out_cls = np.concatenate((out_cls, output.cpu().data.numpy()), axis=0)
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
    margin = torch.Tensor([]).to(device)

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

def compute_mean(weights):
    return torch.mean(weights, dim=0)


def compute_bounds(weights, percentage=0.1):
    num_samples = weights.shape[0]
    sorted_weights, _ = torch.sort(weights, dim=0)

    lower_index = int(num_samples * percentage / 2)
    upper_index = int(num_samples * (1 - percentage / 2))

    lower_bound = sorted_weights[lower_index]
    upper_bound = sorted_weights[upper_index]

    return lower_bound, upper_bound


def prune_weights(weights, percentage=0.1):
    mean = compute_mean(weights)
    # breakpoint()
    weights = weights-mean
    lower_bound, upper_bound = compute_bounds(weights, percentage)

    pruned_weights = torch.where(weights < lower_bound, lower_bound+mean, weights+mean)
    pruned_weights = torch.where(pruned_weights > upper_bound, upper_bound+mean, pruned_weights+mean)

    return pruned_weights




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
    parser.add_argument('--checkpoints_feature', default='Checkpoints/feature', type=str)
    parser.add_argument('--checkpoints_mean', default='Checkpoints/mean', type=str)
    parser.add_argument('--checkpoints_clsf', default='Checkpoints/clsf', type=str)
    parser.add_argument('--checkpoints_clsm', default='Checkpoints/clsm', type=str)
    parser.add_argument('--test_epoch', default='40', type=str)
    parser.add_argument('--prune', default=0.1, type=float)
    parser.add_argument('--no_prune',action= 'store_true')
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
    # model = SimpleCNN(conv_channels=(128, 256, 512), fc_units=1024).to(device)

    # breakpoint()

    # size = [256, 512, 1024, 2048, 4096, 8192, 16384]
    window = [0.0]
    # num_zeroshot = [0]
    num_zeroshot = [0,10, 20, 30, 40, 50]

    size = [64,128]
    # window = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
    # num_zeroshot = [0,10]
    for prune in window:
        for zeroshot in num_zeroshot:
            train_dataset = load_data('train', args.dataset, args.datadir, nchannels,zeroshot_number=zeroshot)
            val_dataset = load_data('val', args.dataset, args.datadir, nchannels,zeroshot_number=zeroshot)

            train_loader = DataLoader(train_dataset, batch_size=args.batchsize, shuffle=True, **kwargs)
            val_loader = DataLoader(val_dataset, batch_size=args.batchsize, shuffle=False, **kwargs)
            for num_unit in size:


    # create an initial model
                nclasses = 50 + zeroshot
                args.zeroshot = zeroshot
                args.prune = prune
                args.nunits = num_unit
                model = nn.Sequential(nn.Linear(32 * 32 * nchannels, num_unit), nn.ReLU(),
                                          nn.Linear(num_unit, args.latent_dim), nn.ReLU())

                # breakpoint()
                best_model_file_feature = os.path.join('backup_Checkpoints', str(num_unit), 'Checkpoints', 'feature',
                                                       str(args.test_epoch), 'model_best.pth')
                checkpoint = torch.load(best_model_file_feature)
                model.load_state_dict(checkpoint)
                # breakpoint()
                if prune != 0:
                    prune_weights_location = os.path.join('prune_backup_zeroshot', str(num_unit) + '_' + str(prune) + '.h5')
                    with h5py.File(prune_weights_location, 'r') as hf:
                        # breakpoint()
                        weights_data = hf['first_layer_weights'][:]
                        weights_data = torch.from_numpy(weights_data)

                        model[0].weight.data = weights_data[int(args.test_epoch)]
                        weights_data = hf['second_layer_weights'][:]
                        weights_data = torch.from_numpy(weights_data)
                        model[2].weight.data = weights_data[int(args.test_epoch)]


                model = model.to(device)


                # breakpoint()
                model_classifier = Classifier(n_feature=args.latent_dim, n_hidden=512, n_output=nclasses).to(device)
                # best_model_file_feature = os.path.join('backup_Checkpoints', str(num_unit), 'Checkpoints', 'clsf',
                #                                        str(args.test_epoch), 'model_best.pth')
                # checkpoint = torch.load(best_model_file_feature)
                # model_classifier.load_state_dict(checkpoint)

                # define loss function (criterion) and optimizer
                criterion = nn.CrossEntropyLoss().to(device)
                optimizer = optim.SGD(model.parameters(), args.learningrate, momentum=args.momentum)


                # loading data
                # train_dataset = load_data('train', args.dataset, args.datadir, nchannels,zeroshot_number=zeroshot)
                # val_dataset = load_data('val', args.dataset, args.datadir, nchannels,zeroshot_number=zeroshot)
                #
                # train_loader = DataLoader(train_dataset, batch_size=args.batchsize, shuffle=True, **kwargs)
                # val_loader = DataLoader(val_dataset, batch_size=args.batchsize, shuffle=False, **kwargs)
                data={}
                mean_dict = {}
                early_stopping = EarlyStopping(patience=10, verbose=False)

                early_stopping.path = os.path.join('cls_checkpoints', str(num_unit), str(prune), str(zeroshot))


                # training the model
                for epoch in range(0, args.epochs):
                    # train for one epoch
                    # breakpoint()
                    tr_loss, error = train(args, model, model_classifier, device, train_loader, mean_dict, optimizer, epoch)

                    val_err, val_loss, cls_err = validate(args, model, model_classifier, device, val_loader, criterion)

                    print(f'Epoch: {epoch + 1}/{args.epochs}\t Training loss: {val_loss:.5f}\t val error rate: {val_err:.3f}\t train error rate: {error:.3f}\t')

                    early_stopping(val_err, model_classifier, cls_err, args)
                    if error < args.stopcond:
                        print('nunit=', num_unit, 'window=', prune, 'zeroshot=', zeroshot, 'best_error',
                              early_stopping.val_loss_min)
                        break

                    if early_stopping.early_stop:
                        # print("Early stopping")
                        # print(f'best Validation error ({ early_stopping.val_loss_min:.6f} ')
                        print('nunit=', num_unit, 'window=', prune, 'zeroshot=', zeroshot, 'best_error',  early_stopping.val_loss_min)
                        break




if __name__ == '__main__':
    main()
