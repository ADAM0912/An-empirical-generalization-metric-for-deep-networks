import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, ConcatDataset
import argparse
from torchvision import transforms, datasets
import os
import shutil
import h5py



class Classifier(nn.Module):
    def __init__(self, n_feature=512, n_hidden=1024, n_output=250):
        super(Classifier, self).__init__()
        self.n_hidden = torch.nn.Linear(n_feature, n_hidden)
        self.out = torch.nn.Linear(n_hidden, n_output)

    def forward(self, x_layer):
        x_layer = torch.relu(self.n_hidden(x_layer))
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

def train(args, model, model_classifier, device, train_loader, mean_dict, optimizer, epoch):
    sum_loss, sum_correct = 0, 0
    optimizerfeature = optim.SGD(model.parameters(), args.learningrate, momentum=args.momentum)

    optimizerCls = optim.SGD(model.parameters(), args.learningrate, momentum=args.momentum)
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
        # criterion = SupConLoss(temperature=0.05)
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
    else:
        if split == 'train':
            dataset1 = get_dataset(root=datadir, train=True, download=True, transform=tr_transform)
            dataset2 = get_dataset(root=datadir, train=False, download=True, transform=val_transform)
            dataset = ConcatDataset([dataset1, dataset2])
            # breakpoint()
        else:
            dataset = get_dataset(root=datadir, train=False, download=True, transform=val_transform)

    return dataset


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

    pruned_weights = torch.where(weights < lower_bound, lower_bound, weights)
    pruned_weights = torch.where(pruned_weights > upper_bound, upper_bound, pruned_weights)
    pruned_weights = pruned_weights+mean

    return pruned_weights

def main():

    # settings
    parser = argparse.ArgumentParser(description='Training a fully connected NN with one hidden layer')
    parser.add_argument('--no-cuda', default=False, action='store_true',
                        help='disables CUDA training')
    parser.add_argument('--datadir', default='datasets', type=str,
                        help='path to the directory that contains the datasets (default: datasets)')
    parser.add_argument('--dataset', default='CIFAR10', type=str,
                        help='name of the dataset (options: MNIST | CIFAR10 | CIFAR100 | SVHN, default: CIFAR10)')
    parser.add_argument('--nunits', default=256, type=int,
                        help='number of hidden units (default: 1024)')
    parser.add_argument('--epochs', default=1000, type=int,
                        help='number of epochs to train (default: 1000)')
    parser.add_argument('--stopcond', default=0.01, type=float,
                        help='stopping condtion based on the cross-entropy loss (default: 0.01)')
    parser.add_argument('--batchsize', default=64, type=int,
                        help='input batch size (default: 64)')
    parser.add_argument('--learningrate', default=0.01, type=float,
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='momentum (default: 0.9)')
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
    parser.add_argument('--window', default=0.1, type=float)
    parser.add_argument('--zeroshot', default=20, type=int)
    parser.add_argument('--test_epoch', default='40', type=str)
    parser.add_argument('--n_sample', default=50, type=int)
    args = parser.parse_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    # breakpoint()
    kwargs = {'num_workers': 20, 'pin_memory': True} if use_cuda else {}
    nchannels, nclasses = 3, 10
    if args.dataset == 'MNIST': nchannels = 1
    if args.dataset == 'CIFAR100': nclasses = 100


    model = nn.Sequential(nn.Linear(32 * 32 * nchannels, args.nunits), nn.ReLU(), nn.Linear(args.nunits, args.latent_dim), nn.ReLU())
    model = model.to(device)


    model_classifier = Classifier(n_feature=512, n_hidden=512, n_output=nclasses).to(device)

    base_dir = os.path.join('backup_Checkpoints', str(args.nunits),'Checkpoints/feature')

    model_weights = []

    # Assuming all models have the same architecture and you're using the same model class for all

    # Loop through model directories
    prune_weights_location = os.path.join('sample_weight', str(args.nunits) + '_' + str(args.window) + '.h5')
    with h5py.File(prune_weights_location, 'w') as hf:
        dset1_shape = (args.n_sample, args.nunits, 3072)
        dset2_shape = (args.n_sample, 512, args.nunits)
        # dset_first = hf.create_dataset('first_layer_weights', dset1_shape, dtype=np.float32)
        # dset_second = hf.create_dataset('second_layer_weights', dset2_shape, dtype=np.float32)
        model_path = os.path.join(base_dir, str(args.test_epoch), 'model_best.pth')
        # Check if the file exists, then load
        if os.path.exists(model_path):
            # Load model weights
            model.load_state_dict(torch.load(model_path))

            # Extract weights
            dset_first_base = model[0].weight.detach().cpu().numpy()
            dset_second_base = model[2].weight.detach().cpu().numpy()
            # breakpoint()

            means = dset_first_base.reshape(1, args.nunits, 3072)
            std_dev = 0.05
            lower_bounds = means - std_dev * args.window*2
            upper_bounds = means + std_dev * args.window*2

            new_samples = np.random.uniform(lower_bounds, upper_bounds, (args.n_sample,args.nunits, 3072 )).astype(np.float32)

            # breakpoint()
            hf.create_dataset('first_layer_weights', data=new_samples)

            means = dset_second_base.reshape(1, 512, args.nunits)

            std_dev = 0.05
            lower_bounds = means - std_dev * args.window*2
            upper_bounds = means + std_dev * args.window*2
            new_samples = np.random.uniform(lower_bounds, upper_bounds, (args.n_sample,512, args.nunits)).astype(np.float32)

            # breakpoint()
            hf.create_dataset('second_layer_weights', data=new_samples)






if __name__ == '__main__':
    main()
