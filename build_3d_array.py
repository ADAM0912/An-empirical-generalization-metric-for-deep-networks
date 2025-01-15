import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, ConcatDataset, Subset
import torch.nn.utils.prune as prune
import copy
import argparse
from torchvision import transforms, datasets
import pandas as pd
import os
from itertools import combinations
import matplotlib.pyplot as plt


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

def get_all_preds_labels(model, classifier, loader, device):
    all_preds = torch.tensor([]).to(device)
    all_labels = torch.tensor([]).to(device)
    with torch.no_grad():
        for i, (data, target) in enumerate(loader):
            data, target = data.to(device).view(data.size(0), -1), target.to(device)
            outputs = model(data)
            outputs = classifier(outputs)
            all_preds = torch.cat((all_preds, outputs), dim=0)
            all_labels = torch.cat((all_labels, target), dim=0)
    return all_preds, all_labels

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
            dataset1 = get_dataset(root=datadir, train=True, download=True, transform=tr_transform)
            dataset2 = get_dataset(root=datadir, train=False, download=True, transform=val_transform)
            # dataset = ConcatDataset([dataset1, dataset2])
            dataset = dataset1
            # breakpoint()
        if split == 'val':
            # dataset1 = get_dataset(root=datadir, train=True, download=True, transform=tr_transform)
            test_dataset = get_dataset(root=datadir, train=False, download=True, transform=val_transform)
            # dataset = ConcatDataset([dataset1, dataset2])
            # dataset = test_dataset
            train_indices = [i for i, (_, label) in enumerate(test_dataset) if label < 50+zeroshot_number]
            dataset = Subset(test_dataset, train_indices)
        else:
            test_dataset = get_dataset(root=datadir, train=True, download=True, transform=val_transform)
            # dataset = ConcatDataset([dataset1, dataset2])
            # dataset = test_dataset
            train_indices = [i for i, (_, label) in enumerate(test_dataset) if label < 50]
            dataset = Subset(test_dataset, train_indices)


    return dataset

def compute_kappa(confusion_matrix):
    """
    Compute Kappa for a given confusion matrix.

    Args:
        confusion_matrix (np.ndarray): Square confusion matrix.

    Returns:
        k (float): Kappa statistic.
    """
    # Total observations
    n = np.sum(confusion_matrix)

    # Compute observed agreement (P_0)
    P_0 = np.trace(confusion_matrix) / n

    # Compute row and column margins
    row_margins = np.sum(confusion_matrix, axis=1)
    col_margins = np.sum(confusion_matrix, axis=0)

    # Compute expected agreement (P_e)
    P_e = np.sum((row_margins * col_margins) / (n ** 2))
    # P_e = 0
    # for i in range(len(row_margins)):
    #     for j in range(len(col_margins)):
    #         P_e += row_margins[i] * col_margins[j]
    # Compute Kappa
    kappa = (P_0 - P_e) / (1 - P_e)

    return 1 - kappa


def compute_fifth_kappa(confusion_matrix):
    """
    Compute pairwise kappa for all possible class pairs from a confusion matrix.

    Args:
        confusion_matrix (np.ndarray): Confusion matrix (square, m x m).

    Returns:
        pairwise_kappas (dict): Dictionary of pairwise kappa values with class pairs as keys.
    """
    num_classes = confusion_matrix.shape[0]
    total_samples = np.sum(confusion_matrix)
    pairwise_kappas = []

    # Iterate over all unique class pairs
    for i in range(num_classes):
        # Extract pairwise confusion matrix
        n_row = np.sum(confusion_matrix[i])
        n_ii = confusion_matrix[i, i]
        n_ij = n_row - n_ii
        n_ji = np.sum(confusion_matrix[:,i]) - n_ii
        n_jj = total_samples - n_ii - n_ij - n_ji

        # Marginal totals
        n_i_row = n_ii + n_ij
        n_j_row = n_ji + n_jj
        n_i_col = n_ii + n_ji
        n_j_col = n_ij + n_jj

        # Compute probabilities
        p_ii = n_ii / total_samples
        p_ij = n_ij / total_samples
        p_ji = n_ji / total_samples
        p_jj = n_jj / total_samples
        p_i = (n_i_row / total_samples) * (n_i_col / total_samples)
        p_j = (n_j_row / total_samples) * (n_j_col / total_samples)
        # p__i = p_ii + p_ji
        # p__j = p_ij + p_jj
        # p_i_ = p_ii + p_ij
        # p_j_ = p_ji + p_jj

        p__i = (n_ii + n_ji)/total_samples
        p__j = (n_ij + n_jj)/total_samples
        p_i_ = (n_ii + n_ij)/total_samples
        p_j_ = (n_ji + n_jj)/total_samples


        P_0 = p_ii + p_jj
        # P_e = (n_i_row*n_i_col + n_i_row*n_j_col + n_j_row*n_i_col + n_j_row*n_j_col)/(total_samples**2)
        # P_e = p_i_*p__i + p_i_*p__j + p_j_*p__i + p_j_*p__j
        P_e = p_i_ * p__i +  p_j_ * p__j

        # Pairwise Kappa
        if 1 - P_e != 0:
            kappa = (P_0 - P_e) / (1 - P_e)
        else:
            kappa = 0.0  # Avoid division by zero

        # Store result
        pairwise_kappas.append(1 - kappa)

    return pairwise_kappas


def plot_error_kappa(error_rates,kappa_values):
    plt.figure(figsize=(10, 6))
    plt.scatter(error_rates, kappa_values, color='b', marker='o', edgecolor='k', alpha=0.7)

    # Add labels and title
    plt.xlabel('Error Rate', fontsize=12)
    plt.ylabel('Kappa', fontsize=12)
    plt.title('Scatter Plot of Kappa vs. Error Rate', fontsize=14)

    # Add grid for better readability
    plt.grid(True)

    # Show the plot
    plt.show()

def main():

    # settings
    parser = argparse.ArgumentParser(description='Training a fully connected NN with one hidden layer')
    parser.add_argument('--no-cuda', default=False, action='store_true',
                        help='disables CUDA training')
    parser.add_argument('--datadir', default='datasets', type=str,
                        help='path to the directory that contains the datasets (default: datasets)')
    parser.add_argument('--dataset', default='CIFAR100', type=str,
                        help='name of the dataset (options: MNIST | CIFAR10 | CIFAR100 | SVHN, default: CIFAR10)')
    parser.add_argument('--nunits', default=256, type=int,
                        help='number of hidden units (default: 1024)')
    parser.add_argument('--epochs', default=1000, type=int,
                        help='number of epochs to train (default: 1000)')
    parser.add_argument('--stopcond', default=0.01, type=float,
                        help='stopping condtion based on the cross-entropy loss (default: 0.01)')
    parser.add_argument('--batchsize', default=64, type=int,
                        help='input batch size (default: 64)')
    parser.add_argument('--learningrate', default=0.001, type=float,
                        help='learning rate (default: 0.001)')
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
    parser.add_argument('--checkpoints_clsm', default='Checkpoints/clsm', type=str)
    parser.add_argument('--test_epoch', default='40', type=str)
    parser.add_argument('--prune', default=0.1, type=float)
    parser.add_argument('--alpha', default=0.8, type=float)
    parser.add_argument('--zeroshot', default=20, type=int)
    args = parser.parse_args()


    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    # breakpoint()
    kwargs = {'num_workers': 20, 'pin_memory': True} if use_cuda else {}
    nchannels= 3
    if args.dataset == 'CIFAR100': nclasses = 50+args.zeroshot


    # ssim_list = [1.0]
    ssim_list = [0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.0]
    # zeroshot_list = [0, 10, 20, 30, 40, 50]
    zeroshot_list = [0,10, 20, 30, 40, 50]
    available_models = ['efficientnet-b0', 'efficientnet-b1', 'efficientnet-b2', 'efficientnet-b3',
                               'efficientnet-b4', 'efficientnet-b5', 'efficientnet-b6', 'efficientnet-b7']
    # available_models = ['RN50', 'RN101', 'RN50x4', 'RN50x16', 'RN50x64']

    # randomness_list = [0.1]
    # zeroshot_list = [0]
    # amount_list = [256]

    num_points = len(ssim_list)*len(zeroshot_list)*len(available_models)
    # each point has format like this:(randomness,zeroshot,amount,AUC,KL)
    output = []
    i = 0
    # breakpoint()
    for zeroshot in zeroshot_list:
        for model_index, model_type in enumerate(available_models):

          for ssim in ssim_list:
            # breakpoint()
            # output[i][0] = ssim
            # output[i][1] = zeroshot
            # output[i][2] = model_type

            file_location_kappa = os.path.join('experiment result eff',
                                         str(model_type) + str('_') + str(ssim) + str(
                                             '_zs') + str(zeroshot) + str('_') + 'kappa.xlsx')

            file_location_acc = os.path.join('experiment result eff',
                                         str(model_type) + str('_') + str(ssim) + str(
                                             '_zs') + str(zeroshot) + str('_') + 'accuracy.xlsx')
            data_model_2_kappa = pd.read_excel(file_location_kappa, header=None)

            data_model_2_accuracy = pd.read_excel(file_location_acc, header=None)

            kappa_matrix = data_model_2_kappa.iloc[1:, 1:].values
            kappa_all = compute_kappa(kappa_matrix)
            fifth_kappa = compute_fifth_kappa(kappa_matrix)
            mean_kappa = np.mean(fifth_kappa)
            std_kappa = np.std(fifth_kappa)
            percentile_kappa = np.percentile(fifth_kappa, 10)
            print('kappa', kappa_all)
            # breakpoint()
            # output[i][4] = mean_distance

            error_array = 1-data_model_2_accuracy[1][1:]
            mean_error = np.mean(error_array)
            std_error = error_array.std()
            percentile_error = error_array.quantile(0.10)
            output.append([ssim, zeroshot, model_type,mean_error,mean_kappa, std_error, std_kappa,percentile_error, percentile_kappa ])
            # plot_error_kappa(list(error_array), fifth_kappa)



            i = i+1

#

    df = pd.DataFrame(output, columns=['SSIM', 'ZeroShot', 'ModelType','Error Rate','Kappa','Std Error','Std Kappa', 'percentile_error', 'percentile_kappa'])

    # Save the DataFrame to an Excel file
    df.to_excel('output_eff_cifar.xlsx', index=False)

if __name__ == '__main__':
    main()