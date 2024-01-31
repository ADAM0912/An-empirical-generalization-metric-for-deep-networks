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
import shutil
from sklearn.metrics.pairwise import cosine_similarity
from utils import EarlyStopping, largest_unstructured
import h5py
from matplotlib import cm
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
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


    randomness_list = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    zeroshot_list = [0, 10, 20, 30, 40, 50]
    amount_list = [64, 128, 256, 512, 1024, 2048, 4096, 8192]

    # randomness_list = [0.2]
    # zeroshot_list = [0]
    # amount_list = [256]

    num_points = len(randomness_list)*len(zeroshot_list)*len(amount_list)
    # each point has format like this:(randomness,zeroshot,amount,AUC,KL)
    output = np.zeros((num_points,5))
    i = 0
    # breakpoint()
    for zeroshot in zeroshot_list:
        for amount in amount_list:

          for randomness in randomness_list:
            # breakpoint()
            output[i][0] = randomness
            output[i][1] = zeroshot
            output[i][2] = amount

            data_model_1_kappa = pd.read_excel(os.path.join('experiment result',
                                         str(16384) + str('_') + str(0.0) + str('_') + str(args.test_epoch) + str('_zs') + str(
                                             0) + str('_') + 'kappa_lp.xlsx'), header=None)

            data_model_1_accuracy = pd.read_excel(os.path.join('experiment result',
                                         str(16384) + str('_') + str(0.0) + str('_') + str(args.test_epoch) + str('_zs') + str(
                                             0) + str('_') + 'accuracy.xlsx'), header=None)


            data_model_2_kappa = pd.read_excel(os.path.join('experiment result',
                                         str(amount) + str('_') + str(randomness) + str('_') + str(args.test_epoch) + str('_zs') + str(
                                             zeroshot) + str('_') + 'kappa_lp.xlsx'), header=None)

            data_model_2_accuracy = pd.read_excel(os.path.join('experiment result',
                                         str(amount) + str('_') + str(randomness) + str('_') + str(args.test_epoch) + str('_zs') + str(
                                             zeroshot) + str('_') + 'accuracy.xlsx'), header=None)

            data_model_1_kappa.columns = ['index','Kappa Value']
            data_model_2_kappa.columns = ['index','Kappa Value']

            data_model_1_accuracy.columns = ['index','accuracy']
            data_model_2_accuracy.columns = ['index','accuracy']


            # Define the bins for error rate and kappa value
            # Adjust these bins to match the range and distribution of your data
            # accuracy_bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
            accuracy_bins = [0, 0.2, 0.4, 0.6, 0.8, 1]
            # kappa_bins = [0, 0.2, 0.4, 0.6, 0.8, 1]
            # kappa_bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
            kappa_bins = [0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]

            num_classes = 50 + int(zeroshot)
            # joint_model_1, _, _ = np.histogram2d(data_model_1_error['Error Rate'][1:num_classes+1], data_model_1_kappa['Kappa Value'][1:num_classes+1],
            #                                      bins=[error_bins, kappa_bins])
            hist_model_1, _ = np.histogram(data_model_1_kappa['Kappa Value'][1: 50 + 1], bins=kappa_bins)
            hist_model_2, _ = np.histogram(data_model_2_kappa['Kappa Value'][1:num_classes + 1], bins=kappa_bins)

            # Normalize the histograms to get joint probability distributions
            joint_prob_model_1 = hist_model_1 / np.sum(hist_model_1)
            joint_prob_model_2 = hist_model_2 / np.sum(hist_model_2)

            # Ensure there are no zero probabilities to avoid division by zero in KL divergence
            epsilon = 1e-10
            joint_prob_model_1 = np.maximum(joint_prob_model_1, epsilon)
            joint_prob_model_2 = np.maximum(joint_prob_model_2, epsilon)

            # Calculate the KL divergence for the joint distribution
            mean_distribution = 0.5 * (joint_prob_model_1 + joint_prob_model_2)

            # Calculate the KL divergence for the joint distribution
            kl_divergence_1 = np.sum(joint_prob_model_1 * np.log(joint_prob_model_1 / mean_distribution))
            kl_divergence_2 = np.sum(joint_prob_model_2 * np.log(joint_prob_model_2 / mean_distribution))

            js_divergence = 0.5 * (kl_divergence_1 + kl_divergence_2)

            output[i][4] = js_divergence

            hist_model_1, _ = np.histogram(data_model_1_accuracy['accuracy'][1: 50 + 1], bins=accuracy_bins)
            hist_model_2, _ = np.histogram(data_model_2_accuracy['accuracy'][1:num_classes + 1], bins=accuracy_bins)

            # Normalize the histograms to get joint probability distributions
            joint_prob_model_1 = hist_model_1 / np.sum(hist_model_1)
            joint_prob_model_2 = hist_model_2 / np.sum(hist_model_2)

            # Ensure there are no zero probabilities to avoid division by zero in KL divergence
            epsilon = 1e-10
            joint_prob_model_1 = np.maximum(joint_prob_model_1, epsilon)
            joint_prob_model_2 = np.maximum(joint_prob_model_2, epsilon)

            # Calculate the KL divergence for the joint distribution
            mean_distribution = 0.5 * (joint_prob_model_1 + joint_prob_model_2)

            # Calculate the KL divergence for the joint distribution
            kl_divergence_1 = np.sum(joint_prob_model_1 * np.log(joint_prob_model_1 / mean_distribution))
            kl_divergence_2 = np.sum(joint_prob_model_2 * np.log(joint_prob_model_2 / mean_distribution))

            js_divergence = 0.5 * (kl_divergence_1 + kl_divergence_2)

            output[i][3] = js_divergence
            i = i+1

#

    df1 = pd.DataFrame(output)

    df1.to_excel("output_f1f2.xlsx")


if __name__ == '__main__':
    main()