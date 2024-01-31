import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, ConcatDataset
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
from utils import millions_formatter
import matplotlib.ticker as ticker
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
    parser.add_argument('--prune', default=0.8, type=float)
    parser.add_argument('--alpha', default=0.8, type=float)
    parser.add_argument('--zeroshot', default=0, type=int)
    args = parser.parse_args()

    window = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    zeroshot = [0, 10, 20, 30, 40, 50]
    nunits = [64,128, 256, 512, 1024, 2048,4096, 8192,16384]
    parameter = [1e6,2e6,4e6,8e6,16e6,32e6, 64e6]
    n_window = len(window)
    KL_all = np.random.rand(len(nunits))
    # colors = ['blue', 'green', 'orange', 'cyan', 'magenta', 'red']
    colors = cm.get_cmap('plasma', len(zeroshot))
    # breakpoint()
    for j in range (len(zeroshot)):
        for i in range (len(nunits)):


            data_model_2_error = pd.read_excel(os.path.join('experiment result',
                                         str(nunits[i]) + str('_') + str(args.prune) + str('_') + str(args.test_epoch) + str('_zs') + str(
                                             zeroshot[j]) + str('_') + 'accuracy.xlsx'),header=None)
            #
            accuracy_array = data_model_2_error[1][1:]
            mean_accuracy = np.mean(accuracy_array)
            print('accuracy',mean_accuracy)
            KL_all[i] = mean_accuracy
            # breakpoint()

        x_values =  range(len(nunits))
        labels = ['0.25M','0.5M','1M', '2M', '4M', '8M', '16M', '32M', '64M']

        # Plotting the KL array against the x-axis values
        plt.plot(x_values, KL_all, color=colors(j),label=f'zeroshot percent {zeroshot[j]/(50+zeroshot[j]):.3f}')
        # plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(millions_formatter))
        plt.xticks(x_values, labels)
        # plt.subplots_adjust(left=0.1, right=0.2, top=0.2, bottom=0.1)
    plt.xlabel('number of parameters ',fontsize=16)
    plt.ylabel('Accuracy',fontsize=16)
    plt.title('Accuracy with number of parameters changed',fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(loc="upper right",fontsize=12)
    # plt.grid(True)
    plt.show()


if __name__ == '__main__':
    main()