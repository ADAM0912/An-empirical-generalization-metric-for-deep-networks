import numpy as np

import argparse
import pandas as pd
import os



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




    ssim_list = [0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.0]
    zeroshot_list = [0,10, 20, 30, 40, 50]
    available_models = ['efficientnet-b0', 'efficientnet-b1', 'efficientnet-b2', 'efficientnet-b3', 'efficientnet-b4', 'efficientnet-b5', 'efficientnet-b6', 'efficientnet-b7']

    output = []
    i = 0
    # breakpoint()
    for zeroshot in zeroshot_list:
        for model_index, model_type in enumerate(available_models):

          for ssim in ssim_list:

            file_location_kappa = os.path.join('experiment result eff',
                                         str(model_type) + str('_') + str(ssim) + str(
                                             '_zs') + str(zeroshot) + str('_') + 'kappa.xlsx')

            file_location_acc = os.path.join('experiment result eff',
                                         str(model_type) + str('_') + str(ssim) + str(
                                             '_zs') + str(zeroshot) + str('_') + 'accuracy.xlsx')
            data_model_2_kappa = pd.read_excel(file_location_kappa, header=None)

            data_model_2_accuracy = pd.read_excel(file_location_acc, header=None)

            kappa_array = data_model_2_kappa[1][1:]*10
            mean_kappa = np.mean(kappa_array)
            std_kappa = kappa_array.std()
            percentile_kappa = kappa_array.quantile(0.10)
            print('kappa', mean_kappa)
            # breakpoint()
            # output[i][4] = mean_distance

            error_array = 1 - data_model_2_accuracy[1][1:]
            mean_error = np.mean(error_array)
            std_error = error_array.std()
            percentile_error = error_array.quantile(0.10)
            output.append([ssim, zeroshot, model_type, mean_error, mean_kappa, std_error, std_kappa, percentile_error,
                           percentile_kappa])
            i = i + 1
    #

    df = pd.DataFrame(output, columns=['SSIM', 'ZeroShot', 'ModelType','Error Rate','Kappa','Std Error','Std Kappa', 'percentile_error', 'percentile_kappa'])

    # Save the DataFrame to an Excel file
    df.to_excel('output_eff_cifar.xlsx', index=False)

if __name__ == '__main__':
    main()