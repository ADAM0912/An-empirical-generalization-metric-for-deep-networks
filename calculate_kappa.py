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
import h5py
from scipy.special import softmax


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

def calculate_mean(output, label):
    mean_dict = np.zeros((len(set(label)),output.shape[1]))
    for i in set(label):
       mean_dict[i] = np.mean(output[label == i], axis=0)



    return mean_dict

def save_checkpoint(state, directory):

    if not os.path.isdir(directory):
        os.makedirs(directory)
    checkpoint_file = os.path.join(directory, 'checkpoint.pth')
    best_model_file = os.path.join(directory, 'model_best.pth')
    torch.save(state, checkpoint_file)
    shutil.copyfile(checkpoint_file, best_model_file)

def calculate_kappa (out,class_idx,out_label):
    # breakpoint()
    out_bin = np.zeros((out.shape))
    prob = softmax(out, axis=1)
    rank = np.argsort(prob,axis=1)
    hi = prob[:, class_idx]
    remaining_columns = np.delete(prob, class_idx, axis=1)
    hmax_remaining = np.max(remaining_columns, axis=1)

    # threshold1 = 0.7
    threshold1 = 0.2
    threshold2 = 0.2

    # Applying conditions
    # all_below_threshold1 = np.all(prob < threshold1, axis=1)
    all_below_threshold1 = np.all(prob < threshold1, axis=1)
    abs_diff_le_threshold2 = np.abs(hi - hmax_remaining) <= threshold2
    hi_gt_hmax_by_threshold2 = hi - hmax_remaining > threshold2
    hmax_gt_hi_by_threshold2 = hmax_remaining - hi > threshold2

    # Counting based on conditions
    # d = np.sum(all_below_threshold1)
    a = np.sum(abs_diff_le_threshold2 & ~all_below_threshold1)
    b = np.sum(hi_gt_hmax_by_threshold2 & ~all_below_threshold1)
    c = np.sum(hmax_gt_hi_by_threshold2 & ~all_below_threshold1)
    d = 10000-a-b-c

    M = a+b+c+d
    p1 = (a+d)/M
    p2 = ((a+b)*(a+c)+(c+d)*(b+d))/(M*M)
    if 1-p2 == 0:
        k_test = 0.5
    else:
        k_test = abs((p1-p2))/(1-p2)

    # breakpoint()
    return k_test

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
    parser.add_argument('--fc', action='store_true')
    args = parser.parse_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    # breakpoint()
    kwargs = {'num_workers': 20, 'pin_memory': True} if use_cuda else {}
    nchannels, nclasses = 3, 10
    if args.dataset == 'MNIST': nchannels = 1
    if args.dataset == 'CIFAR100': nclasses = 50+args.zeroshot

    # size = [256, 512, 1024, 2048, 4096, 8192]
    window = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    # window = [0, 0.2, 0.4, 0.6, 0.8]
    # window = [0.1, 0.3, 0.5, 0.7, 0.9]

    num_zeroshot = [0,10,20,30,40,50]

    size = [64,128]
    # window = [0.0]
    # num_zeroshot = [0]
    for zeroshot in num_zeroshot:
        val_dataset = load_data('val', args.dataset, args.datadir, nchannels, zeroshot)

        val_loader = DataLoader(val_dataset, batch_size=args.batchsize, shuffle=False, **kwargs)
        for num_unit in size:
            for prune in window:

    # create an initial model
                nclasses = 50 + zeroshot
                model = nn.Sequential(nn.Linear(32 * 32 * nchannels, num_unit), nn.ReLU(), nn.Linear(num_unit, args.latent_dim), nn.ReLU())

                model = model.to(device)

                best_model_file_feature = os.path.join('backup_Checkpoints', str(num_unit), 'Checkpoints', 'feature',
                                                       str(args.test_epoch), 'model_best.pth')
                checkpoint = torch.load(best_model_file_feature)
                model.load_state_dict(checkpoint)

                model_classifier = Classifier(n_feature=args.latent_dim, n_hidden=512, n_output=nclasses).to(device)
                best_model_file_cls = os.path.join('cls_checkpoints', str(num_unit), str(0.0), str(zeroshot),
                                                   str(args.test_epoch), 'model_best.pth')
                checkpoint = torch.load(best_model_file_cls)
                model_classifier.load_state_dict(checkpoint)
                num_sample = 10
                if prune != 0:
                    div_matrix = np.zeros((nclasses, 1))
                    for i in range(num_sample):
                        prune_weights_location = os.path.join('sample_weight', str(num_unit) + '_' + str(prune) + '.h5')
                        with h5py.File(prune_weights_location, 'r') as hf:
                            # breakpoint()
                            weights_data = hf['first_layer_weights'][:]
                            weights_data = torch.from_numpy(weights_data)
                            model[0].weight.data = weights_data[i]
                            weights_data = hf['second_layer_weights'][:]
                            weights_data = torch.from_numpy(weights_data)
                            model[2].weight.data = weights_data[i]
                        prune_weights_location = os.path.join('sample_weight_lp',
                                                              str(num_unit) + '_' + str(prune) + '_zs' + str(zeroshot) + '.h5')
                        with h5py.File(prune_weights_location, 'r') as hf:
                            weights_data = hf['linear_probe_weights'][:]
                            weights_data = torch.from_numpy(weights_data)
                            model_classifier.out.weight.data = weights_data[i]
                            model = model.to(device)
                            model_classifier = model_classifier.to(device)

                        for j, (data, target) in enumerate(val_loader):
                            data, target = data.to(device).view(data.size(0), -1), target.to(device)
                            # breakpoint()
                            if j == 0:
                                out = model_classifier(model(data)).cpu().detach().numpy()
                                out_label = target.cpu().detach().numpy()
                            else:
                                out = np.concatenate(
                                    (out, model_classifier(model(data)).cpu().detach().numpy()), axis=0)
                                out_label = np.concatenate((out_label, target.cpu().data.numpy()), axis=0)


                        for k in range(nclasses):
                            k_test = calculate_kappa(out, k, out_label)
                            div_matrix[k] += k_test

                    div_matrix = div_matrix/num_sample
                    k_test_mean = np.mean(div_matrix)
                    print('nunit=', num_unit, 'window=', prune, 'zeroshot=', zeroshot, 'k_test=', k_test_mean)
                    # breakpoint()
                    df1 = pd.DataFrame(div_matrix, index=list(set(out_label)))
                    file_location = os.path.join('experiment result',
                                                 str(num_unit) + str('_') + str(prune) + str('_') + str(
                                                     args.test_epoch) + str('_zs') + str(
                                                     zeroshot) + str('_') + 'kappa_lp.xlsx')
                    df1.to_excel(file_location)



                else:

                    for j, (data, target) in enumerate(val_loader):
                        data, target = data.to(device).view(data.size(0), -1), target.to(device)
                        # breakpoint()
                        if j == 0:
                            out = model_classifier(model(data)).cpu().detach().numpy()
                            out_label = target.cpu().detach().numpy()
                        else:
                            out = np.concatenate((out, model_classifier(model(data)).cpu().detach().numpy()), axis=0)
                            out_label = np.concatenate((out_label, target.cpu().data.numpy()), axis=0)

                    predicted_labels = np.argmax(out, axis=1)

                    div_matrix = np.zeros((nclasses, 1))
                    for i in range(nclasses):
                        k_test = calculate_kappa(out, i, out_label)
                        div_matrix[i] = k_test
                    k_test_mean = np.mean(div_matrix)
                    # error_all = 1-sum(np.argmax(sim, axis=1) == out_label)/sim.shape[0]
                    # error_all = 1-sum(np.argmax(sim, axis=1) == out_label)/sim.shape[0]
                    print('nunit=', num_unit, 'window=', prune, 'zeroshot=', zeroshot, 'k_test=', k_test_mean)
                    # breakpoint()
                    df1 = pd.DataFrame(div_matrix, index=list(set(out_label)))
                    file_location = os.path.join('experiment result',
                                                 str(num_unit) + str('_') + str(prune) + str('_') + str(args.test_epoch) + str('_zs') + str(
                                                     zeroshot) + str('_') + 'kappa_lp.xlsx')
                    df1.to_excel(file_location)

    # existing_df = pd.read_excel('output.xlsx', sheet_name='Sheet1', header=None)
    # existing_df.iloc[:len(div_matrix.flatten()), 1] = div_matrix.flatten()
    # existing_df.to_excel('output.xlsx', sheet_name='Sheet1',index=False, header=None)

if __name__ == '__main__':
    main()