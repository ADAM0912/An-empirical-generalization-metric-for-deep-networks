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
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize


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
        output = model_classifier(feature_output.detach())
        criterion = nn.CrossEntropyLoss()
        loss = criterion(output, target)
        # breakpoint()
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

def validate(args, model, classifier, device, val_loader, criterion):
    num_classes = len(val_loader.dataset.classes)  # Assuming your dataset provides a 'classes' attribute
    class_correct = torch.zeros(num_classes, dtype=torch.float32).to(device)
    class_total = torch.zeros(num_classes, dtype=torch.float32).to(device)

    sum_loss, sum_correct = 0, 0
    margin = torch.Tensor([]).to(device)

    # switch to evaluation mode
    model.eval()
    classifier.eval()
    with torch.no_grad():
        for i, (data, target) in enumerate(val_loader):
            data, target = data.to(device).view(data.size(0), -1), target.to(device)

            # compute the output
            feature_output = model(data)
            output = classifier(feature_output)

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

def validate_and_collect_data_micro_average(args, model, classifier, device, val_loader, num_classes):
    model.eval()
    classifier.eval()
    all_targets = []
    all_scores = []

    with torch.no_grad():
        for data, target in val_loader:
            if args.fc:
                data, target = data.to(device).view(data.size(0), -1), target.to(device)
            else:
                data, target = data.to(device), target.to(device)
            feature_output = model(data)
            output = classifier(feature_output)

            # Store probabilities and true labels
            probabilities = torch.softmax(output, dim=1)
            all_scores.append(probabilities.cpu().numpy())
            all_targets.append(target.cpu().numpy())
            # breakpoint()
    all_scores = np.concatenate(all_scores, axis=0)
    all_targets_one_hot = np.eye(num_classes)[np.concatenate(all_targets, axis=0)]

    return all_targets_one_hot, all_scores

def plot_micro_averaged_roc(targets_one_hot, scores, num_classes):
    # Compute micro-average ROC curve and ROC area
    # breakpoint()
    fpr, tpr, _ = roc_curve(targets_one_hot.ravel(), scores.ravel())
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'Micro-Average ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Micro-Averaged Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()

def plot_macro_averaged_roc(targets_one_hot, scores, num_classes):
    fpr_dict = {}
    tpr_dict = {}
    roc_auc_dict = {}

    for i in range(num_classes):
        fpr_dict[i], tpr_dict[i], _ = roc_curve(targets_one_hot[:, i], scores[:, i])
        roc_auc_dict[i] = auc(fpr_dict[i], tpr_dict[i])

    # Compute macro-average ROC curve and ROC area
    all_fpr = np.unique(np.concatenate([fpr_dict[i] for i in range(num_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(num_classes):
        mean_tpr += np.interp(all_fpr, fpr_dict[i], tpr_dict[i])

    mean_tpr /= num_classes
    fpr_macro = all_fpr
    tpr_macro = mean_tpr
    roc_auc_macro = auc(fpr_macro, tpr_macro)

    plt.figure()
    plt.plot(fpr_macro, tpr_macro, color='darkorange', lw=2,
             label=f'Macro-Average ROC curve (area = {roc_auc_macro:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Macro-Averaged Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()


def get_all_preds_labels(model, classifier, loader, device):
    all_preds = torch.tensor([]).to(device)
    all_labels = torch.tensor([]).to(device)
    with torch.no_grad():
        for i, (data, target) in enumerate(loader):
            data, target = data.to(device).view(data.size(0), -1), target.to(device)
            outputs = model(data)
            # breakpoint()
            outputs = classifier(outputs)
            all_preds = torch.cat((all_preds, outputs), dim=0)
            all_labels = torch.cat((all_labels, target), dim=0)
    return all_preds, all_labels


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
    parser.add_argument('--learningrate', default=0.0001, type=float,
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
    parser.add_argument('--test_epoch', default='40', type=str)
    parser.add_argument('--prune', default=0.1, type=float)
    parser.add_argument('--zeroshot', default=0, type=int)
    parser.add_argument('--fc', action='store_true')
    args = parser.parse_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    # breakpoint()
    kwargs = {'num_workers': 20, 'pin_memory': True} if use_cuda else {}
    nchannels, nclasses = 3, 10
    if args.dataset == 'MNIST': nchannels = 1
    if args.dataset == 'CIFAR100': nclasses = 50+args.zeroshot

    num_sample = 10
    window = [0.0,0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    # nunits = [256, 512, 1024,2048, 4096, 8192, 16384]
    n_zeroshot = [0, 10, 20, 30, 40, 50]

    nunits=[64,128]

    for zeroshot in n_zeroshot:
        val_dataset = load_data('val', args.dataset, args.datadir, nchannels, zeroshot_number= zeroshot)

        val_loader = DataLoader(val_dataset, batch_size=args.batchsize, shuffle=False, **kwargs)
        for amount in nunits:

            for size in window:

                nclasses = 50 +zeroshot

                model_classifier = Classifier(n_feature=args.latent_dim, n_hidden=512, n_output=nclasses).to(device)
                best_model_file_cls = os.path.join('cls_checkpoints', str(amount), str(0.0), str(zeroshot), str(args.test_epoch), 'model_best.pth')
                # breakpoint()
                checkpoint = torch.load(best_model_file_cls)
                model_classifier.load_state_dict(checkpoint)

                model = nn.Sequential(nn.Linear(32 * 32 * nchannels, amount), nn.ReLU(),
                                      nn.Linear(amount, args.latent_dim), nn.ReLU())

                best_model_file_feature = os.path.join('backup_Checkpoints', str(amount), 'Checkpoints', 'feature',
                                                       str(args.test_epoch), 'model_best.pth')
                checkpoint = torch.load(best_model_file_feature)
                model.load_state_dict(checkpoint)
                # breakpoint()
                if size != 0:
                    pred_all = 0

                    for i in range(num_sample):
                        prune_weights_location = os.path.join('sample_weight', str(amount) + '_' + str(size) + '.h5')
                        with h5py.File(prune_weights_location, 'r') as hf:
                            # breakpoint()
                            weights_data = hf['first_layer_weights'][:]
                            weights_data = torch.from_numpy(weights_data)
                            model[0].weight.data = weights_data[i]
                            weights_data = hf['second_layer_weights'][:]
                            weights_data = torch.from_numpy(weights_data)
                            model[2].weight.data = weights_data[i]
                        # prune_weights_location = os.path.join('sample_weight_lp',
                        #                                       str(amount) + '_' + str(size) + '_zs' + str(zeroshot) + '.h5')
                        # with h5py.File(prune_weights_location, 'r') as hf:
                        #     weights_data = hf['linear_probe_weights'][:]
                        #     weights_data = torch.from_numpy(weights_data)
                        #     model_classifier.out.weight.data = weights_data[i]
                        model = model.to(device)
                        model_classifier = model_classifier.to(device)

                        predictions, labels = get_all_preds_labels(model, model_classifier, val_loader, device)
                        # breakpoint()
                        pred = predictions.max(1)[1]
                        pred_all += pred.eq(labels).sum().item()

                        class_correct = torch.zeros(nclasses, dtype=torch.float32).to(device)
                        class_total = torch.zeros(nclasses, dtype=torch.float32).to(device)
                        for t, p in zip(labels.view(-1), pred.view(-1)):
                            # breakpoint()
                            t = t.cpu().numpy()
                            p = p.cpu().numpy()
                            class_correct[t] += (t == p).item()
                            class_total[t] += 1

                    # breakpoint()
                    accuracy = pred_all/(num_sample*len(labels))
                    print('nunits:', amount, ' zeroshot', zeroshot, 'window size', size, 'Accuracy:', accuracy)
                    class_accuracy_rates = (class_correct / class_total)
                    output = class_accuracy_rates.detach().cpu().numpy()
                    df1 = pd.DataFrame(output)
                    file_location = os.path.join('experiment result',
                                                 str(amount) + str('_') + str(size) + str('_') + str(
                                                     args.test_epoch) + str(
                                                     '_zs') + str(zeroshot) + str('_') + 'accuracy.xlsx')
                    df1.to_excel(file_location)

                else:

                    model = model.to(device)
                    model_classifier = model_classifier.to(device)

                    predictions, labels = get_all_preds_labels(model, model_classifier, val_loader, device)
                    # breakpoint()
                    pred = predictions.max(1)[1]
                    accuracy = pred.eq(labels).sum().item()/len(labels)

                    class_correct = torch.zeros(nclasses, dtype=torch.float32).to(device)
                    class_total = torch.zeros(nclasses, dtype=torch.float32).to(device)
                    for t, p in zip(labels.view(-1), pred.view(-1)):
                        # breakpoint()
                        t = t.cpu().numpy()
                        p = p.cpu().numpy()
                        class_correct[t] += (t == p).item()
                        class_total[t] += 1
                    print('nunits:', amount, ' zeroshot',zeroshot ,'window size', size, 'Accuracy:', accuracy )
                    class_accuracy_rates = (class_correct / class_total)
                    output = class_accuracy_rates.detach().cpu().numpy()
                    df1 = pd.DataFrame(output)
                    file_location = os.path.join('experiment result',
                                                 str(amount) + str('_') + str(size) + str('_') + str(args.test_epoch) + str(
                                                     '_zs') + str(zeroshot) + str('_') + 'accuracy.xlsx')
                    df1.to_excel(file_location)

                # breakpoint()




if __name__ == '__main__':
    main()