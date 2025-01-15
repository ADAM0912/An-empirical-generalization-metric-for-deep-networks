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
# import h5py
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
from utils import CustomDataset
import clip
from scipy.special import softmax
from efficientnet_pytorch import EfficientNet
from sklearn.metrics import confusion_matrix
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

def load_data(split, dataset_name, datadir, preprocess, ssim, zeroshot_number):
    dataset = CustomDataset(root_dir=f'augmented_dataset_half/cifar100_{ssim:.2f}_{zeroshot_number}',
                            transform=preprocess)
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

    # Compute Kappa
    kappa = (P_0 - P_e) / (1 - P_e)

    return 1 - kappa

class CLIPClassifier(nn.Module):
    def __init__(self, clip_model,hidden_size, num_classes):
        super(CLIPClassifier, self).__init__()
        self.clip_model = clip_model
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, images):
        image_features = self.clip_model.encode_image(images)
        logits = self.classifier(image_features)
        return image_features, logits



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
    parser.add_argument('--checkpoints_feature', default='Checkpoints eff/feature', type=str)
    parser.add_argument('--checkpoints_mean', default='Checkpoints clip/mean', type=str)
    parser.add_argument('--checkpoints_clsf', default='Checkpoints eff/clsf', type=str)
    parser.add_argument('--checkpoints_clsm', default='Checkpoints clip/clsm', type=str)
    parser.add_argument('--test_epoch', default='40', type=str)
    parser.add_argument('--prune', default=0.1, type=float)
    parser.add_argument('--zeroshot', default=0, type=int)
    parser.add_argument('--fc', action='store_true')
    args = parser.parse_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # breakpoint()
    kwargs = {'num_workers': 20, 'pin_memory': True} if use_cuda else {}
    nchannels, nclasses = 3, 10
    if args.dataset == 'MNIST': nchannels = 1
    if args.dataset == 'CIFAR100': nclasses = 50+args.zeroshot


    available_models = ['efficientnet-b0', 'efficientnet-b1', 'efficientnet-b2', 'efficientnet-b3', 'efficientnet-b4',
                        'efficientnet-b5', 'efficientnet-b6', 'efficientnet-b7']
    batchsize = [32, 32, 32, 16, 16, 8, 8, 4]
    n_zeroshot = [0,10, 20, 30, 40, 50]
    ssim_list = [0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.0]


    for model_index, model_type in enumerate(available_models):
         for zeroshot in n_zeroshot:
            for ssim in ssim_list:

                args.zeroshot = zeroshot
                args.prune = prune
                nclasses = 50 + zeroshot
                model = EfficientNet.from_pretrained(model_type).to(device)
                model_classifier = Classifier(n_feature=1000, n_hidden=512, n_output=nclasses).to(device)

                model = model.to(device)
                preprocess = transforms.Compose([transforms.Resize(224), transforms.ToTensor(),
                                                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), ])

                if ssim == 1.0:
                    val_dataset = load_data('val', 'CIFAR100', args.datadir, preprocess, ssim, zeroshot_number=zeroshot)

                    val_loader = DataLoader(val_dataset, batch_size=batchsize[model_index], shuffle=False, **kwargs)
                else:
                    val_dataset = load_data('val', 'GEN', args.datadir, preprocess, ssim, zeroshot_number=zeroshot)
                    val_loader = DataLoader(val_dataset, batch_size=batchsize[model_index], shuffle=False, **kwargs)



                best_model_file_cls = os.path.join('cls_checkpoints_eff', str(model_type), str(zeroshot),
                                                   'checkpoint.pth')
                # breakpoint()
                checkpoint = torch.load(best_model_file_cls)
                model_classifier.load_state_dict(checkpoint)
                model_classifier = model_classifier.to(device)

                model.eval()
                model_classifier.eval()



                all_preds = []
                all_labels = []
                with torch.no_grad():
                    for inputs, labels in val_loader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        feature_out = model(inputs)
                        outputs = model_classifier(feature_out)

                        # Get predicted class indices
                        _, preds = torch.max(outputs, 1)
                        all_preds.extend(preds.cpu().numpy())
                        all_labels.extend(labels.cpu().numpy())

                    # Generate confusion matrix
                conf_matrix = confusion_matrix(all_labels, all_preds, labels=range(nclasses))
                kappa = compute_kappa(conf_matrix)
                print('model_type=', model_type, 'ssim=', ssim, 'zeroshot=', zeroshot, 'k_test=', kappa)
                df1 = pd.DataFrame(conf_matrix)
                models_name = ['efficientnet-b0', 'efficientnet-b1', 'efficientnet-b2', 'efficientnet-b3',
                               'efficientnet-b4', 'efficientnet-b5', 'efficientnet-b6', 'efficientnet-b7']
                file_location = os.path.join('experiment result eff',
                                             str(models_name[model_index]) + str('_') + str(ssim) + str(
                                                 '_zs') + str(zeroshot) + str('_') + 'kappa.xlsx')
                df1.to_excel(file_location)

                # breakpoint()




if __name__ == '__main__':
    main()
