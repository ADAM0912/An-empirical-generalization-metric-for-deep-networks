import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, ConcatDataset, Subset
import argparse
from torchvision import transforms, datasets
import pandas as pd
import os
import shutil
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from utils import CustomDataset
from efficientnet_pytorch import EfficientNet

class Classifier(nn.Module):
    def __init__(self, n_feature=512, n_hidden=1024, n_output=250):
        super(Classifier, self).__init__()
        self.out = torch.nn.Linear(n_feature, n_output)

    def forward(self, x_layer):
        feature = self.out(x_layer)
        return feature


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



def get_all_preds_labels(model, classifier, loader, device):
    all_preds = torch.tensor([]).to(device)
    all_labels = torch.tensor([]).to(device)
    model.eval()
    classifier.eval()
    with torch.no_grad():
        for i, (data, target) in enumerate(loader):
            data, target = data.to(device), target.to(device)
            feature_output= model(data)
            outputs = classifier(feature_output)
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
    parser.add_argument('--checkpoints_feature', default='Checkpoints eff/feature', type=str)
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

    ssim_list = [0.60,0.65, 0.70, 0.75, 0.80,0.85, 0.90,0.95,1.0]
    n_zeroshot = [0,10, 20, 30, 40, 50]
    available_models = ['efficientnet-b0', 'efficientnet-b1', 'efficientnet-b2', 'efficientnet-b3', 'efficientnet-b4', 'efficientnet-b5', 'efficientnet-b6', 'efficientnet-b7']
    batchsize = [32, 32, 32, 16, 16,8,8,4]

    for model_index, model_type in enumerate(available_models):
         for zeroshot in n_zeroshot:
            for ssim in ssim_list:
                nclasses = 50 + zeroshot

                model = EfficientNet.from_pretrained(model_type)
                model_classifier = Classifier(n_feature=1000, n_hidden=512, n_output=nclasses)

                model = model.to(device)
                preprocess = transforms.Compose([transforms.Resize(224), transforms.ToTensor(),
                                                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), ])
                model_classifier = model_classifier.to(device)
                # breakpoint()
                if ssim == 1.0:
                    val_dataset = load_data('val', 'CIFAR100', args.datadir, preprocess, ssim, zeroshot_number=zeroshot)

                    val_loader = DataLoader(val_dataset, batch_size=batchsize[model_index], shuffle=False, **kwargs)
                else:
                    val_dataset = load_data('val', 'CIFAR100', args.datadir, preprocess, ssim, zeroshot_number=zeroshot)
                    val_loader = DataLoader(val_dataset, batch_size=batchsize[model_index], shuffle=False, **kwargs)
                nclasses = 50+zeroshot

                file_location = os.path.join(args.checkpoints_feature,
                                             str(model_type))
                best_model_file_feature = os.path.join(file_location, 'model_best.pth')
                checkpoint = torch.load(best_model_file_feature)
                model.load_state_dict(checkpoint)

                best_model_file_cls = os.path.join('cls_checkpoints_eff',  str(model_type), str(zeroshot), 'checkpoint.pth')
                checkpoint = torch.load(best_model_file_cls)
                model_classifier.load_state_dict(checkpoint)
                model_classifier = model_classifier.to(device)

                predictions, labels = get_all_preds_labels(model, model_classifier, val_loader, device)
                pred = predictions.max(1)[1]
                accuracy = pred.eq(labels).sum().item() / len(labels)

                class_correct = torch.zeros(nclasses, dtype=torch.float32).to(device)
                class_total = torch.zeros(nclasses, dtype=torch.float32).to(device)
                for t, p in zip(labels.view(-1), pred.view(-1)):
                    # breakpoint()
                    t = t.cpu().numpy()
                    p = p.cpu().numpy()
                    class_correct[t] += (t == p).item()
                    class_total[t] += 1
                print('model_type:', model_type, ' zeroshot', zeroshot, 'ssim', ssim, 'Accuracy:', accuracy)
                class_accuracy_rates = (class_correct / class_total)
                output = class_accuracy_rates.detach().cpu().numpy()
                df1 = pd.DataFrame(output)
                models_name = ['efficientnet-b0', 'efficientnet-b1', 'efficientnet-b2', 'efficientnet-b3', 'efficientnet-b4', 'efficientnet-b5', 'efficientnet-b6', 'efficientnet-b7']
                file_location = os.path.join('experiment result eff',
                                             str(models_name[model_index]) + str('_') + str(ssim) + str(
                                                 '_zs') + str(zeroshot) + str('_') + 'accuracy.xlsx')
                df1.to_excel(file_location)



                # breakpoint()




if __name__ == '__main__':
    main()