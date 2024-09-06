
import os
from sklearn.manifold import TSNE

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
import os
from PIL import Image
def tSNEplot(feature,label,mean ):



    # feature = normalize(feature)
    if feature.shape[0]<30:
        perplexity=5
    else:
        perplexity = 30
    ts = TSNE(n_components=2, init='pca', random_state=0, perplexity=perplexity)

    sample_cls = 20
    count = 0
    final_feature=0
    for i in set(label):
        if count == 0:

            final_feature = feature[label==i][0:sample_cls]
            final_label   = label[label==i][0:sample_cls]
            final_mean = mean[label==i][0:sample_cls]
            count+=1
        else:
            final_feature = np.append(final_feature, feature[label==i][0:sample_cls], axis=0)
            final_label = np.append(final_label, label[label==i][0:sample_cls])
            final_mean = np.append(final_mean, mean[label == i][0:sample_cls], axis=0)

    # breakpoint()
    final_feature = np.append(final_feature, final_mean, axis=0)
    final_label = np.append(final_label, final_label)
    result = ts.fit_transform(final_feature)



    sns.set(rc={'figure.figsize':(11.7,8.27)})
    # palette = sns.color_palette("coolwarm", len(set(sketchlabel1))+2)
    palette = sns.color_palette("Spectral", len(set(label)))

    # label = []
    # new_TU_class = {v:k for k,v in TU_class.items()}
    # for i in range(seen_label.shape[0]):
    #     label.append(new_TU_class[seen_label[i]])
    len_feature = int(len(final_feature)/2)
    len_mean = len(mean)
    # breakpoint()
    sns.scatterplot(x=result[0:len_feature, 0], y=result[0:len_feature,1], hue=final_label[0:len_feature], legend=True, palette=palette)
    sns.scatterplot(x=result[len_feature:, 0], y=result[len_feature:, 1], hue=final_label[len_feature:], legend=True,  marker="+",
                    palette=palette)
    # sns.scatterplot(x=result[len_feature:, 0], y=result[len_feature:, 1], hue=seen_label[len_feature:], legend=False, s=100, palette=palette)
    plt.show()

class TwoCropTransform:
    """Create two crops of the same image"""
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]

class EarlyStopping:
    def __init__(self, patience=5, verbose=False, delta=0, path='checkpoint.pt'):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
            verbose (bool): If True, prints a message for each validation loss improvement.
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
            path (str): Path for the checkpoint to be saved to.
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model, cls_err,args):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, cls_err, args.test_epoch, args.nunits, args.prune, args.zeroshot)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, cls_err, args.test_epoch, args.nunits, args.prune,args.zeroshot)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, cls_err, epoch, nunits, prune, zeroshot):
        '''Saves model when validation loss decreases.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')


        self.val_loss_min = val_loss

        # directory = os.path.join(self.path, str(epoch))
        directory = self.path
        if not os.path.isdir(directory):
            os.makedirs(directory)
        best_model_file = os.path.join(directory, 'checkpoint.pth')
        torch.save(model.state_dict(), best_model_file)



        # existing_df = pd.read_excel('output.xlsx', sheet_name='Sheet1', header=None)
        # existing_df.iloc[:len(output.flatten()), 0] = output.flatten()
        # existing_df.to_excel('output.xlsx', sheet_name='Sheet1', index=False, header=None)

def largest_unstructured(module, name, amount=0.5):
    # Get the parameter (tensor) to be pruned
    tensor = getattr(module, name)

    # Compute the number of weights to prune
    num_prune = int(tensor.nelement() * amount)

    # Compute the threshold: we'll prune weights with magnitudes larger than this value
    threshold = torch.topk(torch.abs(tensor).view(-1), k=num_prune, largest=True).values[-1]

    # Create a mask
    mask = torch.abs(tensor) <= threshold

    return mask.float()

class AdjustableCNN(nn.Module):
    def __init__(self, num_classes=100, scale_factor=1):
        super(AdjustableCNN, self).__init__()
        # Adjusting the number of filters/neurons based on the scaling factor
        conv1_filters = int(16 * scale_factor)
        conv2_filters = int(32 * scale_factor)
        fc1_neurons = int(512)

        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=conv1_filters, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=conv1_filters, out_channels=conv2_filters, kernel_size=3, stride=1,
                               padding=1)

        # Fully connected layers
        self.fc1 = nn.Linear(in_features=conv2_filters * 8 * 8, out_features=fc1_neurons)

        # Activation function
        self.relu = nn.ReLU()

        # Pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        # First Convolutional Layer
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)

        # Second Convolutional Layer
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)

        # Flattening the output for the fully connected layer
        x = x.view(x.size(0), -1)

        # First Fully Connected Layer
        x = self.fc1(x)
        x = self.relu(x)

        # Second Fully Connected Layer

        return x

    def count_parameters(self):
        """Function to count the number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

def warmup_learning_rate(args, epoch, batch_id, total_batches, optimizer):
    if args.warm and epoch <= args.warm_epochs:
        p = (batch_id + (epoch - 1) * total_batches) / \
            (args.warm_epochs * total_batches)
        lr = args.warmup_from + p * (args.warmup_to - args.warmup_from)

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

def millions_formatter(x, pos):
    return '%1.0fM' % (x / 1e6)

def marginalize(result, dimension):
    if not isinstance(dimension, np.ndarray):
        dimension = np.array(dimension)
    dimension = dimension.flatten()
    if isinstance(dimension, (int, float)):
        assert dimension in range(0,
                                  5), 'the dimensions you want to marginalize to should be given as a number between 0 and 4'
        d = 1
    else:
        assert dimension.all() in range(0,
                                        5), 'the dimensions you want to marginalize to should be given as a vector of numbers 0 to 4'
        d = len(result['X1D'])

    if d == 1 and ('marginals' in result.keys()) and len(result['marginals']) - 1 >= dimension:
        marginal = result['marginals'][dimension]
        weight = result['marginalsW'][dimension]
        x = result['marginalsX'][dimension]
    else:
        if not ('Posterior' in result.keys()):
            raise ValueError('marginals cannot be computed anymore because posterior was dropped')
        else:
            assert (np.shape(result['Posterior']) == np.shape(
                result['weight'])), 'dimensions mismatch in marginalization'

            if len(dimension) == 1:
                x = result['X1D'][int(dimension)][:]
            else:
                x = np.nan

            # calculate mass at each grid point
            marginal = result['weight'] * result['Posterior']
            weight = result['weight']

            for i in range(0, d):
                if not (any(i == dimension)) and marginal.shape[i] > 1:
                    marginal = np.sum(marginal, i, keepdims=True)
                    weight = np.sum(weight, i, keepdims=True) / (np.max(result['X1D'][i]) - np.min(result['X1D'][i]))
            marginal = marginal / weight

            marginal = np.squeeze(marginal)
            weight = np.squeeze(weight)
            if len(dimension) == 1:
                x = x.flatten()
                marginal = marginal.flatten()
                weight = weight.flatten()
            else:
                x = []
                for ix in np.sort(dimension): x.append(result['X1D'][ix])

    return (marginal, x, weight)

class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        for label in os.listdir(root_dir):
            label_dir = os.path.join(root_dir, label)
            if os.path.isdir(label_dir):
                for image_name in os.listdir(label_dir):
                    self.image_paths.append(os.path.join(label_dir, image_name))
                    self.labels.append(int(label))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label