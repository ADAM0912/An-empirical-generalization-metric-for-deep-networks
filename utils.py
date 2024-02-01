import os
import shutil
import numpy as np
import torch
import torch.nn as nn

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

        directory = os.path.join(self.path, str(epoch))
        if not os.path.isdir(directory):
            os.makedirs(directory)
        best_model_file = os.path.join(directory, 'checkpoint.pth')
        torch.save(model.state_dict(), best_model_file)



        # existing_df = pd.read_excel('output.xlsx', sheet_name='Sheet1', header=None)
        # existing_df.iloc[:len(output.flatten()), 0] = output.flatten()
        # existing_df.to_excel('output.xlsx', sheet_name='Sheet1', index=False, header=None)

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