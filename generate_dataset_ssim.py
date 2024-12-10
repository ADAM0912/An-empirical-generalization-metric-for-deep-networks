import torch
from torchvision import datasets, transforms
import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from skimage.metrics import structural_similarity as ssim
# from train_gan import Generator, Discriminator
import os
import torchvision.transforms.functional as TF
class Autoencoder(nn.Module):

    def __init__(self):
        super(Autoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 7)
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 7),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()  # Using Sigmoid to output values between 0 and 1
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class CNNAutoencoder(nn.Module):
    def __init__(self):
        super(CNNAutoencoder, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1),  # Output: (16, 16, 16)
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),  # Output: (32, 8, 8)
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # Output: (64, 4, 4)
            nn.ReLU(),
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),  # Output: (32, 8, 8)
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),  # Output: (16, 16, 16)
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, 3, stride=2, padding=1, output_padding=1),  # Output: (3, 32, 32)
            nn.Sigmoid()  # Use sigmoid to output values between 0 and 1
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# Load CIFAR-100 validation dataset
transform = transforms.Compose([transforms.ToTensor()])
valid_dataset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)

# Define a list of autoencoders
autoencoders = []

# Load pre-trained weights for each autoencoder if available
for i in range(8):
    autoencoder = CNNAutoencoder()
    avg_ssim = i*0.05+0.60
    # avg_ssim = 0.65
    autoencoder.load_state_dict(torch.load(f'./AE_models/autoencoder_checkpoint_ssim_{avg_ssim:.2f}.pth'))
    autoencoders.append(autoencoder)

# Create directories to save generated datasets
os.makedirs('generated_datasets', exist_ok=True)

# Function to generate modified images using autoencoder
def generate_modified_images(dataset, autoencoder, output_directory):
    os.makedirs(output_directory, exist_ok=True)
    for i, (image, label) in enumerate(dataset):
        # Pass the image through the autoencoder
        reconstructed_image = autoencoder(image.unsqueeze(0))
        # Save the reconstructed image
        class_directory = os.path.join(output_directory, str(label))
        os.makedirs(class_directory, exist_ok=True)
        output_filename = os.path.join(class_directory, f'image_{i}.png')
        torchvision.utils.save_image(reconstructed_image, output_filename)


# Generate modified datasets with different SSIM ranges
for i, autoencoder in enumerate(autoencoders):

    avg_ssim_range = i*0.05+0.60  # SSIM range from 0.6 to 0.95
    print(avg_ssim_range)
    output_directory = f'generated_datasets/cifar100_{avg_ssim_range:.2f}'
    generate_modified_images(valid_dataset, autoencoder, output_directory)
