import os
import torch
import torchvision.transforms.v2 as v2
from torch import nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image

image_size = 224

class ImageDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.transform = None

        # Image classes = Names of subfolders in root image folder
        self.classes = sorted([d for d in os.listdir(root_dir)
                               if os.path.isdir(os.path.join(root_dir, d))])

        # Map classes to numbers for easy labeling
        self.class_to_label = {class_name: label for label, class_name in enumerate(self.classes)}
        self.label_to_class = {label: class_name for label, class_name in enumerate(self.classes)}

        # Raw input images and their labels
        self.images = []
        self.image_to_label = {}

        for class_name in self.classes:
            class_dir = os.path.join(root_dir, class_name)
            label = self.class_to_label[class_name]

            for image_name in os.listdir(class_dir):
                # Accept only selected image file types
                if image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_path = os.path.join(class_dir, image_name)
                    self.images.append(image_path)
                    self.image_to_label[image_path] = label

    def __len__(self):
        return len(self.image_to_label)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        label = self.image_to_label[image_path]

        image = Image.open(image_path)
        if image.mode != 'RGB':
            image = image.convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        # Flatten image tensor
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(

            # First layer
            # Linear layer
            # Input count = RGB image tensor element count
            nn.Linear(3*image_size**2, 512),

            # ReLU activation function to introduce non-linearity
            nn.ReLU(),

            # Second layer
            nn.Linear(512, 224),
            nn.ReLU(),

            # Third and final layer
            nn.Linear(224, 4)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

def get_mean_std(dataset, transform):
    dataset.transform = transform
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

    total_sum_each_channel = torch.zeros(3)

    # To calculate variance = (sum of squares / n) - (mean) ** 2
    total_sum_sq_each_channel = torch.zeros(3)

    # Every channel has same total pixels count
    total_pixels = 0

    for batch, _ in dataloader:
        batch_size, channels_per_image, rows_per_channel, cols_per_row = batch.shape
        batch = batch.view(batch_size, channels_per_image, -1)
        batch_sum_each_channel = batch.sum(dim=[0, 2])
        batch_sum_sq_each_channel = (batch**2).sum(dim=[0, 2])

        total_sum_each_channel += batch_sum_each_channel
        total_sum_sq_each_channel += batch_sum_sq_each_channel
        total_pixels += batch_size * rows_per_channel * cols_per_row

    mean_each_channel = total_sum_each_channel / total_pixels
    var_each_channel = total_sum_sq_each_channel / total_pixels - mean_each_channel ** 2
    std_each_channel = torch.sqrt(var_each_channel)

    return mean_each_channel, std_each_channel

train_dataset = ImageDataset("Cat-Skin-Disease/Training")
val_dataset = ImageDataset("Cat-Skin-Disease/Validation")

base_transform = v2.Compose([
    v2.Resize((image_size, image_size)),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True)
])

mean_std_each_channel = get_mean_std(train_dataset, base_transform)

train_dataset.transform = v2.Compose([
    base_transform,
    v2.Normalize(*mean_std_each_channel)
])

val_dataset.transform = train_dataset.transform