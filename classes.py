import os
from torch.utils.data import Dataset
from PIL import Image
from torch import nn

class ImageDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.transform = None

        # Image classes = Names of subfolders in root image folder
        self.classes = sorted([d for d in os.listdir(root_dir)
                               if os.path.isdir(os.path.join(root_dir, d))])

        # Map classes to numbers for easy labeling
        self.class_to_label = {class_name: label for label, class_name in enumerate(self.classes)}

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
    def __init__(self, image_size):
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