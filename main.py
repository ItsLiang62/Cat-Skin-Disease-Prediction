from classes import ImageDataset
from functions import get_mean_std
import torchvision.transforms.v2 as v2
import torch


if __name__ == "__main__":
    dataset = ImageDataset("Cat-Skin-Disease")

    base_transform = v2.Compose([
        v2.Resize((224, 224)),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True)
    ])

    mean_each_channel, std_each_channel = get_mean_std(dataset, base_transform)

    dataset.transform = v2.Compose([
        base_transform,
        v2.Normalize(mean_each_channel, std_each_channel)
    ])

    print(dataset.__getitem__(0))