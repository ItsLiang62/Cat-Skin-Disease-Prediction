from torch.utils.data import DataLoader
import torch

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