from classes import ImageDataset, NeuralNetwork
from functions import get_mean_std
import torchvision.transforms.v2 as v2
import torch
from torch.utils.data import DataLoader
import torch.nn as nn

def main():
    image_size = 224

    if image_size ** 2 < 512:
        image_size = 224

    dataset = ImageDataset("Cat-Skin-Disease")

    base_transform = v2.Compose([
        v2.Resize((image_size, image_size)),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True)
    ])

    mean_each_channel, std_each_channel = get_mean_std(dataset, base_transform)

    dataset.transform = v2.Compose([
        base_transform,
        v2.Normalize(mean_each_channel, std_each_channel)
    ])

    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else 'cpu'

    model = NeuralNetwork(image_size).to(device)

    # Multi-class loss function
    criterion = nn.CrossEntropyLoss().to(device)

    # Adam optimizer for updating weights and biases based on learning rate
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

    for epoch in range(10):
        total_loss = 0.0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            # Forward pass to produce raw score of each class
            logits = model(x)

            # Compute loss for prediction due to incorrect prediction or low confidence
            loss = criterion(logits, y)

            # Reset loss wrt weight/bias (gradient) of each weight/bias
            optimizer.zero_grad()

            # Back propagation to calculate gradient of each weight/bias
            loss.backward()

            # Update each weight/bias based on its gradient
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1} completed. Average Loss: {total_loss/len(train_loader)}")


if __name__ == "__main__":
    main()