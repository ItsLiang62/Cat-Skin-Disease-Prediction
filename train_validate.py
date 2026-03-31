import torch
import module
import torch.nn as nn
from torch.utils.data import DataLoader

def main():

    train_dataset = module.train_dataset
    val_dataset = module.val_dataset

    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else 'cpu'

    model = module.NeuralNetwork().to(device)

    # Multi-class loss function
    criterion = nn.CrossEntropyLoss().to(device)

    # Adam optimizer for updating weights and biases based on learning rate
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

    best_val_loss = float('inf')

    for epoch in range(10):
        model.train()
        train_loss = 0.0

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

            train_loss += loss.item()

        # Model validation
        model.eval()
        val_loss = 0.0

        # Prevent calculating and storing variables for computing loss wrt weight/bias
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)

                logits = model(x)
                loss = criterion(logits, y)
                val_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        print(f"Epoch {epoch+1} completed | Average Training Loss: {avg_train_loss} | Average Validation Loss: {avg_val_loss}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            print(f"Best model updated. Validation loss: {best_val_loss}")
            torch.save(model.state_dict(), f"model.pth")


if __name__ == "__main__":
    main()