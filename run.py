import os
import torch
import module
from module import train_dataset
from PIL import Image
from pathlib import Path

def main():
    model = module.NeuralNetwork()
    model.load_state_dict(torch.load('model.pth'))
    model.eval()

    def predict(image_name):
        transform = train_dataset.transform

        root_dir = Path("Input")
        root_dir.mkdir(parents=True, exist_ok=True)

        image = transform(Image.open(os.path.join(root_dir, image_name))).unsqueeze(0)

        with torch.no_grad():
            logits = model(image)

            # Convert logits to probabilities using Softmax
            probabilities = torch.nn.functional.softmax(logits.flatten(), dim=0)
            label = torch.argmax(probabilities).item()

            return train_dataset.label_to_class[label], probabilities[label]

    prediction, confidence = predict("example.png")
    print(f"Prediction: {prediction} with {confidence*100:.2f}% confidence")

if __name__ == "__main__":
    main()