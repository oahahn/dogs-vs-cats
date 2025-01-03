from dataset import get_dataloaders
from model import SimpleCNN
from train import train_model
from evaluate import evaluate_model, plot_misclassified_examples
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
import sys
import os

# Add the project root directory to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def main():
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # TensorBoard writer
    writer = SummaryWriter(log_dir="./logs")

    # Load data
    train_loader, val_loader, test_loader = get_dataloaders()

    # Initialize placeholder model
    model = SimpleCNN().to(device)

    # Set up loss function and optimizer
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)

    # Train the model
    for epoch in range(20):  # Number of epochs
        print(f"Epoch {epoch + 1}")
        train_model(model, train_loader, val_loader, loss_function, optimizer, scheduler, device, writer, epochs=1)

        # Evaluate the model after each epoch
        evaluate_model(model, val_loader, device, writer, epoch)

    torch.save(model.state_dict(), "model.pth")

    # Close writer
    writer.close()


def test():
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    _, _, test_loader = get_dataloaders()

    # Initialize placeholder model
    model = SimpleCNN().to(device)
    model.load_state_dict(torch.load("model.pth", weights_only=True))
    model.to(device)

    # Evaluate the model on the test set
    print("Evaluating on the test set...")
    evaluate_model(model, test_loader, device, None, "test")


def visualise():
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    _, val_loader, test_loader = get_dataloaders()

    # Initialize placeholder model
    model = SimpleCNN().to(device)
    model.load_state_dict(torch.load("model.pth", weights_only=True))
    model.to(device)

    # Visualize misclassified examples
    print("Visualizing misclassified examples...")
    plot_misclassified_examples(model, val_loader, device)


if __name__ == "__main__":
    main()
    # test()
    # visualise()