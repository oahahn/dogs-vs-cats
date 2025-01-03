import torch
import matplotlib.pyplot as plt
import numpy as np


def evaluate_model(model, dataloader, device, writer=None, epoch="test"):
    """
    Evaluate the model's performance and optionally log metrics to TensorBoard.

    Args:
        model (nn.Module): The trained model.
        dataloader (DataLoader): DataLoader for validation or test set.
        device (torch.device): Device to use (CPU or GPU).
        writer (SummaryWriter, optional): TensorBoard writer for logging metrics.
        epoch (int or str): Epoch number or "test" (for logging purposes).

    Returns:
        dict: Dictionary with evaluation metrics (e.g., accuracy, loss).
    """
    model.eval()
    all_preds = []
    all_labels = []
    running_loss = 0.0
    loss_function = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = loss_function(outputs, labels)
            running_loss += loss.item()

            # Collect predictions and true labels
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Compute metrics
    accuracy = sum([p == l for p, l in zip(all_preds, all_labels)]) / len(all_labels)

    # Optionally log metrics to TensorBoard
    if writer:
        writer.add_scalar("Loss/Test" if epoch == "test" else "Loss/Validation", running_loss / len(dataloader), epoch)
        writer.add_scalar("Accuracy/Test" if epoch == "test" else "Accuracy/Validation", accuracy, epoch)

    # Print results for the test set
    if epoch == "test":
        print(f"Test Loss: {running_loss / len(dataloader):.4f}, Accuracy: {accuracy:.4f}")

    return {"loss": running_loss / len(dataloader), "accuracy": accuracy}


def plot_misclassified_examples(model, dataloader, device, num_examples=10):
    """
    Visualize examples where the model misclassified the input.

    Args:
        model (nn.Module): Trained model.
        dataloader (DataLoader): DataLoader for the test/validation set.
        device (torch.device): Device to run the model on.
        num_examples (int): Number of misclassified examples to show.

    Returns:
        None
    """
    model.eval()
    misclassified_images = []
    misclassified_labels = []
    predicted_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            # Find misclassified examples
            misclassified_idx = (predicted != labels).cpu().numpy()
            if np.any(misclassified_idx):
                misclassified_images.extend(images[misclassified_idx].cpu())
                misclassified_labels.extend(labels[misclassified_idx].cpu())
                predicted_labels.extend(predicted[misclassified_idx].cpu())

            # Stop if we have enough examples
            if len(misclassified_images) >= num_examples:
                break

    # Plot the misclassified examples
    misclassified_images = misclassified_images[:num_examples]
    misclassified_labels = misclassified_labels[:num_examples]
    predicted_labels = predicted_labels[:num_examples]

    fig, axes = plt.subplots(1, num_examples, figsize=(15, 5))
    for img, true_label, pred_label, ax in zip(misclassified_images, misclassified_labels, predicted_labels, axes):
        img = img.permute(1, 2, 0)  # Convert from (C, H, W) to (H, W, C)
        ax.imshow(img)
        ax.set_title(f"True: {true_label}, Pred: {pred_label}")
        ax.axis("off")
    plt.tight_layout()
    plt.show()