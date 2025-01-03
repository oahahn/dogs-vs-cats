import torch

def predict_on_test_set(model, test_loader, device):
    """
    Generate predictions on the test set.

    Args:
        model (nn.Module): Trained model.
        test_loader (DataLoader): DataLoader for the test set.
        device (torch.device): Device to use for computation.

    Returns:
        list of tuples: List of (filename, predicted_label).
    """
    model.eval()
    predictions = []

    with torch.no_grad():
        for images, filenames in test_loader:
            images = images.to(device)

            # Get model predictions
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)  # Get class with highest score

            # Store predictions with filenames
            predictions.extend(zip(filenames, predicted.cpu().numpy()))

    return predictions