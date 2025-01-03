import os
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, random_split


# Custom Dataset class
class CatsAndDogsDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(data_dir) if f.endswith(".jpg")]
        self.labels = [0 if "cat" in f else 1 for f in self.image_files]  # 0 for cat, 1 for dog

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.data_dir, self.image_files[idx])
        image = Image.open(img_path).convert("RGB")  # Ensure all images are 3-channel
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


# Handling the Test Set
class TestDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(data_dir) if f.endswith(".jpg")]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.data_dir, self.image_files[idx])
        image = Image.open(img_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
        
        return image
    

def get_dataloaders(train_dir="data/train", test_dir="data/test1", batch_size=32, val_split=0.2):
    """
    Create DataLoaders for training, validation, and testing.

    Args:
        train_dir (str): Path to the training dataset directory.
        test_dir (str): Path to the testing dataset directory.
        batch_size (int): Number of samples per batch.
        val_split (float): Fraction of the training data to use for validation.

    Returns:
        train_loader, val_loader, test_loader (DataLoader): DataLoaders for train, validation, and test sets.
    """
    # Define transformations for training and testing
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),  # Randomly flip images horizontally
        transforms.RandomRotation(degrees=15),  # Rotate images randomly
        transforms.Resize((128, 128)),  # Resize to 128x128
        transforms.ToTensor(),         # Convert to tensor
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize((128, 128)),  # Resize to 128x128
        transforms.ToTensor(),         # Convert to tensor
    ])

    # Load training dataset
    train_dataset = CatsAndDogsDataset(data_dir=train_dir, transform=train_transform)

    # Split training dataset into train and validation
    val_size = int(len(train_dataset) * val_split)
    train_size = len(train_dataset) - val_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

    # Load test dataset
    test_dataset = TestDataset(data_dir=test_dir, transform=test_transform)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
