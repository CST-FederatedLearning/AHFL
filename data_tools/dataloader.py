import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np

class Dataset(Dataset):
    def __init__(self, data, labels):
        """
        Initialize the dataset with data and labels.

        :param data: The input data.
        :param labels: The corresponding labels.
        """
        self.data = data
        self.labels = labels

    def __len__(self):
        """
        Return the number of samples in the dataset.

        :return: The length of the dataset.
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        Get a sample from the dataset at the specified index.

        :param idx: The index of the sample.
        :return: A tuple containing the data and the corresponding label.
        """
        return self.data[idx], self.labels[idx]

def get_datasets(args):
    """
    Get the training, validation, and test datasets.

    :param args: Command-line arguments.
    :return: Training, validation, and test datasets.
    """
    train_data = np.random.rand(100, 3, 224, 224).astype(np.float32)
    train_labels = np.random.randint(0, 10, 100)
    val_data = np.random.rand(20, 3, 224, 224).astype(np.float32)
    val_labels = np.random.randint(0, 10, 20)
    test_data = np.random.rand(20, 3, 224, 224).astype(np.float32)
    test_labels = np.random.randint(0, 10, 20)

    train_dataset = Dataset(train_data, train_labels)
    val_dataset = Dataset(val_data, val_labels)
    test_dataset = Dataset(test_data, test_labels)

    return train_dataset, val_dataset, test_dataset

def get_dataloaders(args, batch_size, datasets):
    """
    Get the data loaders for training, validation, and testing.

    :param args: Command-line arguments.
    :param batch_size: The batch size.
    :param datasets: A tuple of datasets (train_dataset, val_dataset, test_dataset).
    :return: Training, validation, and test data loaders.
    """
    train_dataset, val_dataset, test_dataset = datasets
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

def get_user_groups(train_set, val_set, test_set, args):
    """
    Get the user groups for clients.

    :param train_set: The training set.
    :param val_set: The validation set.
    :param test_set: The test set.
    :param args: Command-line arguments.
    :return: Training, validation, and test user groups.
    """
    num_clients = args.num_clients  # Assume args has a num_clients parameter
    train_user_groups = [[] for _ in range(num_clients)]
    val_user_groups = [[] for _ in range(num_clients)]
    test_user_groups = [[] for _ in range(num_clients)]

    # Simply distribute the data evenly among clients.
    for i in range(len(train_set)):
        client_idx = i % num_clients
        train_user_groups[client_idx].append(i)

    for i in range(len(val_set)):
        client_idx = i % num_clients
        val_user_groups[client_idx].append(i)

    for i in range(len(test_set)):
        client_idx = i % num_clients
        test_user_groups[client_idx].append(i)

    return train_user_groups, val_user_groups, test_user_groups