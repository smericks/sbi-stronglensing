import torch
import h5py
import pandas as pd
import numpy as np
import os

def load_csv_labels(file_path: str, parameter_labels: list) -> torch.Tensor:
    """Loads parameter labels from a CSV file and converts them to a tensor."""
    df = pd.read_csv(file_path)
    return torch.tensor(df[parameter_labels].values, dtype=torch.float32)

def load_hdf5_labels(file_path: str, parameter_labels: list) -> torch.Tensor:
    """Loads parameter labels from a .h5 file and converts them to a tensor."""
    with h5py.File(file_path, 'r') as f:
        label_vals = []
        for label in parameter_labels:
            label_vals.append(f[label][:])
        label_vals = np.column_stack(label_vals)
    return torch.tensor(label_vals, dtype=torch.float32)

def load_hdf5_images(file_path: str, key: str = "data") -> torch.Tensor:
    """Loads image data from an HDF5 file."""
    with h5py.File(file_path, 'r') as f:
        data = f[key][:]  # Reads all data under the specified key
    return torch.tensor(data, dtype=torch.float32)

def load_data(data_directory, parameter_labels):
    labels_path = os.path.join(data_directory, "metadata.csv")
    data_path = os.path.join(data_directory, "image_data.h5")
    labels = load_csv_labels(labels_path, parameter_labels)
    data = load_hdf5_images(data_path)
    return labels, data