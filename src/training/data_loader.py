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

def load_hdf5_images(file_path: str, key: str = "data", norm_images=False) -> torch.Tensor:
    """Loads image data from an HDF5 file."""
    with h5py.File(file_path, 'r') as f:
        data = f[key][:]  # Reads all data under the specified key
    
    if norm_images: # normalize each img by it's std. dev.
        data = data / np.std(data,axis=(-2,-1),keepdims=True) # compress over (..,numpix,numpix) dims.
    return torch.tensor(data, dtype=torch.float32)

def load_data(data_directory, parameter_labels):
    labels_path = os.path.join(data_directory, "metadata.csv")
    data_path = os.path.join(data_directory, "image_data.h5")
    labels = load_csv_labels(labels_path, parameter_labels)
    data = load_hdf5_images(data_path)
    return labels, data

def load_h5_data(data_file_list, image_type, parameter_labels, 
    norm_images=False):
    """
    Args:
        data_file_list ([string]): list of .h5 filepaths, .h5 file contains both images and metadata
        image_type (string): ex: 'image_flux_Roman_F158'. The key to extract images from .h5 file
        parameter_labels ([string]): keys of target parameters for training
        norm_images (bool): passed to load_hdf5_images()

    Returns: 
        theta, ims
    """
    # instantiate with first file in list
    h5_file_path = data_file_list[0]
    theta = load_hdf5_labels(h5_file_path, parameter_labels)
    ims = load_hdf5_images(h5_file_path,image_type,norm_images=norm_images)

    # then add other files in sequence
    if len(data_file_list) > 1:
        for i in range(1,len(data_file_list)):
            new_theta = load_hdf5_labels(data_file_list[i],parameter_labels)
            new_ims = load_hdf5_images(data_file_list[i],image_type,
                norm_images=norm_images)

            # add to existing tensors
            theta = torch.cat([theta, new_theta], dim=0)
            ims = torch.cat([ims, new_ims], dim=0)


    return theta, ims