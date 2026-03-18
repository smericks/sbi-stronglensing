import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import h5py
from sbi.inference import NPE_C
from datetime import datetime
import os
import pickle
from torch.utils.tensorboard.writer import SummaryWriter

# Local imports
try:
    from feature_extractor import get_feature_extractor
    from posterior_estimator import get_posterior_estimator
except ImportError:
    from training.feature_extractor import get_feature_extractor
    from training.posterior_estimator import get_posterior_estimator

def make_summary_writer(save_folder):
    """Creates a function that returns a SummaryWriter with the specified save folder."""
    def _summary_writer(self):
        return SummaryWriter(save_folder)
    return _summary_writer

def get_npe_model(network_architecture="cnn", 
                feature_dim=64, 
                weights=None, 
                input_shape=(180, 180), 
                in_channels=1, out_channels_per_layer=[32, 64, 128, 256], num_conv_layers=2, num_linear_layers=2, num_linear_units=256, kernel_size=5, pool_kernel_size=2,
                density_estimator="maf",
                z_score_theta=None,
                z_score_x=None,
                hidden_features=50,
                num_transforms=5,
                num_components=1,
                num_bins=10,
                use_batch_norm=True,
                device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Trains a Neural Posterior Estimation model using a feature extractor and density estimator.
    
    Args:
        theta_train: Training parameters (N x param_dim tensor)
        x_train: Training data (N x C x H x W tensor)
        feature_dim: Dimension of extracted features
        batch_size: Batch size for training
        density_estimator: Type of density estimator ('maf', 'nsf', or 'mdn')
        hidden_features: Number of hidden features in density estimator
        num_transforms: Number of transforms in flow-based models
        num_components: Number of components for MDN
        num_bins: Number of bins for NSF
        ...
        device: Device to train on
    """
    
    # Get feature extractor
    embedding_net = get_feature_extractor(
        network_architecture=network_architecture,
        output_dim=feature_dim,
        weights=weights,
        input_shape=input_shape,
        in_channels=in_channels,
        out_channels_per_layer=out_channels_per_layer,
        num_conv_layers=num_conv_layers,
        num_linear_layers=num_linear_layers,
        num_linear_units=num_linear_units,
        kernel_size=kernel_size,
        pool_kernel_size=pool_kernel_size
    )
    
    # Get density estimator with feature extractor as embedding network
    npe_model = get_posterior_estimator(
        density_estimator=density_estimator,
        z_score_theta=z_score_theta,
        z_score_x=z_score_x,
        hidden_features=hidden_features,
        num_transforms=num_transforms,
        num_components=num_components,
        num_bins=num_bins,
        embedding_net=embedding_net,
        use_batch_norm=use_batch_norm
    )

    return npe_model

def train_npe_model(model, theta, x, 
              batch_size = 200,
              learning_rate = 5e-4,
              validation_fraction = 0.1,
              stop_after_epochs = 20,
              max_num_epochs = 2**31 - 1,
              resume_training = False,
              retrain_from_scratch = False,
              show_train_summary = True,
              show_progress_bars = True,
              save_folder = None):
    
    # Build inference object
    if save_folder is not None:
        # Create a custom summary writer method and assign it
        NPE_C._default_summary_writer = make_summary_writer(save_folder)
        print(f"To view training progress, run:\ntensorboard --logdir={save_folder}")
    
    inference = NPE_C(density_estimator=model, show_progress_bars=show_progress_bars)
    
    # Append simulations to inference object
    inference.append_simulations(theta, x, data_device = 'cpu')
    
    # Train 
    estimator = inference.train(training_batch_size=batch_size,
                    learning_rate=learning_rate,
                    validation_fraction=validation_fraction,
                    stop_after_epochs=stop_after_epochs,
                    max_num_epochs=max_num_epochs,
                    resume_training=resume_training,
                    retrain_from_scratch=retrain_from_scratch,
                    show_train_summary=show_train_summary)
    
    # Build posterior
    posterior = inference.build_posterior(density_estimator=estimator)
    
    return inference, estimator, posterior
