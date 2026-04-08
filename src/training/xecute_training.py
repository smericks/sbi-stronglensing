import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard.writer import SummaryWriter
import yaml
import json
from datetime import datetime
import pickle
import argparse


# USAGE: "python3 xecute_training.py config_files/p3_nsf.yaml"

# Local imports
try:
    # When running as a module from src directory
    from training.data_loader import load_data, load_h5_data
    from training.feature_extractor import get_feature_extractor
    from training.posterior_estimator import get_posterior_estimator
    from training.sbi_trainer import get_npe_model, train_npe_model
    from analysis.visualization_utils import plot_trainval_loss
except ImportError:
    # When running directly from training directory
    from data_loader import load_data, load_h5_data
    from feature_extractor import get_feature_extractor
    from posterior_estimator import get_posterior_estimator
    from sbi_trainer import get_npe_model, train_npe_model
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from analysis.visualization_utils import plot_trainval_loss

def execute_training(config_path):
    # Load configuration (flexible yaml or json)
    if config_path.endswith('.yaml'):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    elif config_path.endswith('.json'):
        with open(config_path, 'r') as f:
            config = json.load(f)
    else:
        raise ValueError(f"Unsupported configuration file type: {config_path}")
    
    # Define save directory
    save_directory = config['misc']['save_directory']
    density_estimator_type = config['posterior_estimator']['density_estimator']
    feature_extractor_type = config['feature_extractor']['network_architecture']
    date_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    folder_name = f"{density_estimator_type}_{feature_extractor_type}_{date_time}"
    save_folder = os.path.join(save_directory, f"{density_estimator_type}", folder_name)

    # Save config file
    # Save training results
    os.makedirs(save_folder, exist_ok=True)

    # save config before running anything
    with open(os.path.join(save_folder, "config.yaml"), "w") as f:
        yaml.dump(config, f)

    # Load data
    theta, x = load_h5_data(**config['data_loader'])
    #theta, x = load_data(**config['data_loader'])
    print(f"Loaded {len(theta)} training examples") #from {config['data_loader']['data_directory']}")

    # Get NPE model 
    npe_model = get_npe_model(**config['feature_extractor'], **config['posterior_estimator'])
    print(f"Created NPE model...")

    # Train NPE model
    print("Training NPE model...")
    inference, estimator, posterior = train_npe_model(npe_model, theta, x, 
        **config['sbi_trainer'], save_folder=save_folder)
    print("Training completed. Saving results...")

    # Save training results
    with open(os.path.join(save_folder, "inference.pkl"), "wb") as f:
        pickle.dump(inference, f)

    with open(os.path.join(save_folder, "estimator.pkl"), "wb") as f:
        pickle.dump(estimator, f)

    with open(os.path.join(save_folder, "posterior.pkl"), "wb") as f:
        pickle.dump(posterior, f)

    if config['misc']['save_loss_plot']:
        plot_trainval_loss(inference, 'k', save_folder=save_folder)

    print(f"Training results saved to {save_folder}.")

    return inference, estimator, posterior
        
        
if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Execute training")
    parser.add_argument("config_path", type=str, help="Path to the configuration file")
    args = parser.parse_args()

    # Execute training
    execute_training(args.config_path)  