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

    # Load data
    theta, x = load_h5_data(**config['data_loader'])
    #theta, x = load_data(**config['data_loader'])
    print(f"Loaded {len(theta)} training examples") #from {config['data_loader']['data_directory']}")

    # Get NPE model 
    npe_model = get_npe_model(**config['feature_extractor'], **config['posterior_estimator'])
    print(f"Created NPE model...")

    # Train NPE model
    print("Training NPE model...")
    inference, estimator, posterior = train_npe_model(npe_model, theta, x, **config['sbi_trainer'], save_folder=save_folder)
    print("Training completed. Saving results...")

    
    # Save training results
    os.makedirs(save_folder, exist_ok=True)

    with open(os.path.join(save_folder, "config.yaml"), "w") as f:
        yaml.dump(config, f)

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

    # # Print help message
    # print("Usage: python xecute_training.py <config_path>")

  


#   def _default_summary_writer(self) -> SummaryWriter:
#         """Return summary writer logging to method- and simulator-specific directory."""

#         method = self.__class__.__name__
#         logdir = Path(
#             get_log_root(), method, datetime.now().isoformat().replace(":", "_")
#         )
#         return SummaryWriter(logdir)

# def _summarize(
#         self,
#         round_: int,
#     ) -> None:
#         """Update the summary_writer with statistics for a given round.

#         During training several performance statistics are added to the summary, e.g.,
#         using `self._summary['key'].append(value)`. This function writes these values
#         into summary writer object.

#         Args:
#             round: index of round

#         Scalar tags:
#             - epochs_trained:
#                 number of epochs trained
#             - best_validation_loss:
#                 best validation loss (for each round).
#             - validation_loss:
#                 validation loss for every epoch (for each round).
#             - training_loss
#                 training loss for every epoch (for each round).
#             - epoch_durations_sec
#                 epoch duration for every epoch (for each round)

#         """

#         # Add most recent training stats to summary writer.
#         self._summary_writer.add_scalar(
#             tag="epochs_trained",
#             scalar_value=self._summary["epochs_trained"][-1],
#             global_step=round_ + 1,
#         )

#         self._summary_writer.add_scalar(
#             tag="best_validation_loss",
#             scalar_value=self._summary["best_validation_loss"][-1],
#             global_step=round_ + 1,
#         )

#         # Add validation loss for every epoch.
#         # Offset with all previous epochs.
#         offset = (
#             torch.tensor(self._summary["epochs_trained"][:-1], dtype=torch.int)
#             .sum()
#             .item()
#         )
#         for i, vlp in enumerate(self._summary["validation_loss"][offset:]):
#             self._summary_writer.add_scalar(
#                 tag="validation_loss",
#                 scalar_value=vlp,
#                 global_step=offset + i,
#             )

#         for i, tlp in enumerate(self._summary["training_loss"][offset:]):
#             self._summary_writer.add_scalar(
#                 tag="training_loss",
#                 scalar_value=tlp,
#                 global_step=offset + i,
#             )

#         for i, eds in enumerate(self._summary["epoch_durations_sec"][offset:]):
#             self._summary_writer.add_scalar(
#                 tag="epoch_durations_sec",
#                 scalar_value=eds,
#                 global_step=offset + i,
#             )

#         self._summary_writer.flush()