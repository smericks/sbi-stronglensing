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
import yaml
import json
import csv

# Local imports
try:
    from feature_extractor import get_feature_extractor
    from posterior_estimator import get_posterior_estimator
    from data_loader import load_data, load_h5_data
except ImportError:
    from training.feature_extractor import get_feature_extractor
    from training.posterior_estimator import get_posterior_estimator
    from training.data_loader import load_data, load_h5_data

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
              save_checkpoints=False,
              save_loss_log=False,
              show_progress_bars = True,
              save_folder = None):
    
    # Build inference object
    if save_folder is not None:
        # Create a custom summary writer method and assign it
        NPE_C._default_summary_writer = make_summary_writer(save_folder)
        print(f"To view training progress, run:\ntensorboard --logdir={save_folder}")
    
    inference = NPE_C_CUSTOM(density_estimator=model, show_progress_bars=show_progress_bars)
    
    # Append simulations to inference object
    inference.append_simulations(theta, x, data_device = 'cpu') #TODO: why is it set to cpu?
    
    # Train 
    # TODO: trying to add checkpointing
    if save_checkpoints and save_folder is None:
        raise ValueError('Must specify save_folder to use checkpointing')
    
    estimator = inference.train(training_batch_size=batch_size,
                    learning_rate=learning_rate,
                    validation_fraction=validation_fraction,
                    stop_after_epochs=stop_after_epochs,
                    max_num_epochs=max_num_epochs,
                    resume_training=resume_training,
                    retrain_from_scratch=retrain_from_scratch,
                    show_train_summary=show_train_summary,
                    save_checkpoints=save_checkpoints, # NEW PARAMETER
                    save_loss_log=save_loss_log, # NEW PARAMETER
                    checkpoint_dir=save_folder) # NEW PARAMETER
    
    # Build posterior
    posterior = inference.build_posterior(density_estimator=estimator)
    
    return inference, estimator, posterior


def load_from_checkpoint(config_path,checkpoint_path):

    # load in configuration (flexible yaml or json)
    if config_path.endswith('.yaml'):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    elif config_path.endswith('.json'):
        with open(config_path, 'r') as f:
            config = json.load(f)
    else:
        raise ValueError(f"Unsupported configuration file type: {config_path}")
    
    # build up architecture
    npe_model = get_npe_model(**config['feature_extractor'], **config['posterior_estimator'])
    inference = NPE_C_CUSTOM(density_estimator=npe_model)

    # TODO: have to append some simulations for z-scoring to be correct
    # but am I loading in too many? (will it be way too slow?)
    theta, x = load_h5_data(**config['data_loader'])
    inference.append_simulations(theta, x, data_device = 'cpu')
    theta_zscore, x_zscore, _ = inference.get_simulations(starting_round=0)
    inference._neural_net =  inference._build_neural_net(
                theta_zscore.to("cpu"),
                x_zscore.to("cpu"),
            )
    # load in network weights from checkpoint
    checkpoint = torch.load(checkpoint_path,map_location='cpu')
    # add weights to inference object
    inference._neural_net.load_state_dict(checkpoint['model_state_dict'])
    inference._neural_net.to('cpu') # NOTE: same as checkpoint map_location..
    inference._neural_net.eval() # turn off stochastic portions so evaluations are stable

    # build posterior object (NOTE: deepcopy or no?)
    posterior = inference.build_posterior(density_estimator=inference._neural_net)

    # return the posterior object...
    return posterior


# new imports
from typing import Optional,Callable,Dict
from sbi.neural_nets.estimators import ConditionalDensityEstimator
from sbi.inference.posteriors import DirectPosterior
from pyknos.mdn.mdn import MultivariateGaussianMDN as mdn
from sbi.utils import check_dist_class
from torch.distributions import MultivariateNormal, Uniform
from torch import ones
from sbi.neural_nets.estimators.shape_handling import (
    reshape_to_batch_event,
    reshape_to_sample_batch_event,
)
from sbi.utils import test_posterior_net_for_multi_d_x
from torch.optim.adam import Adam
import time
from torch.nn.utils.clip_grad import clip_grad_norm_
from copy import deepcopy

""""
NOTE: This is a copy of the existing implementation of the .train() function in 
    sbi NPE_C() class, with added options for checkpointing and TODO a LR schedule
"""
class NPE_C_CUSTOM(NPE_C):
    # combine NPE_C and PosteriorEstimatorTrainer train() methods and customize
    # for LR schedule and for writing checkpoints 
    def train(
        self,
        num_atoms: int = 10,
        training_batch_size: int = 200,
        learning_rate: float = 5e-4,
        validation_fraction: float = 0.1,
        stop_after_epochs: int = 20,
        max_num_epochs: int = 2**31 - 1,
        clip_max_norm: Optional[float] = 5.0,
        calibration_kernel: Optional[Callable] = None,
        resume_training: bool = False,
        force_first_round_loss: bool = False,
        discard_prior_samples: bool = False,
        use_combined_loss: bool = False,
        retrain_from_scratch: bool = False,
        show_train_summary: bool = False,
        dataloader_kwargs: Optional[Dict] = None,
        save_checkpoints: bool = False,  # NEW PARAMETER
        save_loss_log: bool = False, # NEW PARAMETER
        checkpoint_dir: str = "./checkpoints"  # NEW PARAMETER.
    ) -> ConditionalDensityEstimator:
        r"""Return density estimator that approximates the distribution $p(\theta|x)$.

        Args:
            num_atoms: Number of atoms to use for classification.
            training_batch_size: Training batch size.
            learning_rate: Learning rate for Adam optimizer.
            validation_fraction: The fraction of data to use for validation.
            stop_after_epochs: The number of epochs to wait for improvement on the
                validation set before terminating training.
            max_num_epochs: Maximum number of epochs to run. If reached, we stop
                training even when the validation loss is still decreasing. Otherwise,
                we train until validation loss increases (see also
                ``stop_after_epochs``).
            clip_max_norm: Value at which to clip the total gradient norm in order to
                prevent exploding gradients. Use None for no clipping.
            calibration_kernel: A function to calibrate the loss with respect to the
                simulations ``x``. See Lueckmann, Gonçalves et al., NeurIPS 2017.
            resume_training: Can be used in case training time is limited, e.g. on a
                cluster. If ``True``, the split between train and validation set, the
                optimizer, the number of epochs, and the best validation log-prob will
                be restored from the last time ``.train()`` was called.
            force_first_round_loss: If ``True``, train with maximum likelihood,
                i.e., potentially ignoring the correction for using a proposal
                distribution different from the prior.
            discard_prior_samples: Whether to discard samples simulated in round 1, i.e.
                from the prior. Training may be sped up by ignoring such less targeted
                samples.
            use_combined_loss: Whether to train the neural net also on prior samples
                using maximum likelihood in addition to training it on all samples using
                atomic loss. The extra MLE loss helps prevent density leaking with
                bounded priors.
            retrain_from_scratch: Whether to retrain the conditional density
                estimator for the posterior from scratch each round.
            show_train_summary: Whether to print the number of epochs and validation
                loss and leakage after the training.
            dataloader_kwargs: Additional or updated kwargs to be passed to the training
                and validation dataloaders (like, e.g., a collate_fn)
            save_checkpoints: 
            save_loss_log:
            checkpoint_dir: path to save .pt for weights and .csv for loss log

        Returns:
            Density estimator that approximates the distribution $p(\theta|x)$.
        """

        # WARNING: sneaky trick ahead. We proxy the parent's `train` here,
        # requiring the signature to have `num_atoms`, save it for use below, and
        # continue. It's sneaky because we are using the object (self) as a namespace
        # to pass arguments between functions, and that's implicit state management.
        self._num_atoms = num_atoms
        self._use_combined_loss = use_combined_loss

        # Load data from most recent round.
        self._round = max(self._data_round_index)

        if self._round > 0:
            # Set the proposal to the last proposal that was passed by the user. For
            # atomic SNPE, it does not matter what the proposal is. For non-atomic
            # SNPE, we only use the latest data that was passed, i.e. the one from the
            # last proposal.
            proposal = self._proposal_roundwise[-1]
            self.use_non_atomic_loss = (
                isinstance(proposal, DirectPosterior)
                and isinstance(proposal.posterior_estimator.net._distribution, mdn)
                and isinstance(self._neural_net.net._distribution, mdn)
                and check_dist_class(
                    self._prior, class_to_check=(Uniform, MultivariateNormal)
                )[0]
            )

            algorithm = "non-atomic" if self.use_non_atomic_loss else "atomic"
            print(f"Using SNPE-C with {algorithm} loss")

            if self.use_non_atomic_loss:
                # Take care of z-scoring, pre-compute and store prior terms.
                self._set_state_for_mog_proposal()


        if self._round == 0 and self._neural_net is not None:
            assert force_first_round_loss or resume_training, (
                "You have already trained this neural network. After you had trained "
                "the network, you again appended simulations with `append_simulations"
                "(theta, x)`, but you did not provide a proposal. If the new "
                "simulations are sampled from the prior, you can set "
                "`.train(..., force_first_round_loss=True`). However, if the new "
                "simulations were not sampled from the prior, you should pass the "
                "proposal, i.e. `append_simulations(theta, x, proposal)`. If "
                "your samples are not sampled from the prior and you do not pass a "
                "proposal and you set `force_first_round_loss=True`, the result of "
                "SNPE will not be the true posterior. Instead, it will be the proposal "
                "posterior, which (usually) is more narrow than the true posterior."
            )

        # Calibration kernels proposed in Lueckmann, Gonçalves et al., 2017.
        if calibration_kernel is None:

            def default_calibration_kernel(x):
                return ones([len(x)], device=self._device)

            calibration_kernel = default_calibration_kernel

        # Starting index for the training set (1 = discard round-0 samples).
        start_idx = int(discard_prior_samples and self._round > 0)

        # For non-atomic loss, we can not reuse samples from previous rounds as of now.
        # SNPE-A can, by construction of the algorithm, only use samples from the last
        # round. SNPE-A is the only algorithm that has an attribute `_ran_final_round`,
        # so this is how we check for whether or not we are using SNPE-A.
        if self.use_non_atomic_loss or hasattr(self, "_ran_final_round"):
            start_idx = self._round

        # Set the proposal to the last proposal that was passed by the user. For
        # atomic SNPE, it does not matter what the proposal is. For non-atomic
        # SNPE, we only use the latest data that was passed, i.e. the one from the
        # last proposal.
        proposal = self._proposal_roundwise[-1]

        train_loader, val_loader = self.get_dataloaders(
            start_idx,
            training_batch_size,
            validation_fraction,
            resume_training,
            dataloader_kwargs=dataloader_kwargs,
        )
        # First round or if retraining from scratch:
        # Call the `self._build_neural_net` with the rounds' thetas and xs as
        # arguments, which will build the neural network.
        # This is passed into NeuralPosterior, to create a neural posterior which
        # can `sample()` and `log_prob()`. The network is accessible via `.net`.
        if self._neural_net is None or retrain_from_scratch:
            # Get theta,x to initialize NN
            theta, x, _ = self.get_simulations(starting_round=start_idx)
            # Use only training data for building the neural net (z-scoring transforms)

            self._neural_net = self._build_neural_net(
                theta[self.train_indices].to("cpu"),
                x[self.train_indices].to("cpu"),
            )

            theta = reshape_to_sample_batch_event(
                theta.to("cpu"), self._neural_net.input_shape
            )
            x = reshape_to_batch_event(x.to("cpu"), self._neural_net.condition_shape)
            test_posterior_net_for_multi_d_x(self._neural_net, theta, x)

            del theta, x

        # Move entire net to device for training.
        self._neural_net.to(self._device)

        if not resume_training:
            self.optimizer = Adam(list(self._neural_net.parameters()), lr=learning_rate)
            self.epoch, self._val_loss = 0, float("Inf")

        # initialize checkpointing dir if requested
        if save_checkpoints or save_loss_log:
            os.makedirs(checkpoint_dir, exist_ok=True)
        # initialize loss log if requested
        if save_loss_log:
            loss_log_path = os.path.join(checkpoint_dir, 
                f"training_log_round_{self._round}.csv")
            # Create/open CSV file
            loss_log_file = open(loss_log_path, 'w', newline='')
            loss_writer = csv.writer(loss_log_file)
            loss_writer.writerow(['epoch', 'train_loss', 'val_loss', 'epoch_duration_sec'])
            print(f"Logging training metrics to: {loss_log_path}")

        # START training loop!!
        while self.epoch <= max_num_epochs and not self._converged(
            self.epoch, stop_after_epochs
        ):
            # Train for a single epoch.
            self._neural_net.train()
            train_loss_sum = 0
            epoch_start_time = time.time()
            for batch in train_loader:
                self.optimizer.zero_grad()
                # Get batches on current device.
                theta_batch, x_batch, masks_batch = (
                    batch[0].to(self._device),
                    batch[1].to(self._device),
                    batch[2].to(self._device),
                )

                train_losses = self._loss(
                    theta_batch,
                    x_batch,
                    masks_batch,
                    proposal,
                    calibration_kernel,
                    force_first_round_loss=force_first_round_loss,
                )
                train_loss = torch.mean(train_losses)
                train_loss_sum += train_losses.sum().item()

                train_loss.backward()
                if clip_max_norm is not None:
                    clip_grad_norm_(
                        self._neural_net.parameters(), max_norm=clip_max_norm
                    )
                self.optimizer.step()

            self.epoch += 1

            train_loss_average = train_loss_sum / (
                len(train_loader) * train_loader.batch_size  # type: ignore
            )
            self._summary["training_loss"].append(train_loss_average)

            # Calculate validation performance.
            self._neural_net.eval()
            val_loss_sum = 0

            with torch.no_grad():
                for batch in val_loader:
                    theta_batch, x_batch, masks_batch = (
                        batch[0].to(self._device),
                        batch[1].to(self._device),
                        batch[2].to(self._device),
                    )
                    # Take negative loss here to get validation log_prob.
                    val_losses = self._loss(
                        theta_batch,
                        x_batch,
                        masks_batch,
                        proposal,
                        calibration_kernel,
                        force_first_round_loss=force_first_round_loss,
                    )
                    val_loss_sum += val_losses.sum().item()

            # Take mean over all validation samples.
            self._val_loss = val_loss_sum / (
                len(val_loader) * val_loader.batch_size  # type: ignore
            )
            # Log validation loss for every epoch.
            self._summary["validation_loss"].append(self._val_loss)
            self._summary["epoch_durations_sec"].append(time.time() - epoch_start_time)

            # NEW: checkpointing
            # ADDED: Save checkpoint every epoch
            # TODO: why is it os.makedirs every loop? move this out
            # of the training loop and check for both checkpointing and loss_log!!
            if save_checkpoints:
                #import os
                #os.makedirs(checkpoint_dir, exist_ok=True)
                checkpoint_path = os.path.join(
                    checkpoint_dir, 
                    f"checkpoint_round_{self._round}_epoch_{self.epoch}.pt"
                )
                torch.save({
                    'epoch': self.epoch,
                    'model_state_dict': self._neural_net.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'train_loss': train_loss_average,
                    'val_loss': self._val_loss,
                    'round': self._round,
                }, checkpoint_path)
                
                # Optional: Also save best model based on validation loss
                if self._val_loss < self._best_val_loss:
                    best_path = os.path.join(
                        checkpoint_dir, 
                        f"best_model_round_{self._round}.pt"
                    )
                    torch.save({
                        'epoch': self.epoch,
                        'model_state_dict': self._neural_net.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'train_loss': train_loss_average,
                        'val_loss': self._val_loss,
                        'round': self._round,
                    }, best_path)

            if save_loss_log:
                loss_writer.writerow([
                    self.epoch,
                    train_loss_average,
                    self._val_loss,
                    time.time() - epoch_start_time
                ])
                loss_log_file.flush()

            self._maybe_show_progress(self._show_progress_bars, self.epoch)


        self._report_convergence_at_end(self.epoch, stop_after_epochs, max_num_epochs)

        # Update summary.
        self._summary["epochs_trained"].append(self.epoch)
        self._summary["best_validation_loss"].append(self._best_val_loss)

        # Close loss log
        if save_loss_log:
            loss_log_file.close()
            print(f"Training log saved to: {loss_log_path}")

        # Update tensorboard and summary dict.
        self._summarize(round_=self._round)

        # Update description for progress bar.
        if show_train_summary:
            print(self._describe_round(self._round, self._summary))

        # Avoid keeping the gradients in the resulting network, which can
        # cause memory leakage when benchmarking.
        self._neural_net.zero_grad(set_to_none=True)

        return deepcopy(self._neural_net)

        

