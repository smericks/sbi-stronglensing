import numpy as np
import matplotlib.pyplot as plt
import corner
from scipy.stats import norm, gaussian_kde
from analysis.diagnostic_utils import check_coverage, check_parity
from astropy.visualization import simple_norm
import os

# === TRAINING PLOT FUNCTIONS ===

def plot_trainval_loss(inference, color, title = None, save_folder = None):

    plt.figure(figsize=(10, 10))
    plt.plot(inference.summary['training_loss'], color=color, label='Training', linestyle='-')
    plt.plot(inference.summary['validation_loss'], color=color, label='Validation', linestyle='--')

    plt.xlabel('Epoch', fontsize=15)
    plt.ylabel('Loss', fontsize=15)
    plt.tick_params(axis='both', which='major', labelsize=15)
    plt.tick_params(axis='both', which='minor', labelsize=15)
    plt.xticks(np.arange(0, len(inference.summary['training_loss']), 1))
    plt.legend(loc='upper right', fontsize=15, frameon=False)



    if title is not None:
        plt.title(title, fontsize=20)

    # save the plot
    if save_folder is not None:
        plt.savefig(os.path.join(save_folder, 'trainval_loss.png'))

    plt.show()

    return None

def plot_trainval_loss_multiple(inference_list, colors, legend_labels, title = None, save_folder = None):

    fig, ax = plt.subplots(figsize=(10, 10))
    if title is not None:
        ax.set_title(title, fontsize=20)

    for inference, color, legend_label in zip(inference_list, colors, legend_labels):
        ax.plot(inference.summary['training_loss'], color=color, label=legend_label, linestyle='-')
        ax.plot(inference.summary['validation_loss'], color=color, linestyle='--')

    ax.set_xlabel('Epoch', fontsize=15)
    ax.set_ylabel('Loss', fontsize=15)
    ax.tick_params(axis='both', which='major', labelsize=15)
    ax.tick_params(axis='both', which='minor', labelsize=15)

    # add a legend for the training and validation loss
    ax.plot([], [], color='gray', linestyle='-', label='Training')
    ax.plot([], [], color='gray', linestyle='--', label='Validation')

    ax.legend(loc='upper right', fontsize=15, frameon=False)

    plt.show()

    # save the plot
    if save_folder is not None:
        plt.savefig(os.path.join(save_folder, 'trainval_loss.png'))

    return ax


# === CORNER PLOT FUNCTIONS ===

def plot_corner(samples, truths, labels, color, legend_label=None, show_titles=False):

    if labels is None:
        labels = [
            r'$\theta_\mathrm{E}$', r'$\gamma_1$', r'$\gamma_2$', 
            r'$\gamma_\mathrm{lens}$', r'$e_1$', r'$e_2$', 
            r'$x_\mathrm{lens}$', r'$y_\mathrm{lens}$', 
            r'$x_\mathrm{src}$', r'$y_\mathrm{src}$'
        ]

    fig = corner.corner(
        np.array(samples),
        labels=labels,
        truths=truths,
        show_titles=show_titles,
        title_fmt='.2f',
        title_kwargs=dict(fontsize=15),
        label_kwargs=dict(fontsize=15),
        truth_color='k',
        levels=[0.68, 0.95],
        bins=20,
        plot_datapoints=False,
        fill_contours=True,
        max_n_ticks=3,
        smooth=1.0,
        hist_kwargs=dict(density=True, color=color, linewidth=2, histtype='step'),
        color=color,
        fig=None
    )

    for ax in fig.get_axes():
        ax.tick_params(axis="both", labelsize=12)
    
    # Delegate legend creation to our helper
    update_corner_legend(fig, color, legend_label=legend_label, fontsize = 20)

    return fig
    
def plot_corner_overlay(fig, samples, color, legend_label=None):

    fig = corner.corner(
        np.array(samples),
        levels=[0.68, 0.95],
        bins=20,
        plot_datapoints=False,
        fill_contours=True,
        max_n_ticks=3,
        smooth=1.0,
        hist_kwargs=dict(density=True, color=color, linewidth=2, histtype='step'),
        color=color,
        fig=fig
    )
    
    # Delegate legend update to our helper
    update_corner_legend(fig, color, legend_label=legend_label)

    return fig

def update_corner_legend(fig, color, legend_label, lw = 2, fontsize = 20):

    # Retrieve old handles & labels if they exist
    old_handles = getattr(fig, '_my_legend_handles', [])
    old_labels = getattr(fig, '_my_legend_labels', [])

    # Remove any existing figure-level legends (if present)
    if hasattr(fig, 'legends') and fig.legends:
        for legend in fig.legends:
            legend.remove()

    # Create a new handle
    new_handle = plt.Line2D([0], [0], color=color, lw=lw)

    # Append new handle & label
    old_handles.append(new_handle)
    old_labels.append(legend_label)

    # Save them back onto the figure object
    fig._my_legend_handles = old_handles
    fig._my_legend_labels = old_labels

    # Draw a fresh single legend containing all handles/labels so far
    fig.legend(
        handles=fig._my_legend_handles,
        labels=fig._my_legend_labels,
        loc='upper right',
        fontsize=fontsize,
        frameon=False
    )

def plot_corner_image(f, image, position=(0.7, 0.7, 0.175, 0.175), cmap="viridis"):

    norm = simple_norm(image,stretch='log',min_cut=1e-6)
    ax_inset = f.add_axes(position) 
    ax_inset.matshow(image, cmap=cmap, norm = norm)  
    ax_inset.axis("off") 

    return f


# === COVERAGE PLOT FUNCTIONS ===

def plot_coverage(alpha_list, ecp_list, error_list, colors, legend_labels, quality_metrics = None , title = None):

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_title(title, fontsize=20)

    # Draw y = x line for reference (perfect calibration)
    ax.plot([0,1], [0,1], color='k', linestyle=':', label='Perfect Calibration')

    for alpha, ecp, error, color, legend_label in zip(
            alpha_list, ecp_list, error_list, colors, legend_labels):
        
        # Calculate quality metrics if required
        if quality_metrics is not None:
            metric_values = check_coverage(ecp, alpha, metrics=quality_metrics)

            # Format metric values as strings
            metric_strings = [f"{key}: {value:.3f}" for key, value in metric_values.items()]

            # Append metrics to legend label
            legend_label += f' ({", ".join(metric_strings)})'
     
        # Plot calibration curve
        ax.plot(alpha, ecp, color=color, lw=2, label=legend_label)
        if error is not None:
            ax.fill_between(alpha, np.array(ecp) - np.array(error), np.array(ecp) + np.array(error), color=color, alpha=0.3)

    # Annotate overconfidence and underconfidence regions
    ax.text(0.25,0.75,'Underconfident', fontsize=15, ha='center')
    ax.text(0.75,0.25,'Overconfident', fontsize=15, ha='center')

    # Set labels and ticks
    ax.set_xlabel('Percentage Probability Volume', fontsize=15)
    ax.set_ylabel('Percentage Lenses with True Value in the Volume', fontsize=15)
    ax.tick_params(axis='both', which='major', labelsize=15)
    ax.tick_params(axis='both', which='minor', labelsize=15)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)

    # Update legend
    ax.legend(loc= 'upper left', fontsize=15, frameon=False)

    plt.show()

    return ax


# === PARITY PLOT FUNCTIONS ===

def plot_parity(true_values, predicted_values_list, error_values_list,
                colors, legend_labels, residual_distr = 'step', quality_metrics=None, title=None):
    
    fig, axs = plt.subplots(2, 2, figsize=(10, 10), sharey='row', sharex='col',  gridspec_kw=dict(hspace=0, wspace=0, height_ratios=(2.5,1), width_ratios=(5,1)))
    scatter_ax, _, residual_ax, hist_ax  = axs.flatten()
    fig.suptitle(title, fontsize=20)
    
    # Draw y = x line for reference (perfect prediction)
    scatter_ax.plot([true_values.min(), true_values.max()], [true_values.min(), true_values.max()], color='k', linestyle=':', label='Perfect Prediction')
    
    residuals_all = []
    
    for i, (predicted_values, error_values, color, legend_label) in enumerate(zip(predicted_values_list, error_values_list, colors, legend_labels)):
        
        residuals = np.array(predicted_values) - np.array(true_values)
        residuals_all.extend(residuals)
        
        # Calculate quality metrics if required
        if quality_metrics is not None:
            metric_values = check_parity(true_values, predicted_values, error_values, quality_metrics=quality_metrics)
            metric_strings = []
            for key, value in metric_values.items():
                if key != "bias" and key != "bias_err":  
                    metric_strings.append(f"{key}: {value:.2f}")
                elif key == "bias" and "bias_err" in metric_values:
                    metric_strings.append(f"bias: {metric_values['bias']:.3f} ± {metric_values['bias_err']:.3f}")
            legend_label += f' ({", ".join(metric_strings)})'

        # Plot parity plot
        scatter_ax.errorbar(
            true_values, predicted_values, yerr=error_values, fmt='.',
            color=color, markersize=10, capsize=5, elinewidth=2, capthick=2, label=legend_label
        )
        
        # Plot residuals
        residual_ax.errorbar(
            true_values, residuals, yerr=error_values, fmt='.', color=color,
            markersize=10, capsize=5, elinewidth=2, capthick=2
        )

        # Plot residuals histogram
        if residual_distr == 'step':
            hist_ax.hist(residuals, density=True, orientation='horizontal',  color=color, linewidth=1, histtype='step')
        elif residual_distr == 'kde':
            # hist_ax.hist(residuals, density=True, orientation='horizontal',  color=color, alpha = 0.3)
            kde = gaussian_kde(residuals)
            x_vals = np.linspace(min(residuals), max(residuals), 100)
            kde_vals = kde(x_vals)
            hist_ax.plot(kde_vals, x_vals, color=color, linestyle='-', linewidth=1)

        # bias = np.mean(np.array(residuals))
        # bias_error = np.std(np.array(residuals)) / np.sqrt(len(residuals))
        # ypos = 1.0 - 0.5 * ((1-len(predicted_values_list)) * 0.075) - i*0.075
        # hist_ax.text(0.5,  ypos, f"$\mu_{{bias}}$ = {bias:.3f}$\pm${bias_error:.3f}", horizontalalignment='center', transform=hist_ax.transAxes,size=9.5, color=color, verticalalignment='center')

    # Set labels and ticks for scatter plot
    scatter_ax.set_xlabel('True Value', fontsize=15)
    scatter_ax.set_ylabel('Predicted Value', fontsize=15)
    scatter_ax.tick_params(axis='both', which='major', labelsize=15)
    scatter_ax.legend(loc='upper left', fontsize=13, frameon=False)
    
    # Residual plot settings
    residual_ax.axhline(0, color='k', linestyle=':')
    residual_ax.set_xlabel('True Value', fontsize=15)
    residual_ax.set_ylabel('Residuals', fontsize=15)
    residual_ax.tick_params(axis='both', which='major', labelsize=15)
    
    # Residual histogram
    hist_ax.axhline(0, color='k', linestyle=':')
    
    # Hide the unused top-right subplot
    axs[0, 1].axis('off')
    hist_ax.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return scatter_ax, residual_ax, hist_ax

# def plot_parity(true_values, predicted_values_list, error_values_list,
#                 colors, legend_labels, residual_distr='kde', quality_metrics=None, title=None):
    
#     fig, axs = plt.subplots(2, 2, figsize=(10, 10), sharey='row', sharex='col',  gridspec_kw=dict(hspace=0, wspace=0, height_ratios=(2.5,1), width_ratios=(5,1)))
#     scatter_ax, _, residual_ax, hist_ax  = axs.flatten()
#     fig.suptitle(title, fontsize=20)
    
#     # Draw y = x line for reference (perfect prediction)
#     scatter_ax.plot([true_values.min(), true_values.max()], [true_values.min(), true_values.max()], color='k', linestyle=':', label='Perfect Prediction')
    
#     residuals_all = []
    
#     for predicted_values, error_values, color, legend_label in zip(
#             predicted_values_list, error_values_list, colors, legend_labels):
        
#         residuals = true_values - predicted_values
#         residuals_all.extend(residuals)
        
#         # Calculate quality metrics if required
#         if quality_metrics is not None:
#             metric_values = check_parity(true_values, predicted_values, error_values, quality_metrics =quality_metrics)
#             metric_strings = [f"{key}: {value:.2f}" for key, value in metric_values.items()]
#             legend_label += f' ({", ".join(metric_strings)})'
        
#         # Plot parity plot
#         scatter_ax.errorbar(
#             true_values, predicted_values, yerr=error_values, fmt='.',
#             color=color, markersize=10, capsize=5, elinewidth=2, capthick=2, label=legend_label
#         )
        
#         # Plot residuals
#         residual_ax.errorbar(
#             true_values, residuals, yerr=error_values, fmt='.', color=color,
#             markersize=10, capsize=5, elinewidth=2, capthick=2
#         )

#         # Plot residuals histogram
#         bias = np.mean(residuals)
#         bias_error = np.std(residuals) / np.sqrt(len(residuals))

#         if residual_distr == 'step':
#             hist_ax.hist(residuals, density=True, orientation='horizontal',  color=color, linewidth=1, histtype='step')
#         elif residual_distr == 'kde':
#             kde = gaussian_kde(residuals)
#             x_vals = np.linspace(min(residuals), max(residuals), 100)
#             kde_vals = kde(x_vals)
#             hist_ax.plot(kde_vals, x_vals, color=color, linestyle='-', linewidth=1)
        
#     # Set labels and ticks for scatter plot
#     scatter_ax.set_xlabel('True Value', fontsize=15)
#     scatter_ax.set_ylabel('Predicted Value', fontsize=15)
#     scatter_ax.tick_params(axis='both', which='major', labelsize=15)
#     scatter_ax.legend(loc='upper left', fontsize=15, frameon=False)
    
#     # Residual plot settings
#     residual_ax.axhline(0, color='k', linestyle=':')
#     residual_ax.set_xlabel('True Value', fontsize=15)
#     residual_ax.set_ylabel('Residuals', fontsize=15)
#     residual_ax.tick_params(axis='both', which='major', labelsize=15)
    
#     # Residual histogram
#     hist_ax.axhline(0, color='k', linestyle=':')
    
#     # Hide the unused top-right subplot
#     axs[0, 1].axis('off')
#     hist_ax.axis('off')
    
#     plt.tight_layout()
#     plt.show()
    
#     return scatter_ax, residual_ax, hist_ax

def plot_parity_simple(true_values, predicted_values_list, error_values_list,
                colors, legend_labels, quality_metrics=None, title=None):
    
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_title(title, fontsize=20)

    # Draw y = x line for reference (perfect prediction)
    ax.plot([true_values.min(), true_values.max()], [true_values.min(), true_values.max()], color='k', linestyle=':', label='Perfect Prediction')

    for predicted_values, error_values, color, legend_label in zip(
            predicted_values_list, error_values_list, colors, legend_labels):
        
        # Calculate quality metrics if required
        if quality_metrics is not None:
            metric_values = check_parity(true_values, predicted_values, error_values, metrics=quality_metrics)

            # Format metric values as strings
            metric_strings = [f"{key}: {value:.2f}" for key, value in metric_values.items()]

            # Append metrics to legend label
            legend_label += f' ({", ".join(metric_strings)})'
        
        # Plot parity plot
        ax.errorbar(
            true_values,
            predicted_values,
            yerr=error_values,
            fmt='.',
            color=color,
            markersize=10,
            capsize=5,
            elinewidth=2,
            capthick=2,
            label=legend_label
        )

    # Set labels and ticks
    ax.set_xlabel('True Value', fontsize=15)
    ax.set_ylabel('Predicted Value', fontsize=15)
    ax.tick_params(axis='both', which='major', labelsize=15)
    ax.tick_params(axis='both', which='minor', labelsize=15)

    # Update legend
    ax.legend(loc= 'upper left', fontsize=15, frameon=False)

    plt.show()

    return ax


