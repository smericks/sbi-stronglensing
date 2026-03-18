import numpy as np
from tarp import get_tarp_coverage
#from scipy.integrate import trapz
from scipy.stats import kstest

# === COVERAGE FUNCTIONS ===

def get_coverage(samples, theta, references = "random", metric = "euclidean", num_alpha_bins = None, norm = True, num_bootstrap = 100, sigma=1, bootstrap=False, seed=None):

    if bootstrap:
        ecp_bootstrap, alpha = get_tarp_coverage(np.array(samples), np.array(theta), references = references, metric = metric, 
                                                 num_alpha_bins = num_alpha_bins, num_bootstrap = num_bootstrap, norm = norm, bootstrap = True, seed = seed)
        ecp = ecp_bootstrap.mean(axis=0)
        ecp_error = sigma * ecp_bootstrap.std(axis=0)
    else:
        ecp, alpha = get_tarp_coverage(np.array(samples), np.array(theta), references = references, metric = metric, 
                                       num_alpha_bins = num_alpha_bins, num_bootstrap = num_bootstrap, norm = norm, bootstrap = False, seed = seed)
        ecp_error = None

    return ecp, alpha, ecp_error 

def check_coverage(ecp, alpha, metrics=['ECE', 'AUC', 'ATC', 'KS']):

    # Initialize quality metrics dictionary
    metric_values = {}

    # Calculate quality metrics if required
    if 'ECE' in metrics:
        metric_values['ECE'] = np.mean(np.abs(np.array(ecp) - np.array(alpha)))

    #if 'AUC' in metrics:
    #    metric_values['AUC'] = trapz(np.array(ecp), np.array(alpha))

    if 'ATC' in metrics:
        midindex = alpha.shape[0] // 2
        metric_values['ATC'] = (ecp[midindex:] - alpha[midindex:]).sum().item()

    if 'KS' in metrics:
        metric_values['KS'] = kstest(np.array(ecp), np.array(alpha))[1]

    return metric_values


# === PARITY FUNCTIONS ===

def check_parity(true_values, predicted_values, error_values, quality_metrics = None):
        
        # Initialize quality metrics dictionary
        metric_values = {}

        # Calculate quality metrics if required
        if 'RMSE' in quality_metrics:
            metric_values['RMSE'] = np.sqrt(np.mean((np.array(predicted_values) - np.array(true_values)) ** 2))

        if 'accuracy' in quality_metrics:
            metric_values['accuracy'] = np.mean(np.abs(np.array(predicted_values) - np.array(true_values)))

        if 'precision' in quality_metrics:
            metric_values['precision'] = np.mean(error_values)

        if 'bias' in quality_metrics:
            residuals = np.array(predicted_values) - np.array(true_values)
            bias = np.mean(np.array(residuals))
            bias_error = np.std(np.array(residuals)) / np.sqrt(len(residuals))
            metric_values['bias'] = bias
            metric_values['bias_err'] = bias_error

        return metric_values




