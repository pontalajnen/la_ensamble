import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc
import torch
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance, wilcoxon, ttest_rel, shapiro, mannwhitneyu, ttest_ind
from sklearn.metrics import cohen_kappa_score
import json
import seaborn as sns
import pandas as pd
from argparse import ArgumentParser, BooleanOptionalAction
from utils.paths import *
from collections import defaultdict
from statsmodels.stats.multitest import multipletests

def main():

    parser = ArgumentParser()
    parser.add_argument("--save_file_name", type=str, default="", help="The name of the file to save the results to.")
    parser.add_argument("--model_1_path_file", type=str, default="", help=".npy file with probabilities of first model. When comparing SGD and SAM, expects SGD to be first model.")
    parser.add_argument("--number_model_1", type=int, default=5, help="How many different seeds have been trained of model 1")
    parser.add_argument("--type_model_1", type=str, default="SGD", help="The type of model 1 (here optimizer) that produced the confidences.")
    parser.add_argument("--model_2_path_file", type=str, default="", help=".npy file with probabilities of second model.")
    parser.add_argument("--number_model_2", type=int, default=5, help="How many different seeds have been trained of model 2")
    parser.add_argument("--type_model_2", type=str, default="SAM", help="The type of model 2 (here optimizer) that produced the confidences.")
    parser.add_argument("--plot", action=BooleanOptionalAction, default=False, type=bool, help="Whether results should be plotted qualitatively.")
    
    args = parser.parse_args()

    if args.model_1_path_file == "":
        raise Exception("Oops you did not provide a first model to evaluate!")
    if args.model_2_path_file == "":
        print("Oops you did not provide a second model to evaluate!")

    PATH = ROOT + '/experiment_results/cifar10H/probs/' # Adjust this path as needed

    # --- Load the data ---
    # Ensure these files exist in your PATH
    data_H = np.load(PATH + 'cifar10h-probs.npy')
    print(f"Loaded Human annotations shape: {data_H.shape}")
    assert_valid_distributions(data_H)


    model_SGD_preds = []
    model_paths = open(ROOT+args.model_1_path_file, "r")
    for model_path in model_paths.read().splitlines():
        model_path = model_path.strip()
        loaded_data = np.load(ROOT+model_path)
        model_SGD_preds.append(loaded_data)
        assert_valid_distributions(loaded_data)
    print(f"Loaded {len(model_SGD_preds)} models of type {args.type_model_1}. Shape of one: {model_SGD_preds[0].shape}")

    model_SAM_preds = []
    model_paths = open(ROOT+args.model_2_path_file, "r")
    for model_path in model_paths.read().splitlines():
        model_path = model_path.strip()
        loaded_data = np.load(ROOT+model_path)
        model_SAM_preds.append(loaded_data)
        assert_valid_distributions(loaded_data)
    print(f"Loaded {len(model_SAM_preds)} models of type {args.type_model_2}. Shape of one: {model_SAM_preds[0].shape}")

    num_images = data_H.shape[0]

    # This dictionary will store lists of N_images distances for each model *type*
    per_image_type_average_distances = defaultdict(lambda: {
        'KL': np.zeros(num_images),
        'JS': np.zeros(num_images),
        'Wasserstein': np.zeros(num_images),
        'L1': np.zeros(num_images),
        'L2': np.zeros(num_images)
    })

    all_model_data_for_calculation = [
        (args.type_model_1, model_SGD_preds, args.number_model_1),
        (args.type_model_2, model_SAM_preds, args.number_model_2)
    ]

    # Ensure no zeros in human_probs for KL/JS if not already handled by epsilon
    # A common practice is to add a small epsilon to avoid log(0)
    # (1e-10 is used in the calculations below)


    for model_type_name, model_list, num_models_in_type in all_model_data_for_calculation:
        print(f"\nCalculating per-image average metrics for {model_type_name} models...")
        for i, model_probs in enumerate(model_list):
            # KL Divergence
            # Ensure model_probs also has no zeros for log calculation
            kl_divs = np.sum(data_H * (np.log(data_H + 1e-10) - np.log(model_probs + 1e-10)), axis=1)
            per_image_type_average_distances[model_type_name]['KL'] += kl_divs

            # Jensen-Shannon Divergence
            M = 0.5 * (data_H + model_probs)
            js_divs = 0.5 * np.sum(data_H * (np.log(data_H + 1e-10) - np.log(M + 1e-10)), axis=1) + \
                    0.5 * np.sum(model_probs * (np.log(model_probs + 1e-10) - np.log(M + 1e-10)), axis=1)
            per_image_type_average_distances[model_type_name]['JS'] += js_divs

            # Wasserstein Distance (Earth Mover's Distance)
            classes = np.arange(data_H.shape[1])
            wasserstein_dists = np.array([
                wasserstein_distance(classes, classes, u_weights=p, v_weights=q)
                for p, q in zip(data_H, model_probs)
            ])
            per_image_type_average_distances[model_type_name]['Wasserstein'] += wasserstein_dists

            # L1 and L2 distances (Euclidean and Manhattan Distance)
            l1_dists = np.sum(np.abs(data_H - model_probs), axis=1)
            per_image_type_average_distances[model_type_name]['L1'] += l1_dists

            l2_dists = np.linalg.norm(data_H - model_probs, axis=1)
            per_image_type_average_distances[model_type_name]['L2'] += l2_dists

        # After summing up for all models of this type, divide by the number of models
        for metric in per_image_type_average_distances[model_type_name]:
            per_image_type_average_distances[model_type_name][metric] /= num_models_in_type

        print(f"Finished calculating per-image averages for {model_type_name}.")


    # --------------------------------------------------------------------
    # Statistical Analysis: Comparing Model Types (S vs A) on per-image averages
    # --------------------------------------------------------------------
    metrics = ['KL', 'JS', 'Wasserstein', 'L1', 'L2']
    results_summary_paired = defaultdict(dict)

    # Retrieve the per-image average distances for each model type
    type_SGD_per_image_avg = per_image_type_average_distances[args.type_model_1]
    type_SAM_per_image_avg = per_image_type_average_distances[args.type_model_2]

    alpha = 0.05
    raw_p_values = {}
    effect_sizes_to_correct = [] # List to store (p_value, test_type_string) tuples

    
    for metric in metrics:
        sgd_values = type_SGD_per_image_avg[metric]
        sam_values = type_SAM_per_image_avg[metric]

        print(f"\n--- Analyzing Metric (Paired): {metric} ---")
        print(f"Comparing {len(sgd_values)} paired image-level averages.")
        print(f"Mean {args.type_model_1}: {np.mean(sgd_values):.4f}, Mean {args.type_model_2}: {np.mean(sam_values):.4f}")

        differences = sgd_values - sam_values
        
        # Filter out zero differences for Wilcoxon (default behavior of scipy.stats.wilcoxon)
        non_zero_diffs = differences[differences != 0]
        n_non_zero_diffs = len(non_zero_diffs)
        
        # Shapiro-Wilk test for normality of differences
        # Note: Shapiro-Wilk may struggle with N > 5000. For large N, visual inspection of histogram
        # or other tests like D'Agostino-Pearson (scipy.stats.normaltest) might be considered,
        # but CLT generally ensures normality of the mean difference for large N.
        shapiro_diff_stat, shapiro_diff_p = np.nan, np.nan # Initialize as NaN
        is_normal_diff = False
        if n_non_zero_diffs > 0 and n_non_zero_diffs <= 5000: # Shapiro-Wilk limit
            shapiro_diff_stat, shapiro_diff_p = shapiro(non_zero_diffs)
            is_normal_diff = shapiro_diff_p > alpha
            print(f"Shapiro-Wilk test for non-zero differences (SGD-SAM): p={shapiro_diff_p:.4f} (Normal: {is_normal_diff})")
        elif n_non_zero_diffs > 5000:
            print("Shapiro-Wilk test not reliable for N > 5000. Assuming normality of differences for t-test based on CLT.")
            is_normal_diff = True # Assume normality due to large sample size (CLT)
        else:
            print("Not enough non-zero differences to perform Shapiro-Wilk test.")


        # Perform Wilcoxon Signed-Rank test (Non-parametric, paired)
        wilcoxon_result = wilcoxon(sam_values, sgd_values, alternative='less', method='auto') # method='auto' uses exact for small N, approx for large N
        wilcoxon_stat = wilcoxon_result.statistic
        wilcoxon_p_value = wilcoxon_result.pvalue
        wilcoxon_z_statistic = getattr(wilcoxon_result, 'zstatistic', None) # Get zstatistic if available (method='approx')

        raw_p_values[f'{metric}_wilcoxon'] = wilcoxon_p_value
        
        # Calculate Rank-Biserial Correlation (effect size for Wilcoxon)
        r_rb = rank_biserial_correlation(wilcoxon_z_statistic, n_non_zero_diffs) # Pass z-statistic for r_rb

        results_summary_paired[metric]['Wilcoxon_Signed_Rank'] = {
            'statistic': round(float(wilcoxon_stat), 6),
            'p_value_raw': format(wilcoxon_p_value, '.16e'),
            'effect_size_rank_biserial_r': round(float(r_rb) if not np.isnan(r_rb) else np.nan, 6),
            'mean_SGD': round(float(np.mean(sgd_values)), 6),
            'mean_SAM': round(float(np.mean(sam_values)), 6),
            'median_SGD': round(float(np.median(sgd_values)), 6),
            'median_SAM': round(float(np.median(sam_values)), 6),
            'mean_difference_SGD_minus_SAM': round(float(np.mean(differences)), 6),
            'median_difference_SGD_minus_SAM': round(float(np.median(differences)), 6)
        }
        print(f"Wilcoxon Signed-Rank Test (SGD > SAM): W={wilcoxon_stat:.4f}, p={wilcoxon_p_value:.4f}, Effect Size r={r_rb:.4f}")
        effect_sizes_to_correct.append((wilcoxon_p_value, f'{metric}_wilcoxon'))


        # Perform Paired T-test (Parametric)
        if is_normal_diff or n_non_zero_diffs > 5000: # Proceed if normal or N is large enough for CLT
            ttest_stat, ttest_p_value = ttest_rel(sam_values, sgd_values, alternative='less')
            raw_p_values[f'{metric}_ttest_paired'] = ttest_p_value
            
            # Calculate Cohen's d (effect size for paired t-test)
            cohens_d = cohens_d_paired(sgd_values, sam_values)
            
            results_summary_paired[metric]['Paired_T_Test'] = {
                'statistic': round(float(ttest_stat), 6),
                'p_value_raw': format(ttest_p_value, '.16e'),
                'effect_size_cohens_d': round(float(cohens_d), 6),
                'mean_SGD': round(float(np.mean(sgd_values)), 6),
                'mean_SAM': round(float(np.mean(sam_values)), 6),
                'mean_difference_SGD_minus_SAM': round(float(np.mean(differences)), 6)
            }
            print(f"Paired T-Test (SGD > SAM): t={ttest_stat:.4f}, p={ttest_p_value:.4f}, Effect Size d={cohens_d:.4f}")
            effect_sizes_to_correct.append((ttest_p_value, f'{metric}_ttest_paired'))
        else:
            results_summary_paired[metric]['Paired_T_Test'] = {
                'statistic': None,
                'p_value_raw': None,
                'effect_size_cohens_d': None,
                'note': "Paired T-test not performed due to non-normal distribution of differences (N <= 5000)."
            }
            print("Paired T-Test not performed due to non-normal distribution of differences (N <= 5000).")

    # --- Multiple Comparisons Correction ---
    print("\n--- Applying Multiple Comparisons Correction (FDR_BH) ---")
    all_raw_p_values_list = [p_val for p_val, _ in effect_sizes_to_correct]
    test_names_list = [name for _, name in effect_sizes_to_correct]

    if all_raw_p_values_list: # Ensure there are p-values to correct
        reject_fdr, p_values_fdr_corrected, _, _ = multipletests(all_raw_p_values_list, alpha=alpha, method='fdr_bh')
        print("--- correcrted p-values ", p_values_fdr_corrected, " ---")
        corrected_p_values_dict = dict(zip(test_names_list, p_values_fdr_corrected))
        reject_status_dict = dict(zip(test_names_list, reject_fdr))

        for metric in metrics:
            if f'{metric}_wilcoxon' in corrected_p_values_dict:
                wilcoxon_corrected_p = corrected_p_values_dict[f'{metric}_wilcoxon']
                wilcoxon_is_significant = reject_status_dict[f'{metric}_wilcoxon']
                results_summary_paired[metric]['Wilcoxon_Signed_Rank']['p_value_corrected_fdr'] = format(wilcoxon_corrected_p, '.16e')
                results_summary_paired[metric]['Wilcoxon_Signed_Rank']['significant_fdr'] = bool(wilcoxon_is_significant)
                print(f"  {metric} (Wilcoxon): Corrected p={wilcoxon_corrected_p:.4f}, Significant (FDR): {wilcoxon_is_significant}")

            if f'{metric}_ttest_paired' in corrected_p_values_dict:
                ttest_corrected_p = corrected_p_values_dict[f'{metric}_ttest_paired']
                ttest_is_significant = reject_status_dict[f'{metric}_ttest_paired']
                if ttest_corrected_p is not None:
                    results_summary_paired[metric]['Paired_T_Test']['p_value_corrected_fdr'] = format(ttest_corrected_p, '.16e')
                    results_summary_paired[metric]['Paired_T_Test']['significant_fdr'] = bool(ttest_is_significant)
                    print(f"  {metric} (Paired T-Test): Corrected p={ttest_corrected_p:.4f}, Significant (FDR): {ttest_is_significant}")
    else:
        print("No p-values to correct for multiple comparisons.")


    # --- Save the results to a file ---
    RESULT_PATH = PATH + "results/"
    import os
    os.makedirs(RESULT_PATH, exist_ok=True)
    output_path = RESULT_PATH + "per_image_" + args.save_file_name

    with open(output_path, 'w') as f:
        json.dump(results_summary_paired, f, indent=4)

    print(f"\nComprehensive paired analysis results saved to {output_path}")

    if args.plot:
        # --------------------------------------------------------------------
        # Plotting the Results
        # --------------------------------------------------------------------
        print("\n--- Generating Plots ---")

        plt.style.use('seaborn-v0_8-darkgrid') # Use a nice plot style

        # Loop through each metric to create specific plots
        for metric in metrics:
            sgd_values = type_SGD_per_image_avg[metric]
            sam_values = type_SAM_per_image_avg[metric]
            differences = sgd_values - sam_values

            # 1. Histogram of Differences (S - A)
            plt.figure(figsize=(10, 6))
            sns.histplot(differences, kde=True, bins=50, color='skyblue')
            plt.axvline(x=np.mean(differences), color='red', linestyle='--', label=f'Mean Difference: {np.mean(differences):.4f}')
            plt.axvline(x=np.median(differences), color='green', linestyle=':', label=f'Median Difference: {np.median(differences):.4f}')
            plt.title(f'Distribution of Differences: {args.type_model_1} - {args.type_model_2} ({metric} Distance per Image)')
            plt.xlabel(f'Difference in {metric} Distance ({args.type_model_1} - {args.type_model_2})')
            plt.ylabel('Frequency')
            plt.legend()
            plt.tight_layout()
            plt.savefig(RESULT_PATH + f'hist_diff_{metric}.png')
            plt.close()

            # 2. Paired Box Plot (or Violin Plot)
            plt.figure(figsize=(8, 6))
            data_for_plot = np.column_stack([sgd_values, sam_values])
            plt.boxplot(data_for_plot, labels=[args.type_model_1, args.type_model_2], patch_artist=True,
                        boxprops=dict(facecolor='lightblue', edgecolor='black'),
                        medianprops=dict(color='red'))
            plt.title(f'Comparison of {metric} Distances per Image')
            plt.ylabel(f'Average {metric} Distance to Human Data')
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.savefig(RESULT_PATH + f'boxplot_paired_{metric}.png')
            plt.close()

            # 3. Sample Paired Scatter Plot (too dense for 10000 points, take a random sample)
            sample_size = min(1000, num_images) # Plot up to 1000 random points
            if num_images > 0:
                sample_indices = np.random.choice(num_images, sample_size, replace=False)
                sgd_sample = sgd_values[sample_indices]
                sam_sample = sam_values[sample_indices]

                plt.figure(figsize=(8, 8))
                plt.scatter(sgd_sample, sam_sample, alpha=0.3, s=10, color='blue')
                
                # Add a line of equality (y=x)
                min_val = min(sgd_values.min(), sam_values.min())
                max_val = max(sgd_values.max(), sam_values.max())
                plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', label='Line of Equality')

                plt.title(f'Sample Paired Scatter Plot ({metric} Distance, N={sample_size})')
                plt.xlabel(f'{args.type_model_1} Average {metric} Distance')
                plt.ylabel(f'{args.type_model_2} Average {metric} Distance')
                plt.axis('equal') # Ensure equal scaling
                plt.legend()
                plt.tight_layout()
                plt.savefig(RESULT_PATH + f'scatter_paired_sample_{metric}.png')
                plt.close()

        print(f"Plots saved to {RESULT_PATH}")


def cohens_d_paired(x, y):
    """Calculates Cohen's d for paired samples."""
    diff = x - y
    return np.mean(diff) / np.std(diff, ddof=1) # ddof=1 for sample standard deviation

def rank_biserial_correlation(wilcoxon_statistic, n_non_zero_diffs):
    """Calculates rank-biserial correlation for Wilcoxon Signed-Rank test.
    
    Formula: r = Z / sqrt(N) where Z is the standardized test statistic and N is the number of
    non-zero differences.
    Scipy's wilcoxon returns the sum of ranks of positive differences (W+).
    The Z-statistic can be approximated from W+ for large N.
    A simpler, direct calculation of r_rb from W+ is also available for Wilcoxon, often as:
    r_rb = 2 * (W+ - n(n+1)/4) / (n*(n+1)/2)
    However, the Z/sqrt(N) is also commonly used, especially if scipy provides Z-score.
    Scipy's wilcoxon returns 'statistic' (min of positive/negative ranks sum for two-sided, sum of positive for one-sided)
    and 'pvalue'. For 'method=approx' it also provides 'zstatistic'. We will use the zstatistic if available.
    If not, we can approximate, or use a more direct rank-based formula.
    
    For now, we will use Z/sqrt(N) as it is widely cited and easier to implement directly from
    scipy's output if 'zstatistic' is provided or can be derived.
    If 'zstatistic' is not directly returned by `wilcoxon` for your scipy version/method,
    you might need to calculate it manually or use a dedicated library.
    Let's rely on the `zstatistic` if `method='approx'` is used in `wilcoxon`.
    
    A simpler formula often given is r_rb = 1 - (2 * Rneg / (n * (n + 1) / 2)) or 2 * (W / N(N+1)) - 1
    where W is the sum of positive ranks. However, let's use the Z-statistic approach which is more general.
    """
    if wilcoxon_statistic is None or n_non_zero_diffs == 0: # Check if the statistic is available (e.g. from approx method)
        return np.nan # Cannot calculate
        
    # We pass the z-statistic from wilcoxon results to this function.
    return wilcoxon_statistic / np.sqrt(n_non_zero_diffs)

def assert_valid_distributions(probs, tol=1e-4):
    # Check shape (should be 2D)
    assert probs.ndim == 2, "Distributions must be 2D (num_samples, num_classes)"

    # Check all values are ≥ 0
    if not np.all(probs >= 0):
        raise ValueError("Some probability values are negative")

    # Check rows sum to ~1
    row_sums = probs.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=tol):
        raise ValueError("Some rows do not sum to 1 (within tolerance)")


if __name__ == "__main__":
    main()
    print("All models evaluated!")