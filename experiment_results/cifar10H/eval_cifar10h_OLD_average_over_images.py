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
from collections import defaultdict
from argparse import ArgumentParser, BooleanOptionalAction
from utils.paths import *

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


    PATH = ROOT + '/experiment_results/cifar10H/probs/'
    # Load the data
    data_H = np.load(PATH+'cifar10h-probs.npy')
    data_1 = []
    resnets = []
    for i in range(args.number_model_1):
        loaded_data = np.load(PATH+args.model_1_path_file+"_"+str(i)+".npy")
        data_1.append(loaded_data)
        resnets.append(loaded_data)
    data_2 = []
    for i in range(args.number_model_2):
        loaded_data = np.load(PATH+args.model_2_path_file+"_"+str(i)+".npy")
        data_2.append(loaded_data)
        resnets.append(loaded_data)

    assert_valid_distributions(data_H)
    print(data_1[0].shape)
    assert_valid_distributions(data_1[0])
    assert_valid_distributions(data_2[0])

    # --------------------------------------------------------------------
    # Calculate metrics
    # --------------------------------------------------------------------
    model_names = [args.type_model_1]*args.number_model_1 + [args.type_model_2]*args.number_model_2
    print(model_names)

    results = defaultdict(lambda: {
        'KL': [],
        'JS': [],
        'Wasserstein': [],
        'L1': [],
        'L2': []
    })

    for name, model_probs in zip(model_names, resnets):
        print(name)
        # KL Divergence (batch-wise)
        kl_divs = np.sum(data_H * (np.log(data_H + 1e-10) - np.log(model_probs + 1e-10)), axis=1)
        print("kl_divs shape: ", kl_divs.shape)

        # Jensen-Shannon Divergence
        M = 0.5 * (data_H + model_probs)
        js_divs = 0.5 * np.sum(data_H * (np.log(data_H + 1e-10) - np.log(M + 1e-10)), axis=1) + \
                0.5 * np.sum(model_probs * (np.log(model_probs + 1e-10) - np.log(M + 1e-10)), axis=1)

        # Wasserstein Distance (requires loop)
        classes = np.arange(data_H.shape[1])
        wasserstein_dists = np.array([
            wasserstein_distance(classes, classes, u_weights=p, v_weights=q)
            for p, q in zip(data_H, model_probs)
        ])

        # L1 and L2 distances (vectorized)
        l1_dists = np.sum(np.abs(data_H - model_probs), axis=1)
        l2_dists = np.linalg.norm(data_H - model_probs, axis=1)


        # Store results
        results[name]['KL'].append(kl_divs.mean())
        results[name]['JS'].append(js_divs.mean())
        results[name]['Wasserstein'].append(wasserstein_dists.mean())
        results[name]['L1'].append(l1_dists.mean())
        results[name]['L2'].append(l2_dists.mean())



    # --------------------------------------------------------------------
    # Compare SGD and SAM with each other
    # --------------------------------------------------------------------
    # Metrics to test
    metrics = ['KL', 'JS', 'Wasserstein', 'L1', 'L2']
    p_values_wilcoxon = {}
    p_values_ttest = {}
    p_values_mw = {}
    p_values_ttest_ind = {}

    # Get the per-image metrics for both models
    resnet1 = results[args.type_model_1]
    resnet2 = results[args.type_model_2]

    for metric in metrics:
        resnet1_values = resnet1[metric]
        resnet2_values = resnet2[metric]
        print(f"\nTesting metric: {metric}")
        print(f"ResNet1 values: {resnet1_values}")
        print(f"ResNet2 values: {resnet2_values}")
        print(f"Shapes: {len(resnet1_values)}, {len(resnet2_values)}")

        # Perform normality tests
        _, p_value_1 = shapiro(resnet1_values)
        _, p_value_2 = shapiro(resnet2_values)

        # Store the normality test results
        normality_1 = p_value_1 > 0.05
        normality_2 = p_value_2 > 0.05

        print(f"Shapiro test p-value for ResNet1: {p_value_1} (Normal: {normality_1})")
        print(f"Shapiro test p-value for ResNet2: {p_value_2} (Normal: {normality_2})")

        # Perform the Wilcoxon signed-rank test (non-parametric)
        stat_wilcoxon, p_value_wilcoxon = wilcoxon(resnet1_values, resnet2_values, alternative="greater")
        p_values_wilcoxon[metric] = {
            'p_value': round(float(p_value_wilcoxon), 6),
            'statistic': round(float(stat_wilcoxon), 6),
            'significant': bool(p_value_wilcoxon < 0.05)  # Store boolean for significance
        }

        stat_mw, p_value_mw = mannwhitneyu(resnet1_values, resnet2_values, alternative='greater')
        p_values_mw[metric] = {
            'statistic': round(float(stat_mw), 6),
            'p_value_raw': round(float(p_value_mw), 6),
            'significant': bool(p_value_mw < 0.05)
        }


        # Perform t-test only if both distributions are normal
        if normality_1 and normality_2:
            stat_ttest, p_value_ttest = ttest_rel(resnet1_values, resnet2_values, alternative="greater")
            p_values_ttest[metric] = {
                'p_value': round(float(p_value_ttest), 6),
                'statistic': round(float(stat_ttest), 6),
                'significant': bool(p_value_ttest < 0.05)  # Store boolean for significance
            }
        else:
            p_values_ttest[metric] = {
                'p_value': None,
                'statistic': None,
                'significant': None
            }


        # Perform t-test only if both distributions are normal
        if normality_1 and normality_2:
            stat_ttest, p_value_ttest = ttest_ind(resnet1_values, resnet2_values, alternative="greater")
            p_values_ttest_ind[metric] = {
                'p_value': round(float(p_value_ttest), 6),
                'statistic': round(float(stat_ttest), 6),
                'significant': bool(p_value_ttest < 0.05)  # Store boolean for significance
            }
        else:
            p_values_ttest_ind[metric] = {
                'p_value': None,
                'statistic': None,
                'significant': None
            }

    # Save the results to a file
    RESULT_PATH = PATH + "results/"
    output_path = RESULT_PATH + args.save_file_name

    # Save both Wilcoxon and t-test results
    with open(output_path, 'w') as f:
        json.dump({
            'wilcoxon': p_values_wilcoxon,
            'ttest': p_values_ttest,
            'mannwhitneyu': p_values_mw,
            'ttest_ind': p_values_ttest_ind
        }, f, indent=4)

    print(f"Results saved to {output_path}")



    if args.plot:
        # --------------------------------------------------------------------
        # Plot the results
        # --------------------------------------------------------------------
        # Prepare data in long-form DataFrame for seaborn
        data = []

        for model_name in ['SGD', 'SAM']:
            for metric in metrics:
                for value in results[model_name][metric]:
                    data.append({'Metric': metric, 'Model': model_name, 'Value': value})

        df = pd.DataFrame(data)

        # Set up the plot
        plt.figure(figsize=(12, 6))
        sns.violinplot(data=df, x='Metric', y='Value', hue='Model', inner=None, palette='pastel', split=True)
        sns.boxplot(data=df, x='Metric', y='Value', hue='Model', whis=1.5, linewidth=1.0, dodge=True, palette='Set2')

        # Adjust the legend (remove duplicate due to double plots)
        handles, labels = plt.gca().get_legend_handles_labels()
        plt.legend(handles[:2], labels[:2], title='Model')

        plt.title('Distribution of Evaluation Metrics: Human vs ResNet1 vs ResNet2')
        plt.ylabel('Metric Value')
        plt.grid(axis='y', linestyle='--', alpha=0.5)
        plt.tight_layout()
        # Save the plot as PNG (or change to .pdf, etc. if needed)
        plt.savefig(RESULT_PATH + 'violin_boxplot_comparison.pdf', format='pdf')



        # --------------------------------------------------------------------
        # Plot the distribution
        # --------------------------------------------------------------------

        # Class labels
        labels = ["airplane", "automobile", "bird", "cat", "deer",
                "dog", "frog", "horse", "ship", "truck"]
        num_classes = len(labels)
        width = 0.25
        x = np.arange(num_classes)

        # Get hard labels for each source
        human_labels = np.argmax(data_H, axis=1)
        sgd_labels = np.argmax(data_SGD, axis=1)
        sam_labels = np.argmax(data_SAM, axis=1)

        # Plot settings
        fig, axes = plt.subplots(2, 5, figsize=(18, 8), sharey=True)
        axes = axes.flatten()

        for cls in range(num_classes):
            ax = axes[cls]

            # Get indices for images where human predicted this class
            indices = np.where(human_labels == cls)[0]

            if len(indices) == 0:
                continue

            avg_human = data_H[indices].mean(axis=0)
            avg_sgd = data_SGD[indices].mean(axis=0)
            avg_sam = data_SAM[indices].mean(axis=0)

            # Bar positions
            ax.bar(x - width, avg_human, width, label='Human', alpha=0.7)
            ax.bar(x, avg_sgd, width, label='SGD', alpha=0.7)
            ax.bar(x + width, avg_sam, width, label='SAM', alpha=0.7)

            ax.set_title(labels[cls])
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
            ax.set_ylim(0, 1)

        # Shared legend
        handles, legend_labels = ax.get_legend_handles_labels()
        fig.legend(handles, legend_labels, loc='upper center', ncol=3, fontsize=12)

        plt.suptitle('Average Predicted Distributions (per Human Label)', fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(RESULT_PATH + 'classwise_distribution_comparison.pdf', format='pdf')



if __name__ == "__main__":
    main()
    print("All models evaluated!")
