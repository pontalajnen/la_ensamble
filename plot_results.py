import json
import glob
import os
import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = "experiment_results/table_metrics"
OUTPUT = "experiment_results/results_plot.png"

# Friendly display names keyed by filename stem
NAME_MAP = {
    "normal":   "Normal",
    "ensemble": "Ensemble",
    "packed":   "Packed",
    "sam":      "SAM",
    "sgld":     "SGLD",
}

METRICS = {
    "Accuracy (ID)":    ("clean_accuracy",  True,  lambda v: v * 100),
    "Accuracy (Shift)": ("SHIFT ACCURACY",  True,  lambda v: v * 100),
    "ECE (ID)":         ("ECE",             False, lambda v: v),
    "ECE (Shift)":      ("SHIFT ECE",       False, lambda v: v),
    "NLL (ID)":         ("nll",             False, lambda v: v),
    "OOD AUROC":        ("OOD AUROC",       True,  lambda v: v * 100),
}
# (metric_key, higher_is_better, transform)


def load_results():
    results = {}
    for path in sorted(glob.glob(os.path.join(RESULTS_DIR, "*.json"))):
        stem = os.path.splitext(os.path.basename(path))[0]
        # strip leading "resnet20_cifar10_" or similar prefix to get method name
        parts = stem.rsplit("_", 1)
        method = parts[-1]
        label = NAME_MAP.get(method, method)
        with open(path) as f:
            data = json.load(f)
        # each file has exactly one model key
        metrics = next(iter(data.values()))
        results[label] = metrics
    return results


def plot(results):
    labels = list(results.keys())
    n_metrics = len(METRICS)
    fig, axes = plt.subplots(1, n_metrics, figsize=(3.5 * n_metrics, 4.5))
    fig.suptitle("Model comparison — ResNet20 / CIFAR-10", fontsize=13, y=1.02)

    colors = plt.cm.tab10(np.linspace(0, 0.5, len(labels)))

    for ax, (title, (key, higher_is_better, transform)) in zip(axes, METRICS.items()):
        values = []
        for label in labels:
            v = results[label].get(key)
            values.append(transform(v) if v is not None else float("nan"))

        bars = ax.bar(labels, values, color=colors, width=0.6)
        ax.set_title(title, fontsize=10)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
        ax.yaxis.set_tick_params(labelsize=8)

        # annotate each bar with its value
        for bar, v in zip(bars, values):
            if not np.isnan(v):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + ax.get_ylim()[1] * 0.01,
                    f"{v:.2f}",
                    ha="center", va="bottom", fontsize=7,
                )

        arrow = "↑ better" if higher_is_better else "↓ better"
        ax.set_xlabel(arrow, fontsize=8, color="gray")

    plt.tight_layout()
    plt.savefig(OUTPUT, dpi=150, bbox_inches="tight")
    print(f"Saved to {OUTPUT}")
    plt.show()


if __name__ == "__main__":
    results = load_results()
    print(f"Loaded {len(results)} models: {list(results.keys())}")
    plot(results)
