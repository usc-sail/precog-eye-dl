from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np, seaborn as sns
from sklearn.metrics import roc_curve, auc

colors = {0: "g", 1: "b", 2: "r"}


def make_boxplot(y_pred, y_true, group, position):
    plt.boxplot(
        [y_pred[i] for i in range(len(y_true)) if y_true[i] == group],
        positions=[position],
        widths=0.4,
        showfliers=False,
    )
    plt.text(
        position,
        np.median([y_pred[i] for i in range(len(y_true)) if y_true[i] == group]),
        f"{np.median([y_pred[i] for i in range(len(y_true)) if y_true[i] == group]):.2f}",
        ha="center",
        va="bottom",
        fontweight="bold",
    )
    for subject in range(len(y_true)):
        if y_true[subject] == group:
            plt.plot(
                [position + np.random.uniform(-0.1, 0.1)],
                [y_pred[subject]],
                f"{colors[group]}o",
                alpha=0.25,
            )


def make_violin(y_pred, y_true, group, position):
    import pandas as pd

    colors = {0: "#2bc776", 1: "#2b4ee6", 2: "#e62b3c"}

    df = {
        "y_pred": [y_pred[i] for i in range(len(y_true)) if y_true[i] == group],
        "group": [grp for i, grp in enumerate(y_true) if grp == group],
    }
    df = pd.DataFrame(df)
    sns.violinplot(
        x="group",
        y="y_pred",
        data=df,
        palette=[colors[group]],
        hue="group",
        density_norm="count",
        inner=None,
        linecolor="white",
        linewidth=0.5,
        ax=plt.gca(),
        legend=False,
        cut=0,
        common_norm=False,
        alpha=0.6,
    )
    sns.stripplot(
        data=df,
        x="group",
        y="y_pred",
        color="k",
        alpha=0.75,
        size=3,
        jitter=0.1,
        dodge=False,
    )


def make_plot(y_pred, y_true, group, position, mode):
    if mode == "violin":
        make_violin(y_pred, y_true, group, position)
    elif mode == "box":
        make_boxplot(y_pred, y_true, group, position)
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'violin' or 'box'.")


def boxplotting(y_pred, y_true, setting, test_setting, mode="violin"):
    plt.figure(figsize=(4, 4), dpi=200)

    if test_setting == "CvDS":
        make_plot(y_pred, y_true, 0, 0, mode)
        make_plot(y_pred, y_true, 1, 1, mode)
        make_plot(y_pred, y_true, 2, 2, mode)
        plt.xticks([0, 1, 2], ["C", "D", "S"])
        plt.ylabel("Probability of DS")

    elif test_setting == "CvS":
        make_plot(y_pred, y_true, 0, 0, mode)
        make_plot(y_pred, y_true, 2, 1, mode)
        plt.xticks([0, 1], ["C", "S"])
        plt.ylabel("Probability of S")

    elif test_setting == "CvD":
        make_plot(y_pred, y_true, 0, 0, mode)
        make_plot(y_pred, y_true, 1, 1, mode)
        plt.xticks([0, 1], ["C", "D"])
        plt.ylabel("Probability of D")

    elif test_setting == "DvS":
        make_plot(y_pred, y_true, 1, 0, mode)
        make_plot(y_pred, y_true, 2, 1, mode)
        plt.xticks([0, 1], ["D", "S"])
        plt.ylabel("Probability of S (D=0)")

    plt.title(f"Training Setting: {setting}")
    plt.xlabel("")
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.grid(axis="y", linestyle="--", alpha=0.5, linewidth=0.5)
    plt.tight_layout()

    plt.savefig(f"results/figures/{mode}_{setting}_on_{test_setting}.png", dpi=200)


def plot_roc_curve(y_pred_dict, y_true_dict, setting):
    """
    Plot the ROC curve for the given predictions and true labels.
    Args:
        y_pred_dict: Dict of [setting] > [subject] > [10 predictions].
        y_true_dict: Dict of true subject labels.
        setting (str): Setting for the plot title.
    Returns:
        ROC plot with 1 line per setting and STerror from 10 predictions.
    """

    plt.figure(figsize=(4, 4), dpi=200)
    plt.plot([0, 1], [0, 1], "k--", lw=0.75)
    colors = (
        ["#003f5c", "#148baf", "#33e0ff"]
        if len(y_pred_dict.keys()) == 3
        else (
            ["#3025a4", "#007beb", "#00b3b9", "#1cde46"]
            if len(y_pred_dict.keys()) == 4
            else ["#148baf"]
        )
    )

    mean_fpr, labels = np.linspace(0, 1, 100), []
    for i, (model, subject_preds) in enumerate(y_pred_dict.items()):
        tprs, aucs = [], []

        for seed in range(10):
            trues, preds = [], []
            for subject_id, preds_list in subject_preds.items():
                trues.append(y_true_dict[subject_id])
                preds.append(preds_list[seed])

            fpr, tpr, _ = roc_curve(trues, preds)
            interp_tpr = np.interp(mean_fpr, fpr, tpr)
            interp_tpr[0] = 0.0
            tprs.append(interp_tpr)
            aucs.append(auc(fpr, tpr))

        tprs = np.array(tprs)
        mean_tpr = np.mean(tprs, axis=0)
        stderr_tpr = np.std(tprs, axis=0) / np.sqrt(len(tprs))

        plt.plot(
            mean_fpr,
            mean_tpr,
            color=colors[i],
            label=f"{model} (AUC: {np.mean(aucs):.2f})",
        )
        plt.fill_between(
            mean_fpr,
            np.maximum(mean_tpr - stderr_tpr, 0),
            np.minimum(mean_tpr + stderr_tpr, 1),
            color=colors[i],
            alpha=0.2,
            edgecolor="none",
        )
        labels.append(f"{model} (AUC: {np.mean(aucs):.2f})")

    handles = []
    for color, label in zip(colors, labels):
        handles.append(Line2D([0], [0], color=color, lw=3, label=label))

    plt.title(f"Test Setting: {setting}")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")

    plt.legend(loc="lower right", handles=handles, fontsize=9)
    plt.grid(linestyle="--", alpha=0.5, linewidth=0.75)
    plt.tight_layout()
    plt.savefig(f"results/figures/roc_curve_{setting}.png", dpi=200)
