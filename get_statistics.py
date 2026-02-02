import json, numpy as np
from sklearn.metrics import (
    roc_auc_score,
    balanced_accuracy_score,
    f1_score,
    recall_score,
    precision_score,
    brier_score_loss,
)
from utils import groups_at_setting, sns_or_spc
from plot_helper import boxplotting, plot_roc_curve

# lock seed for reproducibility
np.random.seed(79911092)

import argparse

parser = argparse.ArgumentParser(description="Model statistics")
parser.add_argument("--train_setting", type=str, required=True)
parser.add_argument("--test_setting", type=str, required=True)
parser.add_argument("--timing", type=str, required=True, choices=["resp", "read"])

args = parser.parse_args()


timing = args.timing
folds = [0, 1, 2, 3, 4]  # for cross-validation
n_bootstrap = 1000
train_setting, test_setting = args.train_setting, args.test_setting
groups_path = "/PATH/TO/eyelink-processed/metadata/groups.json"

data = {}
for fold_num in folds:
    results_path = f"results/{train_setting}/{timing}_{fold_num}.json"
    with open(results_path) as f:
        fold_data = json.load(f)
        data.update(fold_data)

with open(groups_path) as f:
    groups = json.load(f)
    groups = {k[3:]: int(v) - 1 for k, v in groups.items()}
    groups = groups_at_setting(test_setting, groups)

intersections = set(data.keys()).intersection(groups.keys())
data = {k: v for k, v in data.items() if k in intersections}
groups = {k: v for k, v in groups.items() if k in intersections}

# Average across seeds per subject to remove pseudo-replication
data_avg = {subject: np.mean(preds) for subject, preds in data.items()}
subjects_list = sorted(data_avg.keys())  # Ensure consistent ordering
y_true = [groups[subject] for subject in subjects_list]
y_pred = [data_avg[subject] for subject in subjects_list]

# Compute metrics on averaged predictions
roc_auc = roc_auc_score(y_true, y_pred)
y_pred_b = [1 if x > 0.5 else 0 for x in y_pred]
bac = balanced_accuracy_score(y_true, y_pred_b)
f1 = f1_score(y_true, y_pred_b, average="macro")

print(f"\nCross-validation scores [{len(data_avg)} subjects]:\n")
print(f"ROC-AUC: {roc_auc:.3f}")
print(f"BAC: {bac:.3f}")
print(f"F1 Macro: {f1:.3f}")

# use initial groups
with open(groups_path) as f:
    igroups = json.load(f)
    igroups = {k[3:]: v for k, v in igroups.items() if k[3:] in intersections}
    iy_true = [int(igroups[k]) - 1 for k in data_avg.keys()]

# plot with averaged scores
y_pred_plot = [data_avg[subject] for subject in data_avg]
boxplotting(y_pred_plot, iy_true, train_setting, test_setting, mode="violin")

# plot ROC curve
# save {train_setting: data_avg} to use later
with open(f"results/roc_{train_setting}2{test_setting}_{timing}.json", "w") as f:
    json.dump({train_setting: data_avg}, f, indent=4)

# load settings: CvDS, CvD, CvS, DvS
y_pred_dict = {}
for stting in ["CvDS", "CvD", "CvS", "DvS"]:
    with open(f"results/roc_{stting}2{test_setting}_{timing}.json") as f:
        y_pred_dict.update(json.load(f))
y_true_dict = {k: groups[k] for k in data_avg}
plot_roc_curve(y_pred_dict, y_true_dict, test_setting)

# confidence intervals by bootstrap resampling on subjects (not seeds)
n_subjects = len(subjects_list)
bootstrap_scores = {
    "ROC-AUC": [],
    "BAC": [],
    "F1 Macro": [],
    "Sensitivity": [],
    "Specificity": [],
    "Sensitivity at 70% Specificity": [],
    "Specificity at 70% Sensitivity": [],
    "Precision": [],
    "Negative Predictive Value": [],
    "Brier Score": [],
}

if "C" in test_setting:
    bootstrap_scores["C_proba"] = []
if "D" in test_setting:
    bootstrap_scores["D_proba"] = []
if "S" in test_setting:
    bootstrap_scores["S_proba"] = []

for _ in range(n_bootstrap):
    # Bootstrap resample subjects (not seed predictions)
    bootstrap_subjects = np.random.choice(subjects_list, n_subjects, replace=True)
    
    # Build bootstrapped y_true and y_pred from averaged predictions
    boot_y_true = [groups[subject] for subject in bootstrap_subjects]
    boot_y_pred = [data_avg[subject] for subject in bootstrap_subjects]
    boot_iy_true = [int(igroups[subject]) - 1 for subject in bootstrap_subjects]

    roc_auc = roc_auc_score(boot_y_true, boot_y_pred)
    sns_at_70 = sns_or_spc(boot_y_pred, boot_y_true, target_spc=0.7)
    spc_at_70 = sns_or_spc(boot_y_pred, boot_y_true, target_sns=0.7)

    boot_y_pred_b = [1 if x > 0.5 else 0 for x in boot_y_pred]
    bac = balanced_accuracy_score(boot_y_true, boot_y_pred_b)
    f1 = f1_score(boot_y_true, boot_y_pred_b, average="macro")
    sensitivity = recall_score(boot_y_true, boot_y_pred_b)
    specificity = recall_score(boot_y_true, boot_y_pred_b, pos_label=0)
    precision = precision_score(boot_y_true, boot_y_pred_b)
    npv = precision_score(boot_y_true, boot_y_pred_b, pos_label=0)
    brier = brier_score_loss(boot_y_true, boot_y_pred)

    # Compute group-specific probabilities using initial groups
    if "C" in test_setting:
        c_vals = [boot_y_pred[i] for i, y in enumerate(boot_iy_true) if y == 0]
        bootstrap_scores["C_proba"].append(np.mean(c_vals))
    if "D" in test_setting:
        d_vals = [boot_y_pred[i] for i, y in enumerate(boot_iy_true) if y == 1]
        bootstrap_scores["D_proba"].append(np.mean(d_vals))
    if "S" in test_setting:
        s_vals = [boot_y_pred[i] for i, y in enumerate(boot_iy_true) if y == 2]
        bootstrap_scores["S_proba"].append(np.mean(s_vals))

    bootstrap_scores["ROC-AUC"].append(roc_auc)
    bootstrap_scores["BAC"].append(bac)
    bootstrap_scores["F1 Macro"].append(f1)
    bootstrap_scores["Sensitivity"].append(sensitivity)
    bootstrap_scores["Specificity"].append(specificity)
    bootstrap_scores["Sensitivity at 70% Specificity"].append(sns_at_70)
    bootstrap_scores["Specificity at 70% Sensitivity"].append(spc_at_70)
    bootstrap_scores["Precision"].append(precision)
    bootstrap_scores["Negative Predictive Value"].append(npv)
    bootstrap_scores["Brier Score"].append(brier)

# report 95% confidence intervals
print(f"\nBootstrap scores [{n_subjects} subjects]:\n")
for k, v in bootstrap_scores.items():
    if k not in [
        "ROC-AUC",
        "Sensitivity",
        "Specificity",
        "Precision",
        "Negative Predictive Value",
    ]:
        continue
    ci = np.nanpercentile(v, [2.5, 97.5])
    print(f"{k}: {np.mean(v):.3f} [{ci[0]:.3f}, {ci[1]:.3f}]")

# empirical null distribution
exceed_min, exceed_mean = 0, 0
ref_score = bootstrap_scores["ROC-AUC"]
thresh_min = np.nanpercentile(ref_score, 2.5)
thresh_mean = np.mean(ref_score)

perm_data = {}
for fold_num in folds:
    results_path = f"results/{train_setting}/{timing}_{fold_num}_rand.json"
    with open(results_path) as f:
        fold_data = json.load(f)
        perm_data.update(fold_data)

# Average permuted data across seeds per subject
perm_data = {k: v for k, v in perm_data.items() if k in intersections}
perm_data_avg = {subject: np.mean(preds) for subject, preds in perm_data.items()}

# Ensure permutation data contains same subjects as real data
perm_subjects = sorted([s for s in subjects_list if s in perm_data_avg])
if len(perm_subjects) != n_subjects:
    print(f"Warning: Permutation data has {len(perm_subjects)} subjects vs {n_subjects} in real data")

temp = []
for _ in range(n_bootstrap):
    # Bootstrap resample subjects for permutation test
    bootstrap_subjects = np.random.choice(perm_subjects, len(perm_subjects), replace=True)
    boot_y_true = [groups[subject] for subject in bootstrap_subjects]
    boot_y_pred = [perm_data_avg[subject] for subject in bootstrap_subjects]
    
    roc_auc = roc_auc_score(boot_y_true, boot_y_pred)
    temp.append(roc_auc)
    if roc_auc > thresh_min:
        exceed_min += 1
    if roc_auc > thresh_mean:
        exceed_mean += 1

p_value_min = (exceed_min + 1) / (n_bootstrap + 1)
p_value_mean = (exceed_mean + 1) / (n_bootstrap + 1)
print(f"Empirical null p-value (min): {p_value_min:.3f}\n")
print(f"Empirical null p-value (mean): {p_value_mean:.3f}")
