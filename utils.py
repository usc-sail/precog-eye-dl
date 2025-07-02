import numpy as np, torch
from sklearn.metrics import recall_score


BAD_SUBS = [
    "013",
    "017",
    "020",
    "021",
    "025",
    "054",
    "055",
    "089",
    "094",
    "127",
    "134",
    "140",
    "146",
]

INELIGIBLE = ["016", "023", "036", "052", "076", "159"]


def find_class_weights(labels: list[int]) -> torch.Tensor:
    class_weights = np.unique(labels, return_counts=True)[1]
    class_weights = class_weights.sum() / class_weights
    return torch.tensor(class_weights / class_weights.sum()).float()


def groups_at_setting(setting, groups):
    if setting == "CvDS":
        # convert group 2 to 1
        groups = {k: 1 if v == 2 else v for k, v in groups.items()}
    elif setting == "CvS":
        # remove group 1 and convert group 2 to 1
        groups = {k: v for k, v in groups.items() if v != 1}
        groups = {k: 1 if v == 2 else 0 for k, v in groups.items()}
    elif setting == "CvD":
        # remove group 2
        groups = {k: v for k, v in groups.items() if v != 2}
    elif setting == "DvS":
        # remove group 0 and lower the others
        groups = {k: v for k, v in groups.items() if v != 0}
        groups = {k: v - 1 for k, v in groups.items()}
    elif setting == "CDvS":
        # convert group 1 to 0 and group 2 to 1
        groups = {k: 0 if v == 0 else v - 1 for k, v in groups.items()}
    else:
        raise ValueError("Invalid setting")
    return groups


def sns_or_spc(probas, labels, target_sns=None, target_spc=None):
    """
    Calculate sensitivity at a specific specificity or vice versa.
    """
    if target_sns is None and target_spc is None:
        raise ValueError("Either target_sns or target_spc must be provided.")
    elif target_sns is not None and target_spc is not None:
        raise ValueError("Only one of target_sns or target_spc should be provided.")

    probas = np.array(probas)
    labels = np.array(labels)

    if target_sns is not None:
        # threshold probabilities in ascending order
        threshold = 0.01
        while True:
            sns = recall_score(labels, probas >= threshold, pos_label=1)
            if sns <= target_sns:
                break
            threshold += 0.01

        return recall_score(labels, probas >= threshold, pos_label=0)
    else:
        # threshold probabilities in descending order
        threshold = 0.99
        while True:
            spc = recall_score(labels, probas >= threshold, pos_label=0)
            if spc <= target_spc:
                break
            threshold -= 0.01

        return recall_score(labels, probas >= threshold, pos_label=1)
