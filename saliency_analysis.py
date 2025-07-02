import matplotlib.pyplot as plt
import torch, os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def compute_integrated_gradients(model, x_input, class_idx=1, n_steps=50):
    """
    Compute Integrated Gradients for a batch of eye-tracking input.

    Args:
        model: your trained EyeTrackModel
        x_input: shape (1, 2, T, L, 2) - single sample
        class_idx: class of interest (e.g., 1 for depressed)
        n_steps: number of integration steps (higher = smoother)

    Returns:
        attribution: same shape as input (2, T, L, 2)
    """
    from captum.attr import IntegratedGradients

    model.eval()
    ig = IntegratedGradients(model)

    x_input = x_input.clone().detach().to("cuda")
    baseline = torch.zeros_like(x_input).to("cuda")

    # Compute IG
    attributions = ig.attribute(
        inputs=x_input, baselines=baseline, target=class_idx, n_steps=n_steps
    )
    attibution = attributions[0].detach()  # shape: (2, T, L, 2)
    return attibution


def compute_saliency(model, x_batch):
    x_batch = x_batch.clone().detach().requires_grad_(True)
    attribution = compute_integrated_gradients(model, x_batch)
    return attribution, [0, 0]  # (2, T, L, 2)


def analyze_classifier_weights(model):
    fc_weights = model.classifier[1].weight.data  # (128, seq_len * 4)
    seq_len = fc_weights.shape[1] // 4
    fc_weights = fc_weights.view(128, 4, seq_len)
    weights_abs = fc_weights.abs().mean(0)  # (4, T)

    return {
        "pos_x": weights_abs[0].mean().item(),
        "pos_y": weights_abs[1].mean().item(),
        "neg_x": weights_abs[2].mean().item(),
        "neg_y": weights_abs[3].mean().item(),
        "per_time": weights_abs.view(4, seq_len),
    }


def plot_weight_contributions(weight_dict, mode="read"):
    for k, v in weight_dict.items():
        if isinstance(v, torch.Tensor):
            weight_dict[k] = v.cpu().numpy()

    from scipy.ndimage import gaussian_filter1d

    weight_dict["per_time"][0] = gaussian_filter1d(weight_dict["per_time"][0], sigma=1)
    weight_dict["per_time"][1] = gaussian_filter1d(weight_dict["per_time"][1], sigma=1)
    weight_dict["per_time"][2] = gaussian_filter1d(weight_dict["per_time"][2], sigma=1)
    weight_dict["per_time"][3] = gaussian_filter1d(weight_dict["per_time"][3], sigma=1)

    if mode == "read":
        t = torch.linspace(-0.5, 0.9, weight_dict["per_time"].shape[1])
    else:
        t = torch.linspace(-1.2, 0, weight_dict["per_time"].shape[1])

    plt.figure(figsize=(6, 4), dpi=200)

    plt.subplot(1, 2, 1)
    plt.plot(t, weight_dict["per_time"][0], label="Positive", color="#1b9e77")
    plt.plot(t, weight_dict["per_time"][2], label="Negative", color="darkorchid")
    if mode == "read":
        plt.axvline(
            x=0,
            color="black",
            linestyle="--",
            linewidth=0.5,
        )

    plt.title("Horizontal (x) Contribution")
    plt.ylabel("FC-layer Weight (a.u.)")
    plt.xlabel("Time (s)")
    plt.ylim(0.016, 0.056)
    if mode == "read":
        plt.xlim(-0.55, 1.05)
    plt.legend(loc="upper left")

    plt.subplot(1, 2, 2)
    plt.plot(t, weight_dict["per_time"][1], label="Positive", color="#1b9e77")
    plt.plot(t, weight_dict["per_time"][3], label="Negative", color="darkorchid")
    if mode == "read":
        plt.axvline(
            x=0,
            color="black",
            linestyle="--",
            linewidth=0.5,
        )

    plt.title("Vertical (y) Contribution")
    plt.xlabel("Time (s)")
    plt.ylim(0.016, 0.056)
    if mode == "read":
        plt.xlim(-0.55, 1.05)
    plt.legend(loc="upper left")

    plt.tight_layout()
    plt.savefig(f"weights_per_time_{mode}.png")


def plot_saliency_heatmaps(saliency, name, mode="read", col="C"):
    assert mode in ["read", "resp"], "mode must be either 'read' or 'resp'"

    # saliency: (2, T, L, 2)
    this_sal = {}
    for b, branch in enumerate(["Positive", "Negative"]):
        for a, axis in enumerate(["X", "Y"]):
            # now (T, L) --> collapse over T
            this_sal[f"{branch}-{axis}"] = saliency[a, :, :, b].mean(axis=0)

    if mode == "read":
        t = torch.linspace(-0.5, 0.9, this_sal["Positive-X"].shape[0])
    else:
        t = torch.linspace(-1.2, 0, this_sal["Positive-X"].shape[0])

    color_duo = {
        "C": ["#51e37a", "#1b9e77"],
        "D": ["#80b1d3", "#2859a9"],
        "S": ["#de7a7f", "#a41515"],
    }[col]

    plt.figure(figsize=(6, 4), dpi=200)

    plt.subplot(1, 2, 1)
    plt.plot(t, this_sal["Positive-X"], label="Positive", color=color_duo[0])
    plt.plot(t, this_sal["Negative-X"], label="Negative", color=color_duo[1])

    if mode == "read":
        plt.axvline(
            x=0,
            color="black",
            linestyle="--",
            linewidth=0.5,
        )
    plt.axhline(y=0, color="black", linestyle="--", linewidth=0.5)

    plt.title("Horizontal (x) Saliency")
    plt.ylabel("Gradient Attribution (a.u.)")
    if mode == "read":
        plt.ylim(-0.0015, 0.0015)
        plt.xlim(-0.55, 1.05)
    else:
        plt.ylim(-0.0036, 0.001)
    plt.legend(loc="lower left")

    plt.subplot(1, 2, 2)
    plt.plot(t, this_sal["Positive-Y"], label="Positive", color=color_duo[0])
    plt.plot(t, this_sal["Negative-Y"], label="Negative", color=color_duo[1])

    if mode == "read":
        plt.axvline(
            x=0,
            color="black",
            linestyle="--",
            linewidth=0.5,
        )
    plt.axhline(y=0, color="black", linestyle="--", linewidth=0.5)

    plt.title("Vertical (y) Saliency")
    if mode == "read":
        plt.ylim(-0.0015, 0.0015)
        plt.xlim(-0.55, 1.05)
    else:
        plt.ylim(-0.0036, 0.001)
    plt.legend(loc="lower left")

    plt.tight_layout()
    plt.savefig(f"{name}_{mode}.png")
