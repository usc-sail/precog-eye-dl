import yaml, json, argparse, os
import numpy as np, pandas as pd
from munch import munchify
from torch.utils.data import DataLoader
from torch.optim import AdamW, lr_scheduler
from sklearn.model_selection import StratifiedShuffleSplit

from model import EyeTrackModel
from dataset import PRECOGEye
from trainer import Trainer
from utils import groups_at_setting, find_class_weights, BAD_SUBS


if __name__ == "__main__":
    # Argument parser for optional overrides
    parser = argparse.ArgumentParser()
    parser.add_argument("--timing", type=str, default="resp", help="resp or read")
    parser.add_argument("--test_fold", type=str, help="0, 1, 2, 3, or 4")
    parser.add_argument("--name", type=str, help="Name override")
    parser.add_argument("--ntrials", type=int, default=30, help="Number of trials")
    parser.add_argument("--nlimit", type=int, help="Number of trial sets to load")
    parser.add_argument("--ndevice", type=int, help="Device to use: 0, 1, 2, or 3")
    args = parser.parse_args()

    # Load config file
    with open("config.yaml", "r") as f:
        cfg = munchify(yaml.safe_load(f))

    cfg.timing = args.timing if args.timing else cfg.timing
    cfg.test_fold = args.test_fold if args.test_fold else cfg.test_fold
    cfg.name = args.name if args.name != "." else ""
    cfg.ntrials = args.ntrials if args.ntrials else cfg.ntrials
    cfg.nlimit = args.nlimit if args.nlimit else cfg.nlimit
    cfg.ndevice = args.ndevice if args.ndevice else 1

    SRC = cfg.root + f"input_{cfg.ntrials}trials_{cfg.timing}{cfg.name}/"
    if "cuda" in cfg.device:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.ndevice)
    if cfg.ntrials != 30:
        cfg.name = f"_n{cfg.ntrials}"
    if cfg.nlimit != 200:
        cfg.name = f"_l{cfg.nlimit}"

    with open(cfg.root + "metadata/groups.json", "r") as f:
        all_groups = json.load(f)
        all_groups = {k[3:]: int(v) - 1 for k, v in all_groups.items()}
        all_groups = {
            k: v
            for k, v in all_groups.items()
            if os.path.exists(os.path.join(SRC, f"sub{k}_pos.npy"))
            and k not in BAD_SUBS
        }
        igroups = groups_at_setting(cfg.task, all_groups)

    test_path = cfg.root + f"subject_split_cv2/0{cfg.test_fold}_test_info.csv"
    test_subs = pd.read_csv(test_path)
    test_subs["subID"] = test_subs["subID"].apply(lambda x: str(x)[3:])

    tgroups = {}
    for subject in test_subs["subID"].values:
        if subject not in all_groups:
            continue
        elif subject in igroups:
            tgroups[subject] = igroups[subject]
        else:  # subject is from an excluded group
            if cfg.task == "CvS":
                assert all_groups[subject] == 1
                tgroups[subject] = 1  # from group D
            elif cfg.task == "DvS":
                assert all_groups[subject] == 0
                tgroups[subject] = 0  # from group C
            elif cfg.task == "CvD":
                assert all_groups[subject] == 2
                tgroups[subject] = 1  # from group S
            else:
                raise ValueError("False group assignment")
    igroups = {k: v for k, v in igroups.items() if k not in test_subs["subID"].values}

    subs = list(igroups.keys())
    groups = list(igroups.values())
    skf = StratifiedShuffleSplit(
        n_splits=cfg.n_splits,
        test_size=cfg.test_size,
    )

    all_proba = {s: [] for s in tgroups.keys()}
    for i, (train_index, test_index) in enumerate(skf.split(subs, groups)):
        print(f"\nSeed {i+1}/{cfg.n_splits}")
        tr_sub, val_sub = np.array(subs)[train_index], np.array(subs)[test_index]
        tr_grp, val_grp = np.array(groups)[train_index], np.array(groups)[test_index]

        # permute the training labels
        if cfg.permute:
            np.random.seed(i)
            np.random.shuffle(tr_grp)

        print("Loading train dataset...")
        tr_sub = [str(sub).zfill(3) for sub in tr_sub]
        tr_grp = {k: v for k, v in zip(tr_sub, tr_grp)}
        train_dataset = PRECOGEye(
            path=SRC,
            task="ST",
            subjects=tr_sub,
            groups=tr_grp,
            limit=cfg.nlimit,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=cfg.batch_size,
            shuffle=cfg.shuffle,
            num_workers=cfg.num_workers,
            pin_memory=True,
        )

        print("Loading validation dataset...")
        val_sub = [str(sub).zfill(3) for sub in val_sub]
        val_grp = {k: v for k, v in zip(val_sub, val_grp)}
        val_dataset = PRECOGEye(
            path=SRC,
            task="ST",
            subjects=val_sub,
            groups=val_grp,
            limit=cfg.nlimit,
        )
        valid_loader = DataLoader(
            val_dataset,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=True,
        )

        # Define the model
        model = EyeTrackModel(
            in_channels=2,
            base_filters=32,
            seq_len=train_dataset[0][0].shape[-2],
            n_classes=cfg.n_classes,
        )
        model.to(cfg.device)
        # model = nn.DataParallel(model)

        # Define optimizer, scheduler, loss function
        optimizer = AdamW(model.parameters(), lr=cfg.lr, eps=1e-8, weight_decay=1e-3)
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.1, patience=cfg.patience - 2
        )

        # Define the trainer
        trainer = Trainer(
            device=cfg.device,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            early_stopping=cfg.early_stopping,
            checkpoint=f"results/ckpt/{cfg.task}{cfg.name}_model_f{cfg.test_fold}.pth",
            patience=cfg.patience,
            epochs=cfg.max_epochs,
            class_weights=find_class_weights(train_dataset.labels),
        )
        print("Now training...\n")
        trainer.perform_training(train_loader, valid_loader)

        print("\nLoading test dataset...")
        test_sub = list(tgroups.keys())
        test_grp = {k: v for k, v in tgroups.items()}
        test_dataset = PRECOGEye(
            SRC,
            task="ST",
            subjects=test_sub,
            groups=test_grp,
            limit=cfg.nlimit,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=True,
        )

        print("Now testing...\n")
        proba_dict = trainer.perform_inference(test_loader)
        for sb in proba_dict:
            all_proba[sb].append(proba_dict[sb])

    # Probability score analysis
    is_rand = "_rand" if cfg.permute else ""
    os.makedirs(f"results/{cfg.task}{cfg.name}", exist_ok=True)
    with open(
        f"results/{cfg.task}{cfg.name}/{cfg.timing}_{cfg.test_fold}{is_rand}.json", "w"
    ) as f:
        json.dump(all_proba, f, indent=4)
