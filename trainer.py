import numpy as np, torch
from tqdm import tqdm
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, roc_auc_score, balanced_accuracy_score


class Trainer:
    def __init__(
        self,
        device: str,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        early_stopping: bool,
        checkpoint: str,
        patience: int,
        epochs: int,
        class_weights: torch.Tensor,
    ) -> None:
        self.device = device
        self.model = model
        self.model.to(self.device)
        self.checkpoint = checkpoint
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.class_weights = class_weights.to(self.device)
        self.criterion = CrossEntropyLoss(weight=self.class_weights)
        self.early_stopping = early_stopping
        self.patience = patience
        self.epochs = epochs

    def perform_training(
        self, train_loader: DataLoader, valid_loader: DataLoader
    ) -> None:
        best_metric = 100
        patience_counter = 0
        for ep in range(self.epochs):
            self.train(train_loader, ep)
            metric_dict, _ = self.validate(valid_loader, ep)
            for metric in metric_dict:
                print(f"\tvalid {metric}: {metric_dict[metric]:.7f}")
            print()
            if metric_dict["CE Loss"] < best_metric:
                best_metric = metric_dict["CE Loss"]
                patience_counter = 0
                self.save(self.checkpoint)
            else:
                patience_counter += 1
                if self.early_stopping and patience_counter == self.patience:
                    print("Early stopping now.")
                    break
            self.scheduler.step(metric_dict["CE Loss"])

    def perform_inference(self, data_loader: DataLoader) -> dict[str, float]:
        self.load(self.checkpoint)
        self.metric_dict, proba_dict = self.validate(data_loader)
        for metric in self.metric_dict:
            print(f"\ttest {metric}: {self.metric_dict[metric]:.7f}")
        return proba_dict

    def train(self, data_loader: DataLoader, epoch) -> float:
        self.model.train()
        progress_bar = tqdm(data_loader, desc=f"Epoch {epoch+1}", leave=False)

        total_loss = 0
        for getitem in progress_bar:
            samples_eye, labels, _ = getitem

            samples_eye = samples_eye.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(samples_eye).squeeze()

            loss = self.criterion(outputs.softmax(1), labels.long())
            total_loss += loss.item()
            loss.backward()
            self.optimizer.step()

            progress_bar.set_postfix({"loss": loss.item()})

    def validate(
        self, data_loader: DataLoader, epoch: int | None = None
    ) -> tuple[dict[str, float], dict[str, float]]:
        self.model.eval()
        disp = epoch + 1 if epoch is not None else "TEST"
        progress_bar = tqdm(data_loader, desc=f"Epoch {disp}", leave=False)

        all_pred, all_true, all_name, all_loss = [], [], [], []
        with torch.no_grad():
            for getitem in progress_bar:
                samples_eye, labels, subjects = getitem

                samples_eye = samples_eye.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(samples_eye).squeeze()
                loss = self.criterion(outputs.softmax(1), labels.long())

                all_pred.extend(outputs.cpu().numpy())
                all_true.extend(labels.cpu().numpy())
                all_name.extend(subjects)
                all_loss.append(loss.item())

        all_true = np.array(all_true).reshape(-1)
        all_pred = np.argmax(all_pred, axis=1)

        # aggregate per subject
        agg_pred, agg_true = [], []
        subject_proba_dict = {}
        for subject in np.unique(all_name):
            idx = np.where(np.array(all_name) == subject)[0]
            subject_proba_dict[subject] = all_pred[idx].mean()
            agg_pred.append(all_pred[idx].mean() > 0.5)
            agg_true.append(all_true[idx][0])

        agg_pred = [int(n) for n in agg_pred]
        agg_true = [int(n) for n in agg_true]
        probas = [subject_proba_dict[subject] for subject in np.unique(all_name)]

        return {
            "CE Loss": np.mean(all_loss),
            "ROC-AUC": roc_auc_score(agg_true, probas),
            "F1 Macro": f1_score(agg_true, agg_pred, average="macro"),
            "BAC": balanced_accuracy_score(agg_true, agg_pred),
        }, subject_proba_dict

    def save(self, file_path: str = "./") -> None:
        torch.save(self.model.state_dict(), file_path)

    def load(self, file_path: str = "./") -> None:
        self.model.load_state_dict(torch.load(file_path))
        self.model.to(self.device)
