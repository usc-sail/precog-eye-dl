import os, numpy as np, torch
from torch.utils.data import Dataset
from tqdm import tqdm


class PRECOGEye(Dataset):
    def __init__(
        self,
        path: str,
        task: str,
        subjects: list[str] | None = ["067", "068"],
        groups: dict[str, int] = {"067": 0, "068": 1},
        limit: int = 200,
    ) -> None:
        """
        PyToprch dataset for PRECOG Eye Tracking data.
        Used for both preprocessing and loading the data.

        Parameters
        ----------
        path : str
            Path of the cached data.
        task : str
            Task to load data for, "EM" or "ST".
        subjects : list of str
            List of subjects to load data for.
            If None, all subjects are loaded.
        groups : dict of str:int
            Dictionary of subject to group mapping.
        limit : int
            Limit the number of trial sets to load (max is 200)
        """
        self.path = path
        self.task = task
        self.subjects = (
            [name[3:6] for name in os.listdir(self.path)] if not subjects else subjects
        )
        self.subjects = list(set(self.subjects))
        self.groups = groups
        self.limit = limit

        # load data and labels
        self.pos_data, self.neg_data = [], []
        self.labels, self.names = [], []
        for subject in tqdm(self.subjects):
            subject_path = os.path.join(self.path, f"sub{subject}")

            for cat in ["pos", "neg"]:
                # list all data in the subject folder
                eye_data = np.load(subject_path + f"_{cat}.npy")[: self.limit]

                # normalize to screen dimensions (1920, 1080)
                x = (eye_data[:, ::2] - 960) / 960
                y = (eye_data[:, 1::2] - 540) / 540
                eye_data = np.stack([x, y], axis=1)

                # categorize to positive or negative
                if cat == "pos":
                    self.pos_data.extend(eye_data)
                else:
                    self.neg_data.extend(eye_data)

            # load subject labels for each image
            subject_label = self.groups[subject]
            self.labels.extend([subject_label] * len(eye_data))
            self.names.extend([subject] * len(eye_data))

        self.pos_data = np.array(self.pos_data)
        self.neg_data = np.array(self.neg_data)

        self.labels = np.array(self.labels)
        self.names = np.array(self.names)

    def __len__(self) -> int:
        return len(self.pos_data)

    def __getitem__(self, idx: int) -> tuple[np.ndarray, int, str]:
        eye_d_pos = self.pos_data[idx]
        eye_d_neg = self.neg_data[idx]
        eye_d = np.stack([eye_d_pos, eye_d_neg], axis=-1)

        eye_d = torch.from_numpy(eye_d).float()
        return eye_d, self.labels[idx], self.names[idx]


if __name__ == "__main__":
    SRC = "/PATH/TO/eyelink-processed/input_30trials_resp/"
    dataset = PRECOGEye(SRC, task="ST", limit=200)
    print(dataset[0][0].shape, dataset[0][2], len(dataset))
