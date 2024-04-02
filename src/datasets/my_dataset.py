import pandas as pd
import torchaudio
from torch.utils.data import Dataset
import os
from pathlib import Path
from src.datasets.base_dataset import SimpleAudioFakeDataset


class MyWAVDataset(SimpleAudioFakeDataset):
    def __init__(
        self,
        path,
        subset="train",
        transform=None,
        seed=None,
        partition_ratio=(0.7, 0.15),
        # split_strategy="random",
    ):
        super().__init__(subset=subset, transform=transform)
        self.path = path
        self.read_samples()
        self.partition_ratio = partition_ratio
        self.seed = seed

    def read_samples(self):
        path = Path(self.path)
        meta_path = path / "meta.csv"

        self.samples = pd.read_csv(meta_path)
        self.samples["path"] = self.samples["file"].apply(lambda n: str(path / n))
        self.samples["file"] = self.samples["file"].apply(lambda n: Path(n).stem)
        # self.samples["label"] = self.samples["label"].map({"bona-fide": "bonafide", "spoof": "spoof"})
        self.samples["attack_type"] = self.samples["label"].map(
            {"bonafide": "-", "spoof": "X"}
        )
        self.samples.rename(
            # columns={"file": "sample_name", "speaker": "user_id"}, inplace=True
            columns={"file": "sample_name"},
            inplace=True,
        )


if __name__ == "__main__":
    dataset = MyWAVDataset(
        path="../datasets/my_data",
        subset="test",
        seed=242,
        # split_strategy="per_speaker",
    )

    print(f"Length of Dataset : {len(dataset)}")
    print(dataset[5])  # audio (list) , sampling rate, label 0 (spoof), 1(bonafide)

    # print(len(dataset.samples["user_id"].unique()))
    # print(dataset.samples["user_id"].unique())


"""
    def __init__(
        self, path, subset="train", transform=None, seed=None, amount_to_use=None
    ):

        Initialize the dataset.

        Args:
            path (str): Path to the directory containing WAV files and meta.csv.
            subset (str): Which subset of the data to use ('train', 'test', 'val').
            transform (callable, optional): Optional transform to be applied on a sample.
            seed (int, optional): Random seed for shuffling the data.
            amount_to_use (int, optional): Limit the dataset to this number of files.

        self.path = Path(path)
        self.transform = transform
        self.subset = subset
        self.seed = seed
        self.amount_to_use = amount_to_use
        self.samples = self.read_samples()
"""
