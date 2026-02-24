import lightning as L
from torch.utils.data import DataLoader
from .data import SliceDataset, TGVNDataTransform, collate_fn


class TGVNDataModule(L.LightningDataModule):
    def __init__(
        self,
        train_csv: str = "./csv_files/train.csv",
        val_csv: str = "./csv_files/val.csv",
        test_csv: str = "./csv_files/test.csv",
        train_batch_size: int = 1,
        val_batch_size: int = 1,
        test_batch_size: int = 1,
        workers: int = 4,
        buffer_size: int = 11,
        target_acceleration: int = 15,
        center_fraction: float = 0.5,
    ):
        super().__init__()
        self.train_csv = train_csv
        self.val_csv = val_csv
        self.test_csv = test_csv
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size
        self.workers = workers
        self.buffer_size = buffer_size
        self.target_acceleration = target_acceleration
        self.center_fraction = center_fraction
        self.train_transform = TGVNDataTransform(
            buffer_size=self.buffer_size,
            acceleration=self.target_acceleration,
            center_fraction=self.center_fraction,
            randomize_mask=True
        )
        self.val_transform = TGVNDataTransform(
            buffer_size=self.buffer_size,
            acceleration=self.target_acceleration,
            center_fraction=self.center_fraction,
            randomize_mask=False
        )
        self.test_transform = TGVNDataTransform(
            buffer_size=self.buffer_size,
            acceleration=self.target_acceleration,
            center_fraction=self.center_fraction,
            randomize_mask=False
        )

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_dataset = SliceDataset(
                self.train_csv, transform=self.train_transform
            )
            self.val_dataset = SliceDataset(
                self.val_csv, transform=self.val_transform
            )

        if stage == "test" or stage is None:
            self.test_dataset = SliceDataset(
                self.test_csv, transform=self.test_transform
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            shuffle=True,
            num_workers=self.workers,
            pin_memory=True,
            collate_fn=collate_fn,
            drop_last=True,  # Drop last incomplete batch
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=self.workers,
            pin_memory=True,
            collate_fn=collate_fn,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.test_batch_size,
            shuffle=False,
            num_workers=self.workers,
            pin_memory=True,
            collate_fn=collate_fn,
        )
