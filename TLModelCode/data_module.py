import pytorch_lightning as pl
from torch.utils.data import DataLoader


class DatasetDataModuleTrain(pl.LightningDataModule):
    def __init__(self, dataset_train, batch_size: int = 1):
        super().__init__()
        self.dataset_train = dataset_train
        self.batch_size = batch_size

    def train_dataloader(self):
        return DataLoader(self.dataset_train, batch_size=self.batch_size, num_workers=11)


class DatasetDataModuleTest(pl.LightningDataModule):
    def __init__(self, dataset_test, batch_size: int = 16):
        super().__init__()
        self.dataset_test = dataset_test
        self.batch_size = batch_size

    def test_dataloader(self):
        return DataLoader(self.dataset_test, batch_size=self.batch_size, num_workers=11)