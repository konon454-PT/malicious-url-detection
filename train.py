import logging
import hydra
import pytorch_lightning as pl
import pandas as pd

from TLModelCode import data_module, data_prepare, lightning_model

from omegaconf import DictConfig, OmegaConf


@hydra.main(config_path="conf", config_name="train")
def train(cfg: DictConfig) -> None:
    logger = logging.getLogger(__name__)
    logger.info(f"Training with the following config:\n{OmegaConf.to_yaml(cfg)}")

    cur_path = "~/PycharmProjects/malicious-url-detection/"
    logger.info(f"Working directory: {cur_path}")

    dataset_train_path = cur_path + "./dataset/train_dataset.csv"
    phishing_data = pd.read_csv(dataset_train_path, index_col=False, nrows=100000)
    df = pd.DataFrame(phishing_data)
    dataset_train = data_prepare.CustomTorchDataset(df)
    dm_train = data_module.DatasetDataModuleTrain(dataset_train)
    logger.info(f"Loaded dataset: {dataset_train_path}")

    model = lightning_model.CustomModel()

    trainer = pl.Trainer(limit_train_batches=1000, max_epochs=1, default_root_dir=cur_path+"./checkpoints/")
    trainer.fit(model=model, train_dataloaders=dm_train.train_dataloader())


if __name__ == "__main__":
    train()
