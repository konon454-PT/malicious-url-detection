import logging
import hydra
import torch
import pandas as pd
import numpy as np

from TLModelCode import lightning_model, data_prepare, data_module
from omegaconf import DictConfig, OmegaConf


@hydra.main(config_path="conf", config_name="infer")
def infer(cfg: DictConfig) -> None:
    logger = logging.getLogger(__name__)
    logger.info(f"Inference with the following config:\n{OmegaConf.to_yaml(cfg)}")

    cur_path = "~/PycharmProjects/malicious-url-detection/"
    logger.info(f"Working directory: {cur_path}")

    checkpoint_path = cur_path + "./checkpoints/lightning_logs/version_55/checkpoints/epoch=0-step=1000.ckpt"
    model = lightning_model.CustomModel.load_from_checkpoint(checkpoint_path)
    logger.info(f"Loaded checkpoint: {checkpoint_path}")

    dataset_train_path = cur_path + "./dataset/train_dataset.csv"
    phishing_data = pd.read_csv(dataset_train_path, index_col=False, nrows=100000)
    df = pd.DataFrame(phishing_data)
    dataset_train = data_prepare.CustomTorchDataset(df)

    input_strings = ['example.com']
    df_test = pd.DataFrame({"url": input_strings, "target": [0 for i in range(len(input_strings))]})
    cleaned_df_test = data_prepare.clean_data(df_test)
    urls = cleaned_df_test["url"].values
    cleaned_df_test["text_seq"] = [torch.tensor(data_prepare.text_pipeline(url, dataset_train.vocab)) for url in urls]
    cleaned_df_test.loc[len(cleaned_df_test.index)] = ["terminator", 0, torch.tensor(np.arange(247))]
    pad_seq = torch.nn.utils.rnn.pad_sequence(cleaned_df_test["text_seq"].values, batch_first=True, padding_value=0.0)
    for i in range(len(input_strings)):
        logger.info(f"Input: {urls[i]}\nOutput: {model(pad_seq[i])}")


if __name__ == "__main__":
    infer()


