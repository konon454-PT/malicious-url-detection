import torch
from torch import optim, nn
import pytorch_lightning as pl
import numpy as np

model_seq = nn.Sequential(nn.Embedding(128000, 256),
                          nn.Conv1d(247, 256, 3),
                          nn.ReLU(),
                          nn.MaxPool1d(2),
                          nn.Flatten(),
                          nn.Linear(32512, 64),
                          nn.ReLU(),
                          nn.Dropout(),
                          nn.Linear(64, 32),
                          nn.ReLU(),
                          nn.Dropout(),
                          nn.Linear(32, 1),
                          nn.Sigmoid())


# define the LightningModule
class CustomModel(pl.LightningModule):
    def __init__(self, model=model_seq):
        super().__init__()
        self.model = model

    def forward(self, inputs):
        return self.model(inputs)

    def training_step(self, batch, batch_idx):
        inputs, target = batch
        outputs = self.model(inputs)
        loss = nn.BCELoss()(outputs.float(), torch.reshape(target, (1, 1)).float())
        print("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = optim.SGD(self.model.parameters(), lr=0.1)
        return optimizer
