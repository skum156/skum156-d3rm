import pytorch_lightning as pl
from torch import nn
import torch

class D3RM(pl.LightningModule):
    def __init__(self, encoder, decoder, loss_fn, lr):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.loss_fn = loss_fn
        self.lr = lr
        self.save_hyperparameters(ignore=['encoder', 'decoder', 'loss_fn'])

    def training_step(self, batch, batch_idx):
        audio, label, t = batch
        features = self.encoder(audio)
        if features.ndim == 4 and features.shape[1] == 1:
            features = features.squeeze(1)
        if features.ndim != 3:
            raise ValueError(f"Expected features to have 3 dimensions (B, C, T), got {features.shape} after squeeze.")
        features = features.permute(0, 2, 1).contiguous()
        prediction = self.decoder(label, features, t)
        loss = self.loss_fn(prediction, label)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def forward(self, features, label, t):
        if features.ndim == 4 and features.shape[1] == 1:
            features = features.squeeze(1)
        if features.ndim != 3:
            raise ValueError(f"Expected features to have 3 dimensions (B, C, T), got {features.shape} after squeeze in module.py.")
        features = features.permute(0, 2, 1).contiguous()
        prediction = self.decoder(label, features, t)
        return prediction