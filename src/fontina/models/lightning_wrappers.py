import pytorch_lightning as L
import torch
import torch.nn.functional as F

from .deepfont import DeepFont, DeepFontAutoencoder
from torchmetrics import Accuracy
from pytorch_lightning.utilities.grads import grad_norm


class DeepFontAutoencoderWrapper(L.LightningModule):
    def __init__(self):
        super().__init__()
        # Saving hyperparameters of autoencoder
        self.save_hyperparameters()
        self.autoencoder = DeepFontAutoencoder()
        # Example input array needed for visualizing the graph of the network
        self.example_input_array = torch.zeros(2, 1, 105, 105)

    def forward(self, x):
        return self.autoencoder(x)

    def _get_reconstruction_loss(self, batch):
        x, _ = batch
        x_hat = self.forward(x)
        return F.mse_loss(x, x_hat)

    def training_step(self, batch, batch_idx):
        loss = self._get_reconstruction_loss(batch)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._get_reconstruction_loss(batch)
        self.log("val_loss", loss, prog_bar=True)

    def test_step(self, batch, batch_idx):
        loss = self._get_reconstruction_loss(batch)
        self.log("test_loss", loss, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        # Using a scheduler is optional but can be helpful.
        # The scheduler reduces the LR if the validation
        # performance hasn't improved for the last N epochs
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.2, patience=20, min_lr=5e-5
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}


class DeepFontWrapper(L.LightningModule):
    def __init__(self, model: DeepFont, num_classes: int, learning_rate: float = 0.01):
        super().__init__()

        self.learning_rate = learning_rate
        self.model = model
        self.accuracy = Accuracy(task="multiclass", num_classes=num_classes, top_k=1)
        self.eval_loss: list[torch.Tensor] = []
        self.eval_accuracy: list[torch.Tensor] = []
        self.test_accuracy: list[torch.Tensor] = []
        self.example_input_array = torch.zeros(2, 1, 105, 105)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)

        train_acc = self.accuracy(y_hat.argmax(1), y)

        self.log("train_acc", train_acc, prog_bar=True)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, acc = self._shared_eval(batch)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_accuracy", acc, prog_bar=True)
        self.eval_loss.append(loss)
        self.eval_accuracy.append(acc)
        return {"val_loss": loss, "val_accuracy": acc}

    def test_step(self, batch, batch_idx):
        loss, acc = self._shared_eval(batch)
        self.log_dict({"test_loss": loss, "test_acc": acc})

    def _shared_eval(self, batch):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        acc = self.accuracy(y_hat.argmax(1), y)
        return loss, acc

    def on_before_optimizer_step(self, optimizer):
        # Compute the 2-norm for each layer
        # If using mixed precision, the gradients are already unscaled here
        norms = grad_norm(self.model, norm_type=2)
        self.log_dict(norms)

    def configure_optimizers(self):
        params = filter(lambda p: p.requires_grad, self.parameters())
        optimizer = torch.optim.SGD(
            params, lr=self.learning_rate, momentum=0.9, weight_decay=0.0005
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, "min", patience=5
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }
