import os
from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import torch.nn.functional as F
import matplotlib.pyplot as plt
from datetime import datetime


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader,
        val_loader,
        test_loader,
        device: torch.device,
        lr: float,
        epochs: int,
        patience: int,
        checkpoint_dir: str,
        criterion: nn.Module,
        optimizer_class: optim.Optimizer,
        scheduler_class: optim.lr_scheduler._LRScheduler,
        scheduler_kwargs: Dict,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.epochs = epochs
        self.patience = patience
        self.best_val_loss = float("inf")
        self.epochs_without_improvement = 0
        self.checkpoint_dir = checkpoint_dir

        # Create checkpoint directory for the specific model
        self.model_name = model.__class__.__name__
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.model_checkpoint_dir = os.path.join(
            self.checkpoint_dir, f"{self.model_name}_{self.timestamp}"
        )

        self.criterion = criterion
        self.optimizer = optimizer_class(self.model.parameters(), lr=lr)
        self.scheduler = scheduler_class(self.optimizer, **scheduler_kwargs)

    def train(self):
        self.writer = SummaryWriter()
        os.makedirs(self.model_checkpoint_dir, exist_ok=True)

        for epoch in range(self.epochs):
            self.model.train()
            train_metrics = self._run_epoch(self.train_loader, training=True)
            train_loss = train_metrics["loss"]
            self.writer.add_scalar("Loss/train", train_loss, epoch)

            val_metrics = self.validate()
            val_loss = val_metrics["loss"]
            self.writer.add_scalar("Loss/val", val_loss, epoch)

            for metric_name, metric_value in val_metrics.items():
                self.writer.add_scalar(
                    f"Metrics/val_{metric_name}", metric_value, epoch
                )

            self.scheduler.step(val_loss)

            print(
                f"Epoch {epoch+1}/{self.epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, LR: {self.optimizer.param_groups[0]['lr']:.6f}"
            )
            print(f"Validation Metrics: {val_metrics}")

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.epochs_without_improvement = 0
                self.save_model("best")
                print(f"Best model saved with val loss: {val_loss:.4f}")
            else:
                self.epochs_without_improvement += 1

            if self.epochs_without_improvement >= self.patience:
                print(f"Early stopping at epoch {epoch+1}!")
                break

        self.save_model("last")
        print("Last model saved.")
        self.writer.close()

    def validate(self):
        self.model.eval()
        with torch.no_grad():
            metrics = self._run_epoch(self.val_loader, training=False)
        return metrics

    def test(self, checkpoint: str = None):
        if os.path.isfile(checkpoint):
            self.load_model(checkpoint)

        self.model.eval()
        with torch.no_grad():
            metrics = self._run_epoch(self.test_loader, training=False)

        print(f"Test Loss: {metrics['loss']:.4f}")
        print(f"Test Metrics: {metrics}")
        return metrics

    def _run_epoch(
        self, data_loader, training: bool = True
    ) -> Tuple[float, Dict[str, float]]:
        total_loss = 0.0
        total_tp, total_fp, total_tn, total_fn = 0, 0, 0, 0

        for images, true_masks, _, _ in tqdm(
            data_loader, desc="Training" if training else "Validation"
        ):
            images, true_masks = images.to(self.device), true_masks.to(self.device)

            if training:
                self.optimizer.zero_grad()

            masks_pred = self.model(images)
            loss = self.criterion(masks_pred, true_masks)
            loss += dice_loss(F.sigmoid(masks_pred), true_masks, multiclass=False)

            if training:
                loss.backward()
                self.optimizer.step()

            total_loss += loss.item()

            masks_pred = (masks_pred > 0.5).float()
            tp, fp, tn, fn = self._calculate_metrics(masks_pred, true_masks)
            total_tp += tp
            total_fp += fp
            total_tn += tn
            total_fn += fn

        avg_loss = total_loss / len(data_loader)
        metrics = self._calculate_metrics_from_counts(
            total_tp, total_fp, total_tn, total_fn
        )
        metrics["loss"] = avg_loss

        return metrics

    def _calculate_metrics(
        self, preds: torch.Tensor, masks: torch.Tensor
    ) -> Tuple[int, int, int, int]:
        tp = (preds * masks).sum().item()
        fp = (preds * (1 - masks)).sum().item()
        tn = ((1 - preds) * (1 - masks)).sum().item()
        fn = ((1 - preds) * masks).sum().item()
        return tp, fp, tn, fn

    def _calculate_metrics_from_counts(
        self, tp: int, fp: int, tn: int, fn: int
    ) -> Dict[str, float]:
        epsilon = 1e-7
        precision = tp / (tp + fp + epsilon)
        recall = tp / (tp + fn + epsilon)
        specificity = tn / (tn + fp + epsilon)
        accuracy = (tp + tn) / (tp + tn + fp + fn + epsilon)
        dice = (2 * tp) / (2 * tp + fp + fn + epsilon)
        iou = tp / (tp + fp + fn + epsilon)

        return {
            "precision": precision,
            "recall": recall,
            "specificity": specificity,
            "accuracy": accuracy,
            "dice": dice,
            "iou": iou,
        }

    def save_model(self, checkpoint_type: str):
        """Save model checkpoint."""
        checkpoint_name = f"{checkpoint_type}_model.pth"
        checkpoint_path = os.path.join(self.model_checkpoint_dir, checkpoint_name)
        torch.save(self.model.state_dict(), checkpoint_path)
        print(f"Model saved to {checkpoint_path}")

    def load_model(self, checkpoint: str):
        """Load the checkpoint."""
        if not checkpoint:
            raise FileNotFoundError(f"No {checkpoint} checkpoint found.")

        self.model.load_state_dict(torch.load(checkpoint, map_location=self.device))
        print(f"Loaded checkpoint from {checkpoint}")

    def visualize_predictions(self, checkpoint: str):
        """Visualize predictions using the specified checkpoint."""
        self.load_model(checkpoint)

        # Get the first batch from the test loader
        images, masks, original_images, original_masks = next(iter(self.test_loader))
        images, masks = images.to(self.device), masks.to(self.device)

        # Get the model predictions
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(images)
            preds = F.interpolate(
                outputs, size=(600, 800), mode="bilinear", align_corners=False
            )

        # Move tensors to CPU and convert to numpy for visualization
        original_images = original_images.numpy()  # (B, 600, 800, 3)
        original_masks = original_masks.unsqueeze(3).numpy()  # (B, 600, 800, 1)

        preds = preds.cpu().numpy().transpose(0, 2, 3, 1)  # (B, 600, 800, 1)

        ground_truth = (
            original_images * (original_masks > 0.5) / 255
        )  # (B, 600, 800, 3)
        predicted = original_images * (preds > 0.5) / 255  # (B, 600, 800, 3)

        # Create a 3-row plot
        scale = (images.shape[0] // 4) or 1
        fig, axes = plt.subplots(3, len(images), figsize=(15 * scale, 10))

        for i in range(len(images)):
            # Original image (3 channels)
            axes[0, i].imshow(original_images[i])
            axes[0, i].set_title("Original Image")
            axes[0, i].axis("off")

            # Ground truth mask (1 channel)
            axes[1, i].imshow(ground_truth[i])
            axes[1, i].set_title("Ground Truth Mask")
            axes[1, i].axis("off")

            # Predicted mask (1 channel)
            axes[2, i].imshow(predicted[i])
            axes[2, i].set_title("Predicted Mask")
            axes[2, i].axis("off")

        plt.tight_layout()
        plt.show()


def dice_coeff(input, target, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    # Average of Dice coefficient for all batches, or for a single mask
    assert input.size() == target.size()
    assert input.dim() == 3 or not reduce_batch_first

    sum_dim = (-1, -2) if input.dim() == 2 or not reduce_batch_first else (-1, -2, -3)

    inter = 2 * (input * target).sum(dim=sum_dim)
    sets_sum = input.sum(dim=sum_dim) + target.sum(dim=sum_dim)
    sets_sum = torch.where(sets_sum == 0, inter, sets_sum)

    dice = (inter + epsilon) / (sets_sum + epsilon)
    return dice.mean()


def multiclass_dice_coeff(
    input, target, reduce_batch_first: bool = False, epsilon: float = 1e-6
):
    # Average of Dice coefficient for all classes
    return dice_coeff(
        input.flatten(0, 1), target.flatten(0, 1), reduce_batch_first, epsilon
    )


def dice_loss(input, target, multiclass: bool = False):
    # Dice loss (objective to minimize) between 0 and 1
    fn = multiclass_dice_coeff if multiclass else dice_coeff
    return 1 - fn(input, target, reduce_batch_first=False)
