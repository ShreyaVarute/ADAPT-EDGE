
import csv
import json
import os
import traceback
from pathlib import Path
from datetime import datetime

import torch


class TrainingLogger:
    def __init__(self, experiment_name, config=None):
        self.experiment_name = experiment_name

        # Each experiment gets its own folder
        self.run_dir = Path("outputs") / experiment_name
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.history_file = self.run_dir / "training_history.csv"
        self.log_file = self.run_dir / "training.log"
        self.config_file = self.run_dir / "config.json"

        self.checkpoint_dir = self.run_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Save experiment configuration
        if config is not None:
            with open(self.config_file, "w") as f:
                json.dump(config, f, indent=4)

        self.log("Experiment initialized.")

    def log(self, message):
        """Save messages with timestamps."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        entry = f"[{timestamp}] {message}"

        print(entry)

        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(entry + "\n")

    def log_epoch(self, metrics):
        """
        Append one epoch of metrics to CSV.
        Existing history is preserved.
        """
        file_exists = self.history_file.exists()

        with open(
            self.history_file,
            "a",
            newline="",
            encoding="utf-8"
        ) as f:

            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "epoch",
                    "train_loss",
                    "train_accuracy",
                    "val_loss",
                    "val_accuracy",
                    "learning_rate",
                    "epoch_seconds",
                ],
                extrasaction="ignore",
            )

            if not file_exists:
                writer.writeheader()

            writer.writerow(metrics)

        self.log(
            f"Epoch {metrics['epoch']} completed | "
            f"Train Loss: {metrics['train_loss']:.4f} | "
            f"Train Accuracy: {metrics['train_accuracy']:.2f}% | "
            f"Validation Loss: {metrics['val_loss']:.4f} | "
            f"Validation Accuracy: {metrics['val_accuracy']:.2f}%"
        )

    def save_checkpoint(
        self,
        epoch,
        model,
        optimizer,
        scheduler=None,
        best_val_accuracy=0.0,
        extra=None,
    ):
        """Save complete training state for resuming."""

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": (
                scheduler.state_dict() if scheduler is not None else None
            ),
            "best_val_accuracy": best_val_accuracy,
            "extra": extra or {},
        }

        # Save to a temporary file first to reduce corruption risk
        latest_path = self.checkpoint_dir / "latest_checkpoint.pth"
        temp_path = self.checkpoint_dir / "latest_checkpoint.tmp"

        torch.save(checkpoint, temp_path)
        os.replace(temp_path, latest_path)

        self.log(f"Checkpoint saved after epoch {epoch}.")

        # Preserve the best model separately
        if extra and extra.get("is_best", False):
            best_path = self.checkpoint_dir / "best_model.pth"
            best_temp = self.checkpoint_dir / "best_model.tmp"

            torch.save(checkpoint, best_temp)
            os.replace(best_temp, best_path)

            self.log("New best model checkpoint saved.")

    def load_checkpoint(
        self,
        model,
        optimizer,
        scheduler=None,
        checkpoint_path=None,
    ):
        """Restore model and training state."""

        if checkpoint_path is None:
            checkpoint_path = (
                self.checkpoint_dir / "latest_checkpoint.pth"
            )

        checkpoint_path = Path(checkpoint_path)

        if not checkpoint_path.exists():
            self.log("No checkpoint found. Starting from epoch 1.")
            return 0, 0.0

        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )

        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if scheduler is not None:
            scheduler_state = checkpoint.get("scheduler_state_dict")

            if scheduler_state is not None:
                scheduler.load_state_dict(scheduler_state)

        completed_epoch = checkpoint["epoch"]
        best_accuracy = checkpoint.get("best_val_accuracy", 0.0)

        self.log(
            f"Checkpoint restored. Last completed epoch: "
            f"{completed_epoch}"
        )

        return completed_epoch, best_accuracy

    def log_error(self, error):
        """Save error details for troubleshooting."""

        self.log(f"TRAINING ERROR: {error}")

        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(traceback.format_exc() + "\n")
