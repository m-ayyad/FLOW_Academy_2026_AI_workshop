"""
Workshop utility module for River Ice Image Classification.

Sections
--------
1. Dataset       – loads images from disk, applies augmentation & normalisation
2. Augmentation  – albumentations pipeline for training
3. Visualisation – quick display helpers
4. Training      – TrainEpoch / ValidEpoch loop helpers
5. Evaluation    – test_model(), show_confusion_matrix()
6. EarlyStopping – optional training callback
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import albumentations as albu
from torch.utils.data import Dataset as _BaseDataset
from tqdm import tqdm
from sklearn.metrics import confusion_matrix as _cm


# =============================================================================
# 1. Dataset
# =============================================================================

class Dataset(_BaseDataset):
    """River-ice image classification dataset.

    Loads images from disk, resizes them to IMAGE_WIDTH x IMAGE_HEIGHT,
    optionally applies augmentation and ImageNet normalisation, then returns
    (image_tensor, integer_label) pairs for a DataLoader.

    Args:
        images (list[str])    : Absolute paths to image files.
        labels (list[int])    : Integer class label for each image.
        augmentation          : albumentations.Compose pipeline, or None.
        normalize (bool)      : Apply ImageNet mean/std and convert to float32
                                CHW format expected by PyTorch.
    """

    IMAGE_WIDTH  = 1152
    IMAGE_HEIGHT = 640

    def __init__(self, images, labels, augmentation=None, normalize=False):
        self.images      = images
        self.labels      = labels
        self.augmentation = augmentation
        self.normalize    = normalize

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        image = cv2.imread(self.images[i])
        image = cv2.resize(
            image,
            (self.IMAGE_WIDTH, self.IMAGE_HEIGHT),
            interpolation=cv2.INTER_CUBIC,
        )
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.augmentation:
            image = self.augmentation(image=image)["image"]

        if self.normalize:
            image = self._to_tensor(image)

        return image, self.labels[i]

    @staticmethod
    def _to_tensor(x):
        """uint8 HWC → float32 CHW, ImageNet-normalised."""
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        x    = x.astype(np.float32) / 255.0
        x    = (x - mean) / std
        return x.transpose(2, 0, 1)   # HWC → CHW


# =============================================================================
# 2. Augmentation
# =============================================================================

def get_training_augmentation():
    """Return an albumentations pipeline for training-time augmentation.

    Applies geometric and colour transforms that mimic real-world variability
    in river-ice camera imagery (lighting changes, angles, motion blur, noise).
    """
    return albu.Compose([
        albu.HorizontalFlip(p=0.5),
        albu.OneOf([
            albu.Sharpen(p=1),
            albu.Blur(blur_limit=3, p=1),
        ], p=0.9),
    ])


# =============================================================================
# 3. Visualisation
# =============================================================================

def visualize(**images):
    """Display one or more images side-by-side.

    Usage::

        visualize(original=img1, augmented=img2)
        visualize(**{"Open Water": img})
    """
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(" ".join(name.split("_")).title())
        plt.imshow(image)
    plt.tight_layout()
    plt.show()


# =============================================================================
# 4. Training loop helpers
# =============================================================================

class _Epoch:
    """Base class — one full pass over a DataLoader.

    Subclasses implement _set_mode() and _forward() to define train vs. eval
    behaviour.  run() returns a log dict: {"loss": float, "<MetricName>": float}.
    """

    stage = ""  # set by subclasses ("train" / "valid")

    def __init__(self, model, loss_fn, metrics, device="cpu"):
        self.model   = model.to(device)
        self.loss_fn = loss_fn
        self.metrics = [m.to(device) for m in metrics]
        self.device  = device

    def _set_mode(self):
        raise NotImplementedError

    def _forward(self, x, y):
        raise NotImplementedError

    def run(self, dataloader):
        self._set_mode()
        total_loss = 0.0
        for m in self.metrics:
            m.reset()

        with tqdm(dataloader, desc=self.stage, leave=True) as pbar:
            for x, y in pbar:
                x, y  = x.to(self.device), y.to(self.device)
                loss, preds = self._forward(x, y)
                total_loss += loss.item()
                for m in self.metrics:
                    m(preds, y)
                # Live stats in the progress bar
                acc = self.metrics[0].compute().item() if self.metrics else 0.0
                pbar.set_postfix(
                    loss=f"{loss.item():.4f}",
                    accuracy=f"{100 * acc:.2f}%",
                )

        avg_loss    = total_loss / len(dataloader)
        metric_vals = {
            getattr(m, "name", type(m).__name__): m.compute().item()
            for m in self.metrics
        }
        return {"loss": avg_loss, **metric_vals}


class TrainEpoch(_Epoch):
    """One training pass — computes loss, back-propagates, updates weights."""

    stage = "train"

    def __init__(self, model, loss_fn, metrics, optimizer, device="cpu"):
        super().__init__(model, loss_fn, metrics, device)
        self.optimizer = optimizer

    def _set_mode(self):
        self.model.train()

    def _forward(self, x, y):
        self.optimizer.zero_grad()
        preds = self.model(x)
        loss  = self.loss_fn(preds, y)
        loss.backward()
        self.optimizer.step()
        return loss, preds


class ValidEpoch(_Epoch):
    """One validation pass — no weight updates, runs under torch.no_grad()."""

    stage = "valid"

    def _set_mode(self):
        self.model.eval()

    def _forward(self, x, y):
        with torch.no_grad():
            preds = self.model(x)
            loss  = self.loss_fn(preds, y)
        return loss, preds


# =============================================================================
# 5. Evaluation
# =============================================================================

def test_model(model, test_dataloader, test_dataset_vis, class2idx,
               plot_incorrect=False):
    """Run inference on the full test set.

    Args:
        model             : Trained PyTorch model.
        test_dataloader   : DataLoader with normalised images.
        test_dataset_vis  : Dataset without normalisation (for display only).
        class2idx (dict)  : {"ClassName": int_label, ...}
        plot_incorrect    : If True, display misclassified images.

    Returns:
        preds     (np.ndarray, N)      : Predicted class indices.
        labels    (np.ndarray, N)      : True class indices.
        logits    (np.ndarray, N x C)  : Raw model outputs.
        probs     (np.ndarray, N x C)  : Softmax probabilities.
        incorrect (list[str])          : Filenames of misclassified images.
    """
    device    = "cuda" if torch.cuda.is_available() else "cpu"
    idx2class = {v: k for k, v in class2idx.items()}
    model     = model.to(device)
    model.eval()

    all_logits, all_preds, all_labels = [], [], []
    incorrect = []

    with torch.no_grad():
        for batch_idx, (images, batch_labels) in enumerate(test_dataloader):
            images       = images.to(device)
            batch_logits = model(images).cpu()

            all_logits.append(batch_logits.numpy())
            all_preds.append(batch_logits.argmax(dim=1).numpy())
            all_labels.append(batch_labels.numpy())

            start = batch_idx * test_dataloader.batch_size
            for local_i in range(len(batch_labels)):
                pred = batch_logits[local_i].argmax().item()
                true = batch_labels[local_i].item()
                if pred != true:
                    global_i = start + local_i
                    name = os.path.basename(
                        test_dataloader.dataset.images[global_i]
                    )
                    incorrect.append(name)
                    if plot_incorrect:
                        img_vis = test_dataset_vis[global_i][0].astype("uint8")
                        plt.figure()
                        plt.imshow(img_vis)
                        plt.title(
                            f"{name}\n"
                            f"True: {idx2class[true]},  "
                            f"Predicted: {idx2class[pred]}"
                        )
                        plt.axis("off")
                        plt.tight_layout()
                        plt.show()

    logits = np.concatenate(all_logits, axis=0)
    preds  = np.concatenate(all_preds,  axis=0)
    labels = np.concatenate(all_labels, axis=0)
    probs  = torch.softmax(torch.tensor(logits), dim=1).numpy()

    return preds, labels, logits, probs, incorrect


def show_confusion_matrix(predictions, labels, class2idx, save_path=None):
    """Plot a normalised confusion matrix.

    Args:
        predictions (np.ndarray) : Predicted class indices.
        labels      (np.ndarray) : True class indices.
        class2idx   (dict)       : {"ClassName": int_label, ...}
        save_path   (str | None) : Full path to save the PNG, or None.
    """
    class_names = list(class2idx.keys())
    class_idxs  = list(class2idx.values())

    cm        = _cm(labels, predictions, labels=class_idxs)
    cm_ratios = cm / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(cm_ratios, interpolation="nearest", cmap=plt.cm.Blues)
    ax.set_yticks(range(len(class_names)))
    ax.set_yticklabels(class_names, rotation=90, fontsize=12)
    ax.set_xticks(range(len(class_names)))
    ax.set_xticklabels(class_names, fontsize=12)
    ax.set_ylabel("True Label",      fontsize=15)
    ax.set_xlabel("Predicted Label", fontsize=15)

    for i in range(len(cm)):
        for j in range(len(cm[i])):
            color = "white" if cm_ratios[i][j] >= 0.7 else "black"
            ax.text(
                j, i,
                f"{cm[i][j]}\n({100 * cm_ratios[i][j]:.1f}%)",
                ha="center", va="center",
                color=color, fontsize=15,
            )

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
        print(f"Confusion matrix saved to {save_path}")
    plt.show()


# =============================================================================
# 6. Early stopping (optional)
# =============================================================================

class EarlyStopping:
    """Stop training early to prevent overfitting or plateau.

    Two usage modes (call one method per epoch, then check `early_stop`):

    - ``overfitting(train_loss, val_loss)``  — stops when val loss diverges
      from train loss by more than ``min_delta`` for ``patience`` epochs.
    - ``plateau(train_loss)``                — stops when train loss improvement
      falls below ``min_delta`` for ``patience`` epochs.

    Example::

        stopper = EarlyStopping(patience=5, min_delta=0.005)
        for epoch in range(NUM_EPOCHS):
            ...
            stopper.overfitting(train_loss, val_loss)
            if stopper.early_stop:
                print("Stopping early.")
                break
    """

    def __init__(self, patience=5, min_delta=0.0):
        self.patience   = patience
        self.min_delta  = min_delta
        self.counter    = 0
        self.early_stop = False
        self._prev_loss = float("inf")

    def overfitting(self, train_loss, val_loss):
        if (val_loss - train_loss) > self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.counter = 0

    def plateau(self, train_loss):
        if abs(self._prev_loss - train_loss) < self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.counter = 0
        self._prev_loss = train_loss
