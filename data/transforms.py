"""
Data pre-processing module ("Data pre-processing module" section of the paper):

    "first, normalization is performed using the mean and standard deviation
     of the images and the images are uniformly scaled to (32, 3, 224, 224).
     Then, the data are enhanced by cropping and random rotation ... with an
     offset of no more than +-25 degrees"

So the pipeline is: resize to 224x224 -> per-image standardization ->
(train only) random crop + random rotation within +-25 degrees.

Normalization modes:
  - "per-image" (default, paper): each image is standardized with its own
    per-channel mean/std.
  - "imagenet" (opt-in): fixed ImageNet-1k mean/std, the statistics
    BiFormer-B was pre-trained with.

The mode is saved into every checkpoint by train.py and read back by eval.py,
so train and eval can't silently disagree.
"""
import torch
from torchvision import transforms

IMG_SIZE = 224
MAX_ROTATION_DEG = 25
CROP_PADDING = 16  # the paper doesn't give the crop size; pad-then-crop keeps 224x224

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

NORM_MODES = ("per-image", "imagenet")


def per_image_normalize(tensor: torch.Tensor) -> torch.Tensor:
    """Standardize a (C, H, W) tensor with its own per-channel mean/std."""
    mean = tensor.mean(dim=(1, 2), keepdim=True)
    std = tensor.std(dim=(1, 2), keepdim=True).clamp_min(1e-6)
    return (tensor - mean) / std


class PerImageNormalize:
    """Picklable transform wrapper (DataLoader workers need to pickle it)."""

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        return per_image_normalize(tensor)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


def _normalize(norm: str):
    if norm == "per-image":
        return PerImageNormalize()
    if norm == "imagenet":
        return transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    raise ValueError(f"Unknown norm mode {norm!r}; expected one of {NORM_MODES}")


def _tensor_stage(norm: str, train: bool) -> list:
    """Everything after Resize + ToTensor. data/cache.py stores the output of
    Resize + ToTensor, so cached images go through exactly these steps."""
    steps = [_normalize(norm)]
    if train:
        # Augmentation after normalization, in the paper's order. Rotation
        # fills corners with 0, i.e. the image mean after standardization.
        steps += [
            transforms.RandomCrop(IMG_SIZE, padding=CROP_PADDING, padding_mode="reflect"),
            transforms.RandomRotation(degrees=MAX_ROTATION_DEG),
        ]
    return steps


def _pil_stage() -> list:
    return [transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.ToTensor()]


def build_train_transform(norm: str = "per-image") -> transforms.Compose:
    return transforms.Compose(_pil_stage() + _tensor_stage(norm, train=True))


def build_eval_transform(norm: str = "per-image") -> transforms.Compose:
    return transforms.Compose(_pil_stage() + _tensor_stage(norm, train=False))


def build_train_tensor_transform(norm: str = "per-image") -> transforms.Compose:
    """build_train_transform without Resize/ToTensor (input: cached float tensor)."""
    return transforms.Compose(_tensor_stage(norm, train=True))


def build_eval_tensor_transform(norm: str = "per-image") -> transforms.Compose:
    return transforms.Compose(_tensor_stage(norm, train=False))
