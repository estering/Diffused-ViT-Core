from pathlib import Path

from PIL import Image, ImageFilter, ImageOps
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from .config import DataConfig


class DentalPreprocessing:
    def __init__(self, gaussian_radius=1.0):
        self.gaussian_radius = gaussian_radius

    def __call__(self, image):
        image = ImageOps.grayscale(image)
        image = ImageOps.autocontrast(image)
        image = ImageOps.equalize(image)
        image = image.filter(ImageFilter.GaussianBlur(self.gaussian_radius))
        return image.convert("RGB")


def build_transform(training, config=None):
    config = config or DataConfig()
    operations = [DentalPreprocessing(config.gaussian_radius)]
    if training:
        operations.extend(
            [
                transforms.RandomRotation(config.rotation_degrees),
                transforms.RandomResizedCrop(
                    config.image_size,
                    scale=(config.crop_scale_min, 1.0),
                    ratio=(1.0, 1.0),
                ),
            ]
        )
    else:
        operations.append(transforms.Resize((config.image_size, config.image_size)))
    operations.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )
    return transforms.Compose(operations)


class DentalFolderDataset(Dataset):
    classes = ("deep_caries", "pulpitis")
    suffixes = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

    def __init__(self, root, training=False, config=None):
        self.transform = build_transform(training, config)
        self.samples = []
        for label, class_name in enumerate(self.classes):
            for path in sorted((Path(root) / class_name).glob("*")):
                if path.is_file() and path.suffix.lower() in self.suffixes:
                    self.samples.append((path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        with Image.open(path) as image:
            image = self.transform(image)
        return image, label


def build_loader(root, training, batch_size, workers=0):
    return DataLoader(
        DentalFolderDataset(root, training),
        batch_size=batch_size,
        shuffle=training,
        num_workers=workers,
    )
