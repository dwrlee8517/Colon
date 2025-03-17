import torchvision.transforms as T
import torchvision.transforms.functional as F
from PIL import Image
import random

class CustomTransform:
    def __init__(self, pad_method="zeros", max_size=(1352, 1080), target_size=(224, 224),
                 augment=False, normalize=False, mean=None, std=None):
        """
        Args:
            pad_method (str): Padding method. Options are "zeros" (fill with 0s) or "reflect" (reflection padding).
            max_size (tuple): The shape (width, height) of the largest images in the dataset for padding.
            target_size (tuple): Final image shape expected by the model.
            augment (bool): Whether to apply data augmentation.
            normalize (bool): Whether to apply normalization.
            mean (list): Mean for normalization. Default is ImageNet mean if normalize is True.
            std (list): Standard deviation for normalization. Default is ImageNet std if normalize is True.
        """
        self.pad_method = pad_method
        self.max_size = max_size
        self.augment = augment
        self.target_size = target_size
        
        # Define augmentations if requested.
        if self.augment:
            self.augment_transform = T.Compose([
                T.RandomHorizontalFlip(p=0.5),
                T.RandomVerticalFlip(p=0.5),
                T.RandomRotation(30),
            ])
        else:
            self.augment_transform = None

        # Set up normalization if requested.
        self.normalize = normalize
        if self.normalize:
            if mean is None:
                mean = [0.485, 0.456, 0.406]
            if std is None:
                std = [0.229, 0.224, 0.225]
            self.normalize_transform = T.Normalize(mean=mean, std=std)
        else:
            self.normalize_transform = None

        # Always convert to tensor.
        self.to_tensor = T.ToTensor()

    def __call__(self, image: Image.Image) -> Image.Image:
        # Get current image size.
        w, h = image.size # max:1352, 1080
        max_w, max_h = self.max_size
        
        # Calculate padding amounts (center the image).
        pad_left = (max_w - w) // 2 if w < max_w else 0
        pad_top = (max_h - h) // 2 if h < max_h else 0
        pad_right = max_w - w - pad_left if w < max_w else 0
        pad_bottom = max_h - h - pad_top if h < max_h else 0
        
        padding = (pad_left, pad_top, pad_right, pad_bottom)
        
        # Apply padding using the selected method.
        if self.pad_method == "zeros":
            image = F.pad(image, padding, fill=0)
        elif self.pad_method == "reflect":
            image = F.pad(image, padding, padding_mode="reflect")
        else:
            raise ValueError(f"Unknown pad_method: {self.pad_method}")
        
        image = image.resize(self.target_size)
        
        # Apply data augmentation.
        if self.augment and self.augment_transform:
            image = self.augment_transform(image)
        
        # Convert image to tensor.
        image = self.to_tensor(image)

        # Apply normalization if enabled.
        if self.normalize and self.normalize_transform:
            image = self.normalize_transform(image)

        return image