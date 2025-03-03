import torchvision.transforms as T
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from utils.dataset import RealColonDataset
from utils.transformations import CustomTransform
from utils.helpers import choose_gpu_with_cuda_visible_devices, plot_and_save_training_validation_loss, set_random_seed
from utils.models import ViTClassifier, ResNetClassifier, MODEL_REGISTRY
from utils.trainer import ColonTrainer
import torch
from torch.nn import BCEWithLogitsLoss
from torch.optim import lr_scheduler, AdamW

def load_checkpoint(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    metadata = checkpoint['metadata']
    model_params = metadata["model_parameters"]
    training_params = metadata["training_parameters"]
    transform_params = metadata["transform_parameters"]
    
    # Reconstruct model architecture from metadata
    model_name = model_params["model_name"]
    if model_name in MODEL_REGISTRY:
        model_cls = MODEL_REGISTRY[model_name]
        model_args = model_params.copy()
        model_args = model_params.pop("model_name")
        model = model_cls(**model_args)
    else:
        raise ValueError(f"Unsupported model architecture: {model_name}")

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, metadata