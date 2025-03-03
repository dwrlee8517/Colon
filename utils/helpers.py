import os
import subprocess
import matplotlib.pyplot as plt
import numpy as np
import random
import torch


def set_random_seed(seed: int):
    """
    Set the random seed for reproducibility across random, numpy, and torch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # For deterministic behavior, note that this might impact performance.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def choose_gpu_with_cuda_visible_devices():
    """
    Displays current GPU status using gpustat (or nvidia-smi as a fallback),
    prompts the user to select a GPU ID, and sets CUDA_VISIBLE_DEVICES accordingly.
    
    Returns:
        int: The GPU ID selected by the user.
    """
    try:
        # Use gpustat for a concise status view.
        output = subprocess.check_output("gpustat", shell=True).decode("utf-8")
    except Exception:
        print("gpustat not available. Falling back to nvidia-smi.")
        try:
            output = subprocess.check_output("nvidia-smi", shell=True).decode("utf-8")
        except Exception as e:
            print("nvidia-smi command failed. No GPU info available.")
            return None

    print("Current GPU Status:")
    print(output)
    
    # Prompt the user for a GPU id.
    while True:
        gpu_id = input("Enter the GPU id to use for training: ").strip()
        if gpu_id.isdigit():
            # Set the CUDA_VISIBLE_DEVICES environment variable.
            os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
            print(f"Set CUDA_VISIBLE_DEVICES to {gpu_id}")
            return int(gpu_id)
        else:
            print("Invalid input. Please enter a numeric GPU ID.")

def plot_and_save_training_validation_loss(train_losses, val_losses, img_filename):
    """
    Plot the training and validation loss over epochs.
    """
    epochs = np.arange(1, len(train_losses) + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, marker='o', label="Training Loss")
    plt.plot(epochs, val_losses, marker='o', label="Validation Loss")
    plt.title("Training and Validation Loss over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(img_filename)
    plt.show()