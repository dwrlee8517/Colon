import torchvision.transforms as T
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from src.data.dataset import RealColonDataset
from utils.transformations import CustomTransform
from utils.helpers import choose_gpu_with_cuda_visible_devices, plot_and_save_training_validation_loss, set_random_seed
from src.models.models import ViTClassifier, ResNetClassifier
from src.train.trainer import ColonTrainer
from torch.nn import BCEWithLogitsLoss
from torch.optim import lr_scheduler, AdamW
import gpustat

# Set a global random seed here.
GLOBAL_SEED = 42
set_random_seed(GLOBAL_SEED)

selected_gpu = choose_gpu_with_cuda_visible_devices()
if selected_gpu is not None:
    import torch
    # After setting CUDA_VISIBLE_DEVICES, torch will only see the specified GPU.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training will run on device: {device}")
    print("Visible CUDA devices count:", torch.cuda.device_count())
else:
    print("No GPU selected.")

# Create a composed transform that uses your custom transform and converts the image to a tensor.
train_transform = T.Compose([
    CustomTransform(pad_method="zeros", max_size=(1352,1080), target_size=(224,224), augment=True),
    ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # ImageNet
    #T.Normalize(mean=[0.568, 0.329, 0.252], std=[0.264, 0.202, 0.161]) # train_dataset when random seed = 42
])

test_transform = T.Compose([
    CustomTransform(pad_method="zeros", max_size=(1352,1080), target_size=(224,224), augment=False),
    ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    #T.Normalize(mean=[0.568, 0.329, 0.252], std=[0.264, 0.202, 0.161])
])

print("Creating Train Dataset")
train_dataset = RealColonDataset(
    data_dir="/radraid/dongwoolee/real_colon_data",
    frames_csv="/radraid2/dongwoolee/Colon/data/frames_train.csv",
    num_imgs=20000,
    pos_ratio=0.5,
    min_skip_frames=10,
    apply_skip=True,
    transform=train_transform
)
print("Done.")

print("Creating Test Datset")
test_dataset = RealColonDataset(
    data_dir="/radraid/dongwoolee/real_colon_data",
    frames_csv="/radraid2/dongwoolee/Colon/data/frames_test.csv",
    num_imgs=5000,
    pos_ratio=0.5,
    min_skip_frames=10,
    apply_skip=True,
    transform=test_transform
)
print("Done.")

train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=8, pin_memory=True)
test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=8, pin_memory=True)

for images, labels in train_dataloader:
    print(images.shape, labels.shape)
    break

for images, labels in test_dataloader:
    print(images.shape, labels.shape)
    break

############## Model Parameters ######################
vit_backbone = 'vit_base_patch16_224'
resnet_backbone = 'resnet50'

hidden_dims = [32]
dropout_p = 0.2
activation = "ReLU"
batch_norm_bool = True
num_classes = 1
freeze_backbone = True

model_name = "ViTClassifier"

print(f"Loading Model {model_name}")
if model_name == "ViTClassifier":
    backbone_name = vit_backbone
    model = ViTClassifier(backbone_name=backbone_name, 
    hidden_dims=hidden_dims, 
    activation=activation, 
    batch_norm=batch_norm_bool, 
    dropout=dropout_p, 
    num_classes=num_classes,
    freeze_backbone=freeze_backbone
    )

elif model_name == "ResNetClassifier":
    backbone_name = resnet_backbone
    model = ResNetClassifier(backbone_name=backbone_name, 
    hidden_dims=hidden_dims, 
    activation=activation, 
    batch_norm=batch_norm_bool, 
    dropout=dropout_p, 
    num_classes=num_classes,
    freeze_backbone=freeze_backbone
    )

model.to(device)
print(f"Loaded Model: {type(model).__name__}")

############# Training Parameters #####################
learning_rate = 0.00005
epochs = 10
loss_fn = BCEWithLogitsLoss()
optimizer = AdamW(model.parameters(), lr=learning_rate)
scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=learning_rate * 0.01)
save_filepath = f"/radraid2/dongwoolee/Colon/results/{model_name}.pth"
save_plotpath = f"/radraid2/dongwoolee/Colon/results/{model_name}.png"

############ Train and save results ##################
trainer = ColonTrainer(model, optimizer, scheduler, train_dataloader, test_dataloader, device, epochs, loss_fn)
model, optimizer, train_loss_history, test_loss_history = trainer.train()

print("Saving Plot...")
plot_and_save_training_validation_loss(train_loss_history, test_loss_history, save_plotpath)
print(f"Loss plot is saved to {save_plotpath}")

print("Saving Model...")
model_parameters = {
    "model_name": model_name,
    "backbone_name": backbone_name,
    "hidden_dims": hidden_dims,
    "dropout": dropout_p,
    "activation": activation,
    "batch_norm": batch_norm_bool,
    "num_classes": num_classes,
    "freeze_backbone": freeze_backbone
}

training_parameters = {
    "learning_rate": learning_rate,
    "epochs": epochs,
    "loss": type(loss_fn).__name__,
    "optimizer": type(optimizer).__name__,
    "schedular": type(scheduler).__name__,
}

transform_parameters = {
    "pad_method": train_transform.transforms[0].pad_method,
    "max_size": train_transform.transforms[0].max_size,
    "target_size": train_transform.transforms[0].target_size,
    "mean": train_transform.transforms[2].mean,
    "std": train_transform.transforms[2].std,
}

metadata = {
    "model_parameters": model_parameters,
    "training_parameters": training_parameters,
    "transform_parameters": transform_parameters
}

checkpoint = {
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "metadata": metadata,
    "train_loss": train_loss_history,
    "test_loss": test_loss_history,
}
torch.save(checkpoint, save_filepath)
print(f"Model is saved to {save_filepath}")