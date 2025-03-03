import torchvision.transforms as T
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from utils.dataset import RealColonDataset
from utils.transformations import CustomTransform
from utils.helpers import choose_gpu_with_cuda_visible_devices, plot_and_save_training_validation_loss, set_random_seed
from utils.models import ViTClassifier, ResNetClassifier
from utils.trainer import ColonTrainer
from torch.nn import BCEWithLogitsLoss
from torch.optim import lr_scheduler, AdamW
import gpustat

# Set a global random seed here.
GLOBAL_SEED = 42
set_random_seed(GLOBAL_SEED)

# Create a composed transform that uses your custom transform and converts the image to a tensor.
train_transform = T.Compose([
    CustomTransform(pad_method="reflect", max_size=(1352,1080), target_size=(224,224), augment=True),
    ToTensor()
])

test_transform = T.Compose([
    CustomTransform(pad_method="reflect", max_size=(1352,1080), target_size=(224,224), augment=False),
    ToTensor()
])

print("Creating Train Dataset")
train_dataset = RealColonDataset(
    data_dir="/radraid/dongwoolee/real_colon_data",
    frames_csv="/radraid2/dongwoolee/Colon/data/frames_train.csv",
    num_imgs=8000,
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
    num_imgs=2000,
    pos_ratio=0.5,
    min_skip_frames=10,
    apply_skip=True,
    transform=test_transform
)
print("Done.")

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

for images, labels in train_dataloader:
    print(images.shape, labels.shape)
    break

for images, labels in test_dataloader:
    print(images.shape, labels.shape)
    break

selected_gpu = choose_gpu_with_cuda_visible_devices()
if selected_gpu is not None:
    import torch
    # After setting CUDA_VISIBLE_DEVICES, torch will only see the specified GPU.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training will run on device: {device}")
    print("Visible CUDA devices count:", torch.cuda.device_count())
else:
    print("No GPU selected.")

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

############# Training Parameters #####################
learning_rate = 0.001
epochs = 30
loss_fn = BCEWithLogitsLoss()
optimizer = AdamW(model.parameters(), lr=learning_rate)
scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
save_filepath = "/radraid2/dongwoolee/Colon/results/checkpoint.pth"
save_plotpath = "/radraid2/dongwoolee/Colon/results/lossplot.png"

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
    "schedular": {"name": type(scheduler).__name__, "parameters": scheduler.__dict__},
}

transform_parameters = {
    "pad_method": train_transform.pad_method,
    "max_size": train_transform.max_size,
    "target_size": train_transform.target_size
}

metadata = {
    "model_parameters": model_parameters,
    "training_parameters": training_parameters,
    "transformation_parameters": transform_parameters
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