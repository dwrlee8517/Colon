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

# Define a basic transform that only converts images to tensors
train_transform = T.Compose([
    CustomTransform(pad_method="zeros", max_size=(1352,1080), target_size=(224,224), augment=False),
    ToTensor(),
])

test_transform = T.Compose([
    CustomTransform(pad_method="zeros", max_size=(1352,1080), target_size=(224,224), augment=False),
    ToTensor(),
])

print("Loading Dataset")
# Load your training dataset (replace 'path/to/train' with your actual path)
train_dataset = RealColonDataset(
    data_dir="/radraid/dongwoolee/real_colon_data",
    frames_csv="/radraid2/dongwoolee/Colon/data/frames_train.csv",
    num_imgs=80000,
    pos_ratio=0.5,
    min_skip_frames=5,
    apply_skip=True,
    transform=train_transform
)

test_dataset = RealColonDataset(
    data_dir="/radraid/dongwoolee/real_colon_data",
    frames_csv="/radraid2/dongwoolee/Colon/data/frames_test.csv",
    num_imgs=20000,
    pos_ratio=0.5,
    min_skip_frames=5,
    apply_skip=True,
    transform=test_transform
)


train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

# Initialize variables to store sum and squared sum
mean = 0.0
std = 0.0
nb_samples = 0

print("Calculating mean, std")
for data, _ in train_dataloader:
    batch_samples = data.size(0)  # number of images in the batch
    data = data.view(batch_samples, data.size(1), -1)  # flatten H and W into one dimension
    mean += data.mean(2).sum(0)
    std += data.std(2).sum(0)
    nb_samples += batch_samples

mean /= nb_samples
std /= nb_samples

print(f"Train Dataset Mean: {mean}")
print(f"Train Dataset Std: {std}")

# Initialize variables to store sum and squared sum
mean = 0.0
std = 0.0
nb_samples = 0

for data, _ in test_dataloader:
    batch_samples = data.size(0)  # number of images in the batch
    data = data.view(batch_samples, data.size(1), -1)  # flatten H and W into one dimension
    mean += data.mean(2).sum(0)
    std += data.std(2).sum(0)
    nb_samples += batch_samples

mean /= nb_samples
std /= nb_samples

print(f"Test Dataset Mean: {mean}")
print(f"Test Dataset Std: {std}")