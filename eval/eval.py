import torchvision.transforms as T
import torch.nn.functional as F
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
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve
from tqdm import tqdm
import matplotlib.pyplot as plt

GLOBAL_SEED = 42
set_random_seed(GLOBAL_SEED)

def load_model_from_checkpoint(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    metadata = checkpoint['metadata']
    model_params = metadata["model_parameters"].copy()
    
    # Reconstruct model architecture from metadata
    model_name = model_params.pop("model_name")
    if model_name in MODEL_REGISTRY:
        model_cls = MODEL_REGISTRY[model_name]
        model = model_cls(**model_params)
    else:
        raise ValueError(f"Unsupported model architecture: {model_name}")

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    return model, metadata

selected_gpu = choose_gpu_with_cuda_visible_devices()
if selected_gpu is not None:
    import torch
    # After setting CUDA_VISIBLE_DEVICES, torch will only see the specified GPU.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training will run on device: {device}")
    print("Visible CUDA devices count:", torch.cuda.device_count())
else:
    print("No GPU selected.")

test_transform = T.Compose([
    CustomTransform(pad_method="zeros", max_size=(1352,1080), target_size=(224,224), augment=False),
    ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    #T.Normalize(mean=[0.568, 0.329, 0.252], std=[0.264, 0.202, 0.161])
])

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

test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=8, pin_memory=True)

for images, labels in test_dataloader:
    print(images.shape, labels.shape)
    break

model, metadata = load_model_from_checkpoint("/radraid2/dongwoolee/Colon/results/ViTClassifier.pth", device)

y_score = []
y_true = []
model.eval()
with torch.no_grad():
    for images, labels in tqdm(test_dataloader, desc="Processing batch"):
        images = images.to(device)
        logits = model(images)
        probs = F.sigmoid(logits)
        y_score.append(probs.detach().cpu())
        y_true.append(labels.detach().cpu())

y_score = torch.cat(y_score).numpy()
y_true = torch.cat(y_true).numpy()

auroc = roc_auc_score(y_true, y_score)
auprc = average_precision_score(y_true, y_score)
fpr, tpr, _ = roc_curve(y_true, y_score)
precision, recall, _ = precision_recall_curve(y_true, y_score)

plt.figure()
plt.plot(fpr, tpr, label=f"ROC Curve (area = {auroc:0.2f})")
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.savefig("/radraid2/dongwoolee/Colon/eval/ROC.png")

plt.figure()
plt.plot(recall, precision, color='b', label=f'Precision-Recall curve (AP = {auprc:0.2f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc="lower left")
plt.savefig("/radraid2/dongwoolee/Colon/eval/PRC.png")