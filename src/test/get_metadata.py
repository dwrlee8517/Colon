import torch
import json

ckpt_path = "/radraid2/dongwoolee/Colon/results/ResNetClassifier_3.pth"
ckpt = torch.load(ckpt_path, weights_only=False)

metadata = ckpt["metadata"].copy()
metadata['training_parameters']['schedular'].pop('parameters', None)
print(json.dumps(metadata, indent=2))