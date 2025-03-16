
```bash
├── configs
│   ├── config.yaml
│   ├── model
│   │   ├── ResNetClassifier.yaml
│   │   └── ViTClassifier.yaml
│   ├── optimizer
│   │   ├── Adam.yaml
│   │   └── AdamW.yaml
│   └── scheduler
│       ├── CosineAnnealingLR.yaml
│       └── ReduceOnPlateau.yaml
├── data
├── notebooks
├── results
├── src
│   ├── models
│   ├── train
│   ├── data
│   └── utils
└── test
    ├── eval.py
    └── get_metadata.py