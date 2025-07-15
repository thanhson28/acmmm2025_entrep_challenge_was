model_settings = {
    "efficientnet_b4": {
        "optimizer": "AdamW",
        "head_only_lr": 1e-4,
        "full_fine_tune_lr": 1e-4,
        "input_size": (380, 380),
        "augmentation": ["RandAugment", "Mixup"],
        "weight_decay": 0,
        "normalization": {
            "mean": (0.485, 0.456, 0.406),
            "std": (0.229, 0.224, 0.225)
        }
    },
    "efficientnetv2_rw_m.agc_in1k": {
        "optimizer": "AdamW",
        "head_only_lr": 1e-4,
        "full_fine_tune_lr": 1e-4,
        "input_size": (380, 380),
        "augmentation": ["RandAugment", "Cutmix"],
        "weight_decay": 0,
        "normalization": {
            "mean": (0.485, 0.456, 0.406),
            "std": (0.229, 0.224, 0.225)
        }
    },
    "resnet101": {
        "optimizer": "AdamW",
        "head_only_lr": 3e-4,
        "full_fine_tune_lr": 1e-4,
        "input_size": (380, 380),
        "augmentation": ["AutoAugment", "ColorJitter"],
        "weight_decay": 0,
        "normalization": {
            "mean": (0.485, 0.456, 0.406),
            "std": (0.229, 0.224, 0.225)
        }
    },
    "densenet121": {
        "optimizer": "AdamW",
        "head_only_lr": 3e-4,
        "full_fine_tune_lr": 1e-4,
        "input_size": (380, 380),
        "augmentation": ["AutoAugment", "ColorJitter"],
        "weight_decay": 10,
        "normalization": {
            "mean": (0.485, 0.456, 0.406),
            "std": (0.229, 0.224, 0.225)
        }
    },
    "inception_v3": {
        "optimizer": "AdamW",
        "head_only_lr": 1e-4,
        "full_fine_tune_lr": 1e-4,
        "input_size": (380, 380),
        "augmentation": ["RandAugment", "Mixup"],
        "weight_decay": 0,
        "normalization": {
            "mean": (0.485, 0.456, 0.406),
            "std": (0.229, 0.224, 0.225)
        }
    },
    "swin_tiny_patch4_window7_224": {
        "optimizer": "AdamW",
        "head_only_lr": 1e-4,
        "full_fine_tune_lr": 1e-4,
        "input_size": (224, 224),
        "augmentation": ["RandAugment", "Cutmix"],
        "weight_decay": 0,
        "normalization": {
            "mean": (0.485, 0.456, 0.406),
            "std": (0.229, 0.224, 0.225)
        }
    },
    "vit_base_patch32_clip_448": {
        "optimizer": "AdamW",
        "head_only_lr": 1e-4,
        "full_fine_tune_lr": 1e-5,
        "input_size": (448, 448),
        "augmentation": ["RandAugment", "Mixup"],
        "weight_decay": 0.05,
        "normalization": {
            "mean": (0.48145466, 0.4578275, 0.40821073),
            "std": (0.26862954, 0.26130258, 0.27577711)
        }
    }
}

# model_settings = {
#     "efficientnet_b4": {
#         "optimizer": "AdamW",
#         "head_only_lr": 1e-4,
#         "full_fine_tune_lr": 1e-4,
#         "input_size": (448, 448),
#         "augmentation": ["RandAugment", "Mixup"],
#         "weight_decay": 0,
#         "normalization": {
#             "mean": (0.485, 0.456, 0.406),
#             "std": (0.229, 0.224, 0.225)
#         }
#     },
#     "efficientnetv2_rw_m.agc_in1k": {
#         "optimizer": "AdamW",
#         "head_only_lr": 1e-4,
#         "full_fine_tune_lr": 1e-4,
#         "input_size": (448, 448),
#         "augmentation": ["RandAugment", "Cutmix"],
#         "weight_decay": 0,
#         "normalization": {
#             "mean": (0.485, 0.456, 0.406),
#             "std": (0.229, 0.224, 0.225)
#         }
#     },
#     "resnet101": {
#         "optimizer": "AdamW",
#         "head_only_lr": 3e-4,
#         "full_fine_tune_lr": 1e-4,
#         "input_size": (448, 448),
#         "augmentation": ["AutoAugment", "ColorJitter"],
#         "weight_decay": 0,
#         "normalization": {
#             "mean": (0.485, 0.456, 0.406),
#             "std": (0.229, 0.224, 0.225)
#         }
#     },
#     "densenet121": {
#         "optimizer": "AdamW",
#         "head_only_lr": 3e-4,
#         "full_fine_tune_lr": 1e-4,
#         "input_size": (448, 448),
#         "augmentation": ["AutoAugment", "ColorJitter"],
#         "weight_decay": 10,
#         "normalization": {
#             "mean": (0.485, 0.456, 0.406),
#             "std": (0.229, 0.224, 0.225)
#         }
#     },
#     "inception_v3": {
#         "optimizer": "AdamW",
#         "head_only_lr": 1e-4,
#         "full_fine_tune_lr": 1e-4,
#         "input_size": (448, 448),
#         "augmentation": ["RandAugment", "Mixup"],
#         "weight_decay": 0,
#         "normalization": {
#             "mean": (0.5, 0.5, 0.5),
#             "std": (0.5, 0.5, 0.5)
#         }
#     },
#     "swin_tiny_patch4_window7_224": {
#         "optimizer": "AdamW",
#         "head_only_lr": 1e-4,
#         "full_fine_tune_lr": 1e-4,
#         "input_size": (224, 224),
#         "augmentation": ["RandAugment", "Cutmix"],
#         "weight_decay": 0,
#         "normalization": {
#             "mean": (0.485, 0.456, 0.406),
#             "std": (0.229, 0.224, 0.225)
#         }
#     },
#     "vit_base_patch32_clip_448": {
#         "optimizer": "AdamW",
#         "head_only_lr": 1e-4,
#         "full_fine_tune_lr": 1e-5,
#         "input_size": (448, 448),
#         "augmentation": ["RandAugment", "Mixup"],
#         "weight_decay": 0.05,
#         "normalization": {
#             "mean": (0.48145466, 0.4578275, 0.40821073),
#             "std": (0.26862954, 0.26130258, 0.27577711)
#         }
#     }
# }