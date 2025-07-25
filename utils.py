import json
import os
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import timm
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import pickle
from collections import defaultdict
import uuid
from sklearn.model_selection import StratifiedShuffleSplit
import timm
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR
from model_setting import model_settings


label_mapping = {
    "nose-right": 0,
    "nose-left": 1,
    "ear-right": 2,
    "ear-left": 3,
    "vc-open": 4,
    "vc-closed": 5,
    "throat": 6,
}

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # Compute cross-entropy loss
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        # Compute probabilities
        pt = torch.exp(-ce_loss)
        # Apply focal loss formula
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# Custom Dataset class for JSON-based dataset with albumentations
class MedicalImageDataset(Dataset):
    def __init__(self, data, img_size = 448,  type_='train', mosaic=True, mixup = True, model_name=None, model=None):
        self.data = data
        if isinstance(data, dict):
            self.data = [{"model_path": k, "Classification": v} for k, v in data.items()]
        self.label_encoder = LabelEncoder()
        self.labels = [item["Classification"] for item in self.data]
        self.label_encoder.fit(self.labels)
        # self.img_dir = img_dir
        self.transform = None
        self.mosaic = mosaic  # Set to True to enable mosaic augmentation
        self.mixup = mixup  # Set to True to enable mixup augmentation
        self.type = type_
        self.img_size = model_settings[model_name]['input_size'][0] if model_name in model_settings else img_size
        self.model_name = model_name
        self.model = model
        self.normalization = model_settings[model_name]['normalization'] if model_name in model_settings else \
        {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}
        self.transform_fn()
        # self.timm_transform()



    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data[idx]["Path"]
        label = self.data[idx]["Classification"]
        image = Image.open(img_path)
        image = Image.open(img_path).convert("RGB")
        # image_name = self.data[idx]["Path"]
        image = np.array(image)
        # If self.mosaic , then apply mosaic augmentation
        if self.mixup and np.random.rand() < 0.5 and self.type == 'train':
        # if True:
            # print("Mixup")
            image_same_label = [
                item for item in self.data if item["Classification"] == label
            ]
            if len(image_same_label) > 0:
                # randomly select 3 images from the same label
                selected_images = np.random.choice(
                    image_same_label, size=1, replace=False
                )[0]
                selected_image = Image.open(selected_images["Path"]).convert("RGB")
                selected_image = np.array(selected_image)
                # mix the images
                alpha = np.random.beta(0.8, 0.8)
                image = cv2.resize(image, (self.img_size, self.img_size))
                selected_image = cv2.resize(selected_image, (self.img_size, self.img_size))
                image = cv2.addWeighted(
                    image, alpha, selected_image, 1 - alpha, 0
                )
        #mosaic
        if self.mosaic and np.random.rand() < 0.5 and self.type == 'train':
            image_same_label = [
                item for item in self.data if item["Classification"] == label
            ]
            if len(image_same_label) > 0:
                #randomly select 3 images from the same label
                selected_images = np.random.choice(
                    image_same_label, size=3, replace=False
                )
                selected_images = [Image.open(item["Path"]).convert("RGB") for item in selected_images]
                selected_images = [np.array(img) for img in selected_images]
                # Convert to numpy array and stack 2x2 images
                image = np.array(image)
                selected_images = [np.array(img) for img in selected_images]
                images = [image] + list(selected_images)
                split2_size = self.img_size//2
                images = [cv2.resize(img, (split2_size, split2_size)) for img in images]
                images = np.stack(images, axis=0)
                # Create a mosaic image
                mosaic_image = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
                if np.random.rand() < 0.5:
                    mosaic_image[:split2_size, :split2_size] = images[0]
                if np.random.rand() < 0.5:
                    mosaic_image[:split2_size, split2_size:] = images[1]
                if np.random.rand() < 0.5:
                    mosaic_image[split2_size:, :split2_size] = images[2]
                if np.random.rand() < 0.5:
                    mosaic_image[split2_size:, split2_size:] = images[3]
                image = mosaic_image

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]
        
        #flip left and right and change label
        if self.type == 'train' and np.random.rand() < 0.5:
            # print("type image", image.shape)
            image = torch.flip(image, dims=[-1])  # Flip horizontally
            changed_label = label
            if label == "nose-right":
                changed_label = "nose-left"
            elif label == "nose-left":
                changed_label = "nose-right"
            elif label == "ear-right":
                changed_label = "ear-left"
            elif label == "ear-left":
                changed_label = "ear-right"
            label = changed_label
        label = self.label_encoder.transform([label])[0]
        temp_img_path = f"./cache/{uuid.uuid4()}.jpg"
        cv2.imwrite(temp_img_path, np.asarray(image.permute(1, 2, 0) * 255).astype(np.uint8))
        return image, label, img_path

    def transform_fn(self):
        # Albumentations transforms
        if self.type == 'train':
            # transform 96%
            self.transform = A.Compose(
                [
                    A.LongestMaxSize(max_size=self.img_size),  # Resize keeping aspect ratio
                    A.PadIfNeeded(min_height=self.img_size, min_width=self.img_size, border_mode=cv2.BORDER_CONSTANT, value=[0, 0, 0]),
                    A.GaussNoise(var_limit=(10, 50), p=0.5),
                    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
                    A.GaussianBlur(blur_limit=(3, 5), p=0.3),
                    A.RandomBrightnessContrast(p=0.3),
                    A.Affine(
                        scale=(0.8, 1.2), translate_percent=(0.1, 0.1), rotate=0, shear=10, p=0.5
                    ),
                    A.CoarseDropout(max_holes=8, max_height=8, max_width=8, p=0.2),
                    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    # A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0]),
                    ToTensorV2(),
                ]
            )
        else:
            self.transform = A.Compose(
                [
                    A.LongestMaxSize(max_size=self.img_size, interpolation=cv2.INTER_LINEAR),  # Resize keeping aspect ratio
                    A.PadIfNeeded(min_height=self.img_size, min_width=self.img_size, border_mode=cv2.BORDER_CONSTANT, value=[0, 0, 0]),
                    A.Normalize(mean=self.normalization['mean'], std=self.normalization['std']),
                    ToTensorV2(),
                ]
            )
    def get_num_classes(self):
        return len(self.label_encoder.classes_)

def log_transforms(composed_transform):
    log_lines = ["Augmentation Pipeline:"]
    for transform in composed_transform.transforms:
        # Get transform name and parameters
        transform_name = transform.__class__.__name__
        params = transform.get_params() if hasattr(transform, 'get_params') else {}
        param_str = ", ".join([f"{k}={v}" for k, v in transform.__dict__.items() if not k.startswith('_')])
        log_lines.append(f"- {transform_name}: {param_str}")
    return "\n".join(log_lines)

# Function to count model parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

# Function to select last parameters to match target parameter count
def select_last_trainable_parameters(model, model_name, target_params):
    trainable_params = 0
    frozen_params = 0
    total_params = count_parameters(model)
    trainable_param_tensors = []

    # Freeze all parameters initially
    for param in model.parameters():
        param.requires_grad = False
        frozen_params += param.numel()

    # Collect all parameters in reverse order
    all_params = list(model.parameters())[::-1]

    # Unfreeze last parameters to match target_params
    for param in all_params:
        if trainable_params >= target_params:
            break
        param.requires_grad = True
        trainable_params += param.numel()
        frozen_params -= param.numel()
        trainable_param_tensors.append(param)

    # Adjust if overshot target_params
    if trainable_params > target_params:
        excess_params = trainable_params - target_params
        last_tensor = trainable_param_tensors[-1]
        if excess_params < last_tensor.numel():
            # Create a new tensor with only the required number of parameters
            with torch.no_grad():
                flat_tensor = last_tensor.view(-1)
                keep_params = last_tensor.numel() - excess_params
                flat_tensor[keep_params:] = 0  # Zero out excess parameters
                last_tensor.requires_grad = False
                # Create a new trainable tensor with exact number needed
                new_trainable = flat_tensor[:keep_params].detach().requires_grad_(True)
                trainable_params = target_params
                frozen_params = total_params - trainable_params

    return model, frozen_params, trainable_params

# Training function with accuracy tracking
def train_model(
    model,
    model_name,
    train_loader,
    val_loader,
    num_epochs,
    batch_size,
    device,
    frozen_params=0,
    trainable_params=0,
    encoder=None,
    process_folder=""
):
    criterion = nn.CrossEntropyLoss()
    # criterion = FocalLoss(alpha=1, gamma=2)
    # optimizer = optim.Adam(
    #     filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3
    # )
    optimizer = optim.AdamW(model.parameters(), lr=model_settings[model_name]['full_fine_tune_lr'],
        weight_decay= model_settings[model_name]['weight_decay'])
    best_accuracy = 0.0
    best_model_path = f"{process_folder}/best_model_{model_name}_{str(uuid.uuid4())[:8]}.pth"
    old_best_model_path = best_model_path
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs,eta_min=1e-6)
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels, imgs_path in train_loader:
            images, labels= images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        epoch_loss = running_loss / len(train_loader.dataset)
        print(
            f"Model: {model_name}, Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Learning Rate: {current_lr}, Frozen Params: {frozen_params}, Trainable Params: {trainable_params}"
        )

        # Validation
        model.eval()
        correct = 0
        total = 0
        true_labels = []
        predicted_labels = []
        with torch.no_grad():
            for images, labels, imgs_path in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                true_labels.extend(labels.cpu().numpy())
                predicted_labels.extend(predicted.cpu().numpy())
        accuracy = 100 * correct / total
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            #remove old best model if exists
            if os.path.exists(old_best_model_path):
                os.remove(old_best_model_path)
            old_best_model_path = best_model_path
            best_model_path = f"{process_folder}/best_model_{model_name}_{str(uuid.uuid4())[:8]}_{str(accuracy)[:5]}.pth"
            torch.save(model.state_dict(), best_model_path)
            print(f"Model: {model_name}, New best accuracy: {best_accuracy:.2f}%")
                    #build confusion matrix
            from sklearn.metrics import confusion_matrix
            cm = confusion_matrix(true_labels, predicted_labels)
            print(f"Confusion Matrix for {model_name}:\n{cm}")
            #save confusion matrix by matplotlib'
            import matplotlib.pyplot as plt
            import seaborn as sns
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels= encoder.inverse_transform(np.arange(len(encoder.classes_))),
                            yticklabels= encoder.inverse_transform(np.arange(len(encoder.classes_))))

            plt.title(f"Confusion Matrix for {model_name}")
            plt.xlabel("Predicted")
            plt.ylabel("True")
            plt.savefig(f"{process_folder}/confusion_matrix_{model_name}.png")
            plt.close()


        print(f"Model: {model_name}, Validation Accuracy: {accuracy:.2f}%")

    return best_accuracy, best_model_path

def load_data_fn(model,
                model_name,
                process_folder,
                batch_size = 64,
                split_ratio = 0.5,
                json_paths = [],
                img_dirs = []):
    """ modified load_data_fn1 to use json_paths and img_dirs as input parameters """
    if not os.path.exists(process_folder):
        os.makedirs(process_folder, exist_ok=True)
    train_data = []
    val_data = []
    real_data_train_ratio = split_ratio  # 50% real data for training, 50% for validation
    for json_path, img_dir in zip(json_paths, img_dirs):
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"JSON file not found: {json_path}")
        if not os.path.exists(img_dir):
            raise FileNotFoundError(f"Image directory not found: {img_dir}")
        with open(json_path, "r") as f:
            data = json.load(f)
            if isinstance(data, dict):
                data = [{"Path": os.path.join(img_dir, k), "Classification": v} for k, v in data.items()]
            elif isinstance(data, list):
                data = [{"Path": os.path.join(img_dir, item["Path"]), "Classification": item["Classification"]} for item in data]
            else:
                raise ValueError("Unsupported JSON format. Expected a list or dictionary.")
        # Split data into train and validation sets
        labels = [item["Classification"] for item in data]
        indices = np.arange(len(data))
        sss = StratifiedShuffleSplit(n_splits=1, train_size=0.7, random_state=np.random.randint(0, 100))
        for train_index, val_index in sss.split(indices, labels):
            train_indices = indices[train_index]
            val_indices = indices[val_index]
        train_data.extend([data[i] for i in train_indices])
        val_data.extend([data[i] for i in val_indices])
    # Create datasets
    train_dataset = MedicalImageDataset(train_data, type_='train', model_name=model_name, model=model)
    val_dataset = MedicalImageDataset(val_data, type_='val', mosaic=False, model_name=model_name, model=model)
    # Save train and validation data to JSON files
    with open(f"{process_folder}/train_data_{model_name}.json", "w") as f:
        json.dump(train_data, f, indent=4)
    with open(f"{process_folder}/val_data_{model_name}.json", "w") as f:
        json.dump(val_data, f, indent=4)
    # DataLoaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=4
    )
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with_generatedata = False
    misclassification_data = None  # Placeholder for misclassification data if needed
    return train_loader, val_loader, val_dataset, train_dataset, device, real_data_train_ratio, with_generatedata, misclassification_data

# Main script
def train(process_folder = None,
          model_configs=[],
          data_json_paths=[],
          img_dirs=[],
          num_epochs=200,
          batch_size=64,
          split_ratio=0.5):
    # List of models from different papers

    # Track performance
    import time
    if process_folder is None:
        process_folder = str(time.strftime("%Y%m%d_%H%M%S"))
    else:
        if not os.path.exists(process_folder):
            os.makedirs(process_folder, exist_ok=True)
    model_performance = []
    num_classes = 7 

    # Train each model
    for config in model_configs:
        model = timm.create_model(
            config["name"], pretrained=True, num_classes=7
        )
        train_loader,\
        val_loader,\
        test_dataset,\
        train_dataset,\
        device,\
        real_data_train_ratio,\
        with_generatedata,\
        misclassification_data = load_data_fn(
            model, config["name"],
            process_folder=process_folder, 
            json_paths=data_json_paths,
            batch_size=batch_size,
            img_dirs=img_dirs,
            split_ratio = split_ratio
        )
        # train_loader, val_loader, test_dataset, train_dataset, device, num_epochs, real_data_train_ratio, with_generatedata, misclassification_data = load_data_fn2(model, config["name"], process_folder=process_folder)
        total_params = count_parameters(model)
        print(f"Model {config['name']} total parameters: {total_params}")

        frozen_params = 0
        trainable_params = total_params
        model = model.to(device)
        # frozen_params = 0
        # trainable_params = total_params
        best_accuracy, best_model_path = train_model(
            model,
            config["name"],
            train_loader,
            val_loader,
            num_epochs,
            batch_size,
            device,
            frozen_params,
            trainable_params,
            train_dataset.label_encoder,
            process_folder = process_folder
        )
        model_performance.append(
            {
                "name": config["name"],
                "paper": config["paper"],
                "accuracy": best_accuracy,
                "model_path": best_model_path,
                "frozen_params": frozen_params,
                "trainable_params": trainable_params,
                "train_ratio": real_data_train_ratio,
                "with_generatedata": with_generatedata,
                "transform": log_transforms(train_dataset.transform),
                "include_misclassification": True if misclassification_data else False,
            }
        )

    # Select top 5 models from different papers
    paper_to_best_model = defaultdict(lambda: {"accuracy": 0, "model_info": None})
    for perf in model_performance:
        paper = perf["paper"]
        if perf["accuracy"] > paper_to_best_model[paper]["accuracy"]:
            paper_to_best_model[paper] = {
                "accuracy": perf["accuracy"],
                "model_info": perf,
            }

    # Get top 5 models
    top_models = sorted(
        paper_to_best_model.values(), key=lambda x: x["accuracy"], reverse=True
    )

    # Print results
    print("\nTop models from different papers:")
    for i, model in enumerate(top_models, 1):
        info = model["model_info"]
        print(
            f"{i}. Model: {info['name']}, Paper: {info['paper']}, Accuracy: {info['accuracy']:.2f}%, Frozen Params: {info['frozen_params']}, Trainable Params: {info['trainable_params']}, Path: {info['model_path']}"
        )
    
    with open(f"{process_folder}/label_encoder.pkl", "wb") as f:
        pickle.dump(train_dataset.label_encoder, f)

    # Save top models info
    from time import time
    time_ = time()
    with open(f"{process_folder}/top_models_{time_}.json", "w") as f:
        json.dump([model["model_info"] for model in top_models], f, indent=4)

def test(model_name,model_path):
    # Load label encoder
    with open("label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)
    img_path = "/raid/hvtham/son/data/PublicTest"
    with open("/raid/hvtham/son/ent/cls.csv", "r") as f:
        test_images_name = f.readlines()
    test_images_name = [line.strip() for line in test_images_name if line.strip() != ""]
    test_images = []
    model = timm.create_model(
        model_name, pretrained=False, num_classes=7
    )
    model.load_state_dict(torch.load(model_path))
    model = model.to("cuda" if torch.cuda.is_available() else "cpu")
    for image_name in tqdm(test_images_name):
        image_path = os.path.join(img_path, image_name)
        image = Image.open(image_path).convert("RGB")
        # Apply the same transformations as validation
        image = np.array(image)
        transform = A.Compose(
            [
                # A.Resize(width=380, height=380, p=1.0),
                # A.LongestMaxSize(max_size=448, area_for_downscale="image"),
                # A.PadIfNeeded(min_height=448, min_width=448),
                A.LongestMaxSize(max_size=380, interpolation=cv2.INTER_AREA),  # Resize keeping aspect ratio
                A.PadIfNeeded(min_height=380, min_width= 380, border_mode=cv2.BORDER_CONSTANT, value=[0, 0, 0]),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ]
        )
        augmented = transform(image=image)
        image = augmented["image"]
        test_images.append((image, image_name))
    output_json = {}
    for image, image_name in test_images:
        image = image.unsqueeze(0)
        image = image.to("cuda" if torch.cuda.is_available() else "cpu")
        model.eval()
        with torch.no_grad():
            outputs = model(image)
            _, predicted = torch.max(outputs, 1)
            output_json[image_name] = label_mapping[
                label_encoder.inverse_transform([predicted.item()])[0]
            ]
    # Save output json
    with open("submit.json", "w") as f:
        json.dump(output_json, f, indent=4)

# Function load checkpoint, evaluate model and save false cases to new json file for retraining
def val(models = [], img_dirs = [], json_paths = []):
    # Load label encoder
    with open("label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)
    test_images = []
    val_output = {}
    # real_data_path = "/raid/hvtham/son/data/ent/data.json"
    real_data = []
    for json_path, img_dir in zip(json_paths, img_dirs):
        with open(json_path, "r") as f:
            real_data_ = json.load(f)
            if isinstance(real_data_, dict):
                real_data_ = [{"Path": os.path.join(img_dir, k), "Classification": v} for k, v in real_data_.items()]
            elif isinstance(real_data_, list):
                real_data_ = [{"Path": os.path.join(img_dir, item["Path"]), "Classification": item["Classification"]} for item in real_data_]
            else:
                raise ValueError("Unsupported JSON format. Expected a list or dictionary.")
            real_data.extend(real_data_)
    accuracy = 0
    loaded_models_dict = [
        (timm.create_model(
            model_info["name"], pretrained=False, num_classes=7
        ).to("cuda"), model_info["model_path"])
        for model_info in models
    ]
    model_outputs = {}
    for (model, model_path), model_info in tqdm(zip(loaded_models_dict, models)):
        print(f"Processing model {model_info['name']} from {model_path}")
        model.load_state_dict(torch.load(model_path))
        model.eval()
        image_size = model_settings[model_info["name"]]["input_size"][0]
        normalization = model_settings[model_info["name"]]["normalization"]
        transform = A.Compose(
            [
                # A.Resize(width=image_size, height=image_size, p=1.0), #95 best
                A.LongestMaxSize(max_size=image_size, area_for_downscale="image", interpolation=cv2.INTER_AREA),
                A.PadIfNeeded(min_height=image_size, min_width=image_size, border_mode=cv2.BORDER_CONSTANT, value=[0, 0, 0]),
                # A.PadIfNeeded(min_height=380, min_width=380),
                A.Normalize(mean=normalization["mean"], std=normalization["std"]),
                ToTensorV2(),
            ]
        )
        #infer
        for item in real_data:
            image_path = item["Path"]
            image_name = os.path.basename(image_path)
            image = Image.open(image_path).convert("RGB")
            # Apply the same transformations as validation
            image = np.array(image)
            augmented = transform(image=image)
            image = augmented["image"]
            output = image.unsqueeze(0).to("cuda")
            with torch.no_grad():
                model.eval()
                output = model(output)
                if image_name not in model_outputs:
                    model_outputs[image_path] = []
                model_outputs[image_path].append(output.cpu().numpy())
    #accuracy for val set
    for item in tqdm(real_data):
        image_name = os.path.basename(item["Path"])
        outputs = np.mean(model_outputs[item["Path"]], axis=0)
        predicted = np.argmax(outputs, axis=1)
        val_output[image_name] = label_encoder.inverse_transform([predicted.item()])[0]
        if val_output[image_name] == item["Classification"]:
            accuracy += 1
    accuracy = accuracy / len(real_data) * 100
    print(f"Accuracy for assembled model: {accuracy:.2f}%")

def assemble(models = [], csv_path = None, images_path = None):
    # Load label encoder
    with open("label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)
    img_path = images_path
    with open(csv_path, "r") as f:
        test_images_name = f.readlines()
    test_images_name = [line.strip() for line in test_images_name if line.strip() != ""]
    test_images = []
    val_output = {}
    loaded_models_dict = [
        (timm.create_model(
            model_info["name"], pretrained=False, num_classes=7
        ).to("cuda"), model_info["model_path"])
        for model_info in models
    ]
    model_outputs = {}
    output_json = {}
    model_outputs = {}
    for (model, model_path), model_info in tqdm(zip(loaded_models_dict, models)):
        model.load_state_dict(torch.load(model_path))
        model.eval()
        model_name = model_info["name"]
        image_size = model_settings[model_name]["input_size"][0]
        normalization = model_settings[model_name]["normalization"]
        transform = A.Compose(
            [
                A.LongestMaxSize(max_size=image_size, area_for_downscale="image", interpolation=cv2.INTER_AREA),
                A.PadIfNeeded(min_height=image_size, min_width=image_size, border_mode=cv2.BORDER_CONSTANT),
                A.Normalize(mean=normalization["mean"], std=normalization["std"]),
                ToTensorV2(),
            ]
        )
        # model_outputs
        for image_name in tqdm(test_images_name):
            image_path = os.path.join(img_path, image_name)
            image = Image.open(image_path).convert("RGB")
            # Apply the same transformations as validation
            image = np.array(image)
            augmented = transform(image=image)
            image = augmented["image"]
            output = image.unsqueeze(0).to("cuda")
            with torch.no_grad():
                model.eval()
                output = model(output)
                if image_name not in model_outputs:
                    model_outputs[image_name] = []
                model_outputs[image_name].append(output.cpu().numpy())
    #assemble outputs
    for image_name in test_images_name:
        outputs = np.mean(model_outputs[image_name], axis=0)
        predicted = np.argmax(outputs, axis=1)
        output_json[image_name] = label_mapping[
            label_encoder.inverse_transform([predicted.item()])[0]
        ]

    with open("submission.json", "w") as f:
        json.dump(output_json, f, indent=4)
