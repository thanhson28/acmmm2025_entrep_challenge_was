from utils import train

if __name__ == "__main__":
    model_configs = [
        {
            "name": "efficientnet_b4",
            "paper": "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks",
        },
        {
            "name": "efficientnetv2_rw_m.agc_in1k",
            "paper": "EfficientNetV2: Smaller Models and Faster Training",
        },
        {"name": "resnet101", "paper": "Deep Residual Learning for Image Recognition"},
        {"name": "densenet121", "paper": "Densely Connected Convolutional Networks"},
        {"name": "inception_v3", "paper": "Rethinking the Inception Architecture for Computer Vision"},
    ]

    data_json_paths = ['/raid/hvtham/son/data/ent/data.json',
                       '/raid/hvtham/son/data/ent/train/cls.json']

    img_dirs = ['/raid/hvtham/son/data/ent/images',
                '/raid/hvtham/son/data/ent/train/imgs']

    train(model_configs = model_configs,
          data_json_paths=data_json_paths,
          img_dirs=img_dirs,
          num_epochs=200,
          split_ratio=0.5,
          batch_size=64)