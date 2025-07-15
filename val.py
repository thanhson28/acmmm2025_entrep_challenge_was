from utils import val

if __name__ == '__main__':
    models = [
            {
                "name": "efficientnet_b4",
                "paper": "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks",
                "accuracy": 94.18729817007535,
                "model_path": "/raid/hvtham/son/ent/mixup0.2_mosaic1_transform2_split55/best_model_efficientnet_b4_b206a48f_92.68.pth",
            },
            {
                "name": "efficientnetv2_rw_m.agc_in1k",
                "paper": "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks",
                "accuracy": 94.18729817007535,
                "model_path": "/raid/hvtham/son/ent/mixup0.2_mosaic1_transform2_split55/best_model_efficientnetv2_rw_m.agc_in1k_4cf6b3fe_94.18.pth",
            },
            {
                "name": "densenet121",
                "paper": "Densely Connected Convolutional Networks",
                "accuracy": 93.64908503767492,
                "model_path": "/raid/hvtham/son/ent/mixup0.2_mosaic1_transform2_split55/best_model_densenet121_8861c830_93.64.pth",
            },
            {
                "name": "inception_v3",
                "paper": "inception_v3",
                "accuracy": 93.64908503767492,
                "model_path": "/raid/hvtham/son/ent/mixup0.2_mosaic1_transform2_split55/best_model_inception_v3_ea918bc3_93.64.pth",
            },
            {
                "name": "resnet101",
                "paper": "Deep Residual Learning for Image Recognition",
                "accuracy": 93.1108719052745,
                "model_path": "/raid/hvtham/son/ent/mixup0.2_mosaic1_transform2_split55/best_model_resnet101_1f252ac0_93.11.pth",
            },
            {
                "name": "efficientnet_b4",
                "model_path": "/raid/hvtham/son/ent/20250613_235512/best_model_efficientnet_b4_1be0fc2e_94.62.pth",
            },
            {
                "name": "efficientnetv2_rw_m.agc_in1k",
                "model_path": "/raid/hvtham/son/ent/20250613_235512/best_model_efficientnetv2_rw_m.agc_in1k_afe87641_96.05.pth",
            },
            {
                "name": "resnet101",
                "model_path": "/raid/hvtham/son/ent/20250613_235512/best_model_resnet101_de0dce66_94.44.pth",
            },
            {
                "name": "densenet121",
                "model_path": "/raid/hvtham/son/ent/20250613_235512/best_model_densenet121_0c8d66c1_95.51.pth",
            },
            {
                "name": "inception_v3",
                "model_path": "/raid/hvtham/son/ent/20250613_235512/best_model_inception_v3_e16d0c24_95.16.pth",
            },
            {
                "name": "efficientnet_b4",
                "model_path": "/raid/hvtham/son/ent/mosaic1rand_transform2_split73/best_model_efficientnet_b4_eb5ad184_93.54.pth"},
            {
                "name": "inception_v3",
                "model_path": "/raid/hvtham/son/ent/mosaic1rand_transform2_split55/best_model_inception_v3_85eefd51_94.29.pth",
            },
            {
                "name": "efficientnetv2_rw_m.agc_in1k",
                "paper": "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks",
                "accuracy": 93.8643702906351,
                "model_path": "/raid/hvtham/son/ent/mosaic1rand_transform2_split55/best_model_efficientnetv2_rw_m.agc_in1k_4dfc5a81_93.86.pth",
            },
            {
                "name": "densenet121",
                "model_path": "/raid/hvtham/son/ent/mosaic1rand_transform2_split55/best_model_densenet121_8ef5aea5_93.54.pth",
            },
            {
                "name": "resnet101",
                "model_path": "/raid/hvtham/son/ent/mosaic1rand_transform2_split55/best_model_resnet101_56199572_93.32.pth",
            }
        ]
    json_paths = ['/raid/hvtham/son/data/ent/data.json',
                       '/raid/hvtham/son/data/ent/train/cls.json']

    img_dirs = ['/raid/hvtham/son/data/ent/images',
                '/raid/hvtham/son/data/ent/train/imgs']
                
    val(models=models,
        img_dirs=img_dirs,
        json_paths=json_paths
    )