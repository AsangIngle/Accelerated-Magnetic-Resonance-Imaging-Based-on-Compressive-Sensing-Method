README.md – ViT-Based MRI Denoising (Cartesian & Radial Sampling)
📌 Project Overview

This project implements a Vision Transformer (ViT) + CNN hybrid denoising model for reconstructing undersampled MRI images.
The model removes sampling-pattern–based artifacts from Cartesian and Radial undersampled MRI data (10% and 20% sampling).

It uses:

CNN Encoder–Decoder backbone

6-layer Transformer bottleneck

Patch embeddings + trainable positional embeddings

AveragePooling-based encoder

Upsampling with ConvTranspose

The goal is to outperform plain CNN/U-Net baselines and stabilize reconstruction across different sampling masks.

📂 Dataset

You used two types of data:

1. Input (Masked / Undersampled Images)
/mul_dataset/cart_mask_10_mul_images
/mul_dataset/cart_mask_20_mul_images
/mul_dataset/radial_10_mul_images
/mul_dataset/radial_20_mul_images

2. Ground Truth (Fully Sampled Images)
/dataset/PD_Data/Training_images
/dataset/PD_Data/Test_images
/dataset/PD_Data/Valid_images


All images are grayscale PNGs resized to 256×256.

🧠 Model Architecture

The model follows a CNN Encoder → ViT Bottleneck → CNN Decoder pipeline:

Encoder

Three CNN blocks (64 → 128 → 256 filters)

AveragePooling (not MaxPooling)

Transformer Bottleneck

Patch size = 8

Projection dim = 64

6 Transformer layers

4 Attention heads

Decoder

Three upsampling blocks with ConvTranspose

Skip connections from encoder

Final Conv2D produces 1-channel normalized output

🚀 Training Setup
Component	Value
Image Size	256×256
Batch Size	8
Epochs	150
Optimizer	Adam
Learning Rate	1e-4
Loss	MSE
Metrics	MAE, PSNR, SSIM, MSE

Callbacks:

EarlyStopping

ReduceLROnPlateau

ModelCheckpoint

📈 Quantitative Results
📌 Cartesian 10%
MAE  = 0.016802  
MSE  = 0.001009  
PSNR = 30.16 dB  
SSIM = 0.8700

📌 Cartesian 20%
MAE  = 0.013044  
MSE  = 0.000612  
PSNR = 32.32 dB  
SSIM = 0.9053

📌 Radial 10%
MAE  = 0.017837  
MSE  = 0.001143  
PSNR = 29.61 dB  
SSIM = 0.8543

📌 Radial 20%
MAE  = 0.014063  
MSE  = 0.000659  
PSNR = 31.97 dB  
SSIM = 0.8907


Summary:

20% sampling significantly boosts PSNR/SSIM.

Cartesian sampling consistently outperforms radial due to lower streaking artifacts.

The ViT bottleneck stabilizes reconstruction quality across masks.

🔍 Inference Example

After training:

idx = 10
masked = X_test_data[idx]
gt     = X_test_gt_data[idx]
pred   = y_pred[idx]


Images are visualized side by side.

🛠 Installation & Requirements

Install dependencies:

pip install tensorflow tqdm opencv-python matplotlib


Google Colab environment is recommended.

▶️ Training the Model
vit_denoiser.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=[checkpoint, reduce_lr, es]
)

📦 File Structure
project/
│── model.py
│── train.py
│── utils.py
│── README.md
│── dataset/
│   ├── PD_Data/
│   └── mul_dataset/
│── saved_models/
└── results/

📌 Key Features

Fully custom Vision Transformer encoder (not timm, not pre-trained)

Works on MRI undersampled reconstructions

Supports multiple sampling patterns

Produces stable reconstructions with high SSIM

Minimal preprocessing

Achieves 30–32 dB PSNR on 256×256 MRI data
