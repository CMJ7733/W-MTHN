# WMFormer:A Wavelet-Enhanced Mamba-Transformer Architecture for Image Deraining
OUR CODE IS MAINLY BASED ON https://github.com/jeya-maria-jose/TransWeather.git

----

## 🔍 Overview  
WEMT-Derain is a novel Transformer-based architecture that synergistically integrates **wavelet preprocessing**, **hybrid Mamba-Transformer blocks**, and **direction-adaptive scanning** to achieve robust rain removal. Our key innovations include:  
✅ **Wavelet Preprocessor**: Enhances high-frequency details via multi-level DWT with adaptive gain control  
✅ **Hierarchical Encoder**: 2MLP + 2Mamba block design for optimal local-global feature extraction  
✅ **Spatial-Mamba Modules**: Direction-adaptive scanning (raster/vertical/spiral/reverse) for dynamic rain pattern modeling  
✅ **Dynamic Cross-Attention**: Top-k gating for efficient cross-stage feature fusion  

<img width="776" alt="12514301c4dc930eea19a5bc774c5d5" src="https://github.com/user-attachments/assets/fb55a1ed-1eb1-4c64-8314-14716961f0e9" />

----

## 📊 Performance Highlights  
| Dataset     | PSNR   | SSIM    |
|-------------|--------|---------|
| **Raindrop**| 29.96  | 0.9126  | 
| **Rain100H**| 28.19  | 0.9452  |

----

## ⚙️ Installation
    ```bash
    conda env create -f environment.yml
    conda activate WMFormer

----

## Dataset format
    WEMT-NET
    ├── data 
    |   ├── train # Training  
    |   |   ├── <dataset_name>   
    |   |   |   ├── input         # rain images 
    |   |   |   └── gt            # clean images
    |   |   └── dataset_filename.txt
    |   └── test  # Testing         
    |   |   ├── <dataset_name>          
    |   |   |   ├── input         # rain images 
    |   |   |   └── gt            # clean images
    |   |   └── dataset_filename.txt

----

## Training Command
    ```bash
    python train-individual.py -exp_name wmf \
                --train_batch_size 32 \
                -epoch_start 0 \
                -num_epochs 200
                
## Evaluation
    ```bash
    #FOR Raindrop
    ->  python test_raindrop.py -exp_name wmf
    #FOR Rain100H
    ->  python test_rain100H.py -exp_name wmf







