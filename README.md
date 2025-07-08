# W-MTHN:Wavelet-Driven Mamba-Transformer Hybrid Network for Image Deraining
OUR CODE IS MAINLY BASED ON https://github.com/jeya-maria-jose/TransWeather.git

----

## 🔍 Overview  
 W-MTHN is a novel Transformer-based architecture that synergistically integrates **wavelet preprocessing**, **hybrid Mamba-Transformer blocks**, and **direction-adaptive scanning** to achieve robust rain removal. Our key innovations include:  
✅ **Wavelet Preprocessor**: Enhances high-frequency details via multi-level DWT with adaptive gain control  
✅ **Hierarchical Encoder**: 2MLP + 2Mamba block design for optimal local-global feature extraction  
✅ **Spatial-Mamba Modules**: Direction-adaptive scanning (raster/vertical/spiral/reverse) for dynamic rain pattern modeling  
✅ **Dynamic Cross-Attention**: Top-k gating for efficient cross-stage feature fusion  

<img width="776" alt="0da22dfd08dea39a8c9d2f94ac88a31e" src="https://github.com/user-attachments/assets/c3b7e197-8ed3-4927-9ef5-97b316d83a9b" />

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
    W-MTHN
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







