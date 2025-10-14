# Legacy Code 
This folder contains archived code from the initial development phase. 
# Why These Files Exist 
During initial development, training was attempted on local hardware (PyCharm + CPU). However, this proved impractical (due to the following reasons): 
- **Training speed**: ~10 minutes per epoch 
- **Estimated time for 200 epochs**: ~33 hours 
- **GPU unavailable**: No CUDA-capable hardware 
# Migration to Google Colab 
Training was migrated to Google Colab for access to free T4 GPUs, thus improving the aforementioned statistics to: 
- **Training speed**: ~3.5 minutes per epoch 
- **Total time for 200 epochs**: ~12 hours  
# Files 
- `train.py`: Original CPU-based training script 
- `predict.py`: First version of prediction script (hardcoded paths)
# Other Notes
The dataset that was used for these legacy files will NOT be included in this directory; I used all 134 annotated images in conjunction with the VisDrone dataset that is currently being used for training.
