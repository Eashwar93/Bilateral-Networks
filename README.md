# Realtime-SemanticSegmentation
PyTorch package for Sota Realtime-SemanticSegmentation(2021)
### Tested with:
* Linux Ubuntu 18.04
* Intel 9th gen CPU
* Nvidia RTX 2070
* Pytorch 1.7
* CUDA 11.0

### Environment Setup:
1. Create a virtual environment either using venv or [anaconda](https://docs.anaconda.com/anaconda/install/index.html) and activate the same
```bash   
conda create -n pytorch python=3.6
conda activate pytorch
```
2. Install PyTorch from the [official page](https://pytorch.org/) 
3. Install OpenCV
```bash   
pip install opencv-contrib-python
```
4. Install supportting python libs
```bash   
pip install prettytable
pip install ptflops
pip install tensorboard
pip install tabulate
pip install pytorch-model-summary
pip install torchviz
```

### Dataset preparation:
Please prepare the training testing and evaluation data using [dataset_creation](https://github.com/Eashwar93/Datasetcreator) repo or use standard cityscapes. 



### Train the model:
Use the following command to train the network
```bash
python -m torch.distributed.launch --nproc_per_node=1 train/train.py --model <model of choice>
```
You can find the list of models available in the configs folder to parse the argument ```<model of choice>```

Use the config to adjust training hyperparameters ander configuration. Below is a sample config file
```python
model_type='bisenet_v1_g1',  #name of the model
num_aux_heads=2, #auxilary head for training
aux_output=True, #Enable auxillary heads during traing
lr_start=1e-2, #Initial larning rate of the model
lr_multiplier=10, #Learning rate multiplier for certain parameters
weight_decay=5e-4, #Weight decay for regularisation
warmup_iters=200, #Number of warmup itration to reach the starting learning rate
max_iter=12000, #Total number of training iteration
im_root='./datasets/Rexroth', #Relative path to the root directory of dataset
train_im_anns='./datasets/Rexroth/train.txt', #Relative path to the train data annotation file from dataset_creation repo 
val_im_anns='./datasets/Rexroth/eval.txt', #Relative path to the evaluation data annotation file from dataset_creation repo
scales=[0.5, 2.0], # Scaling of images for dataset augmentation during training
cropsize=[480, 640], # Size of Image to be used for training in H x W 
ims_per_gpu=16, # Batch size of the training
use_fp16=True, # Enable FP16 during training to reduce train time
use_sync_bn=False, # Enable when used to train in a Multi-GPU environment
respth='./res', # Save path of the trained model
categories=4, # Number of semantic classes in the dataset
save_name='bisenet_v1_g1_16_fulldata.pth', # Name in which the PyTorch model will be saved in Save path
```

### Test the model:
Use the following command to test the model on a test image
```bash
python test/demo.py --model <model of choice> --weight-path <path/to/coressponding/.pth file> --img-path <path/to/test/image>
```
Please edit the source code in ```demo.py``` if you wish to create the segmented output with a specific name

### Export the model:
Often it is necessary to export the trained PyTorch model to ONNX to port it to different Inference Engines.

Use the following command to convert the trained pytorch model to ONNX:
```bash
python util/export_onnx.py --model <model of choice> --weight-path <path/to/coressponding/.pth file> --outpath <path/to/output/icluding/filename.onnx>
```

### Reference:
* [CoinCheung/BiSeNet](https://github.com/CoinCheung/BiSeNet)
* [feinanshan/FANet](https://github.com/feinanshan/FANet)
* Yu, Changqian & Wang, Jingbo & Peng, Chao & Gao, Changxin & Yu, Gang & Sang, Nong. (2018). BiSeNet: Bilateral Segmentation Network for Real-Time Semantic Segmentation
* Hu, Ping & Perazzi, Federico & Heilbron, Fabian & Wang, Oliver & Lin, Zhe & Saenko, Kate & Sclaroff, Stan. (2020). Real-Time Semantic Segmentation With Fast Attention
* Xie, Saining & Girshick, Ross & Dollár, Piotr & Tu, Z. & He, Kaiming. (2016). Aggregated Residual Transformations for Deep Neural Networks. 
* Hu, Jie & Shen, Li & Sun, Gang & Albanie, Samuel. (2017). Squeeze-and-Excitation Networks. 


### Recommendations:
Based on the custom dataset I conducted my experiments, I would recommend using _bisnet_v1_g6_ which is _20%_ faster than the author's version without significant change in quality of performance.

### Deployment Pipeline:
To deploy the networks trained and evaluated here in TensorRT and OpenVino use the repository [here](https://github.com/Eashwar93/SemanticSegmentation-Deployments)

### ROS2 Wrapper:
To test the deployment in TensorRT and OpenVino with the ROS2 Framework use the repository [here](https://github.com/Eashwar93/SemanticSegmentation-Deployments/tree/main/ros2_wrapper)


### Further Work:
* Models _bisenetv1_ and _fanetv1_ are author's version of the models from the respective papers. I will publish the details of the modifications and results which were part of my research soon.

