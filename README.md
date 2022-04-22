# HMML--Multi-Label Code Smell Detection with Hybrid Model based on Deep Learning
This repository includes the code and experimental data in our paper entitled "Multi-Label Code Smell Detection with Hybrid Model based on Deep Learning" published in SEKE'2022. It can be used to detect the appropriate code smell in Java code. 

### Requirements
+ python 3.8.10<br>
+ tqdm 4.62.3<br>
+ anytree 2.8.0<br>
+ torch_geometric 2.0.2<br>
+ pytorch 1.10.1<br> 
+ javalang 0.13.0<br>
+ RAM 16GB or more
+ GPU with CUDA support is also needed
+ BATCH_SIZE should be configured based on the GPU memory size

### How to install
Install all the dependent packages via pip:

	$ pip install tqdm==4.62.3 anytree==2.8.0 torch_geometric==2.0.2 javalang==0.13.0
 
Install pytorch according to your environment, see https://pytorch.org/ 


### Prepare for Dataset
1. `cd astnn`
2. run `python pipeline.py` to generate preprocessed data.
3. run `python train.py` for training and evaluation

### Code Smell Detection

 1. `cd clone`
 2. run `python pipeline.py --lang c` or `python pipeline.py --lang java` to generate preprocessed data for the two datasets.
 2. run `python train.py --lang c` to train on OJClone, `python train.py --lang java` on BigCLoneBench respectively.

### How to use it on your own dataset

Please refer to the `pkl` files in the corresponding directories of the two tasks. These files can be loaded by `pandas`.
 
### Citation
  If you find this code useful in your research, please, consider citing our paper:
  > 
