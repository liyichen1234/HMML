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

### Code Smell Detection

 1. run `python train_HMML.py` or `python test_HMML.py` to train and test our model.
 2. `cd baseline`
 3. run `python train.py` or `python random-forest.py` to get the baseline model result.
### How to use it on your own dataset
Since our dataset uses Codesplit and Designite tool to analyze the code and get code smells, you can use other tools like SonarQube to analyze other Java code.
1. `Codesplit: https://github.com/tushartushar/CodeSplitJava`
2. `Designite: https://www.designite-tools.com/`
3. `SonarQube: https://rules.sonarsource.com/java/type/Code%20Smell`

### Citation
  If you find this code useful in your research, please, consider citing our paper:
  > 
