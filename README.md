# GSANet
This project provides the code and results for 'Global Semantic-Sense Aggregation Network for Salient Object Detection in Remote Sensing Images'

# Saliency maps
   We provide saliency maps of our GSANet and all compared methods at [GSANet_SalMap and ComparisonMethods](https://pan.baidu.com/s/1Xp8TSt1UBiaKwQrGjgJtDg?pwd=o4dz) (code:o4dz) on two datasets (ORSSD and EORSSD).
      
# DateSets
ORSSD download  https://github.com/rmcong/ORSSD-dataset

EORSSD download https://github.com/rmcong/EORSSD-dataset

# Training
   We use data_aug.m for data augmentation.

   Download [uniformer_base_ls_in1k.pth](https://pan.baidu.com/s/1Xp8TSt1UBiaKwQrGjgJtDg?pwd=o4dz) (code: o4dz), and put it in './pretrain/'. 

   Modify paths of datasets, then run train_MyNet.py.


# testing
1. Download the pre-trained models (MyNet_EORSSD.pth and MyNet_ORSSD.pth) on [GSAMet_Pretrain](https://pan.baidu.com/s/1Xp8TSt1UBiaKwQrGjgJtDg?pwd=o4dz) (code:o4dz) .
2. Modify paths of pre-trained models  and datasets (EORSSD and ORSSD).
3. Run test_MyNet.py.


   
# Evaluation Tool
   You can use the [evaluation tool (MATLAB version)](https://github.com/MathLee/MatlabEvaluationTools) to evaluate the above saliency maps.


# [ORSI-SOD_Summary](https://github.com/MathLee/ORSI-SOD_Summary)
   
# Citation
       
                
                

