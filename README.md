# Global Semantic-Sense Aggregation Network for Salient Object Detection in Remote Sensing Images

⭐ This code has been completely released ⭐ 

⭐ our [article](https://www.mdpi.com/1099-4300/26/6/445) ⭐ 

# 📖 Introduction
<span style="font-size: 125%">Salient object detection (SOD) aims to accurately identify significant geographical 
objects in remote sensing images (RSI), providing reliable support and guidance for extensive geographical information 
analyses and decisions. However, SOD in RSI faces numerous challenges, including shadow interference, inter-class 
feature confusion, as well as unclear target edge contours. Therefore, we designed an effective Global Semantic-aware 
Aggregation Network (GSANet) to aggregate salient information in RSI. GSANet computes the information entropy of different 
regions, prioritizing areas with high information entropy as potential target regions, thereby achieving precise localization 
and semantic understanding of salient objects in remote sensing imagery. Specifically, we proposed a Semantic Detail 
Embedding Module (SDEM), which explores the potential connections among multi-level features, adaptively fusing shallow texture details 
with deep semantic features, efficiently aggregating the information entropy of salient regions, enhancing information content of 
salient targets. Additionally, we proposed a Semantic Perception Fusion Module (SPFM) to analyze map relationships between contextual 
information and local details, enhancing the perceptual capability for salient objects while suppressing irrelevant information entropy, 
thereby addressing the semantic dilution issue of salient objects during the up-sampling process. The experimental results on two publicly 
available datasets, ORSSD and EORSSD, demonstrated the outstanding performance of our method. The method achieved 93.91% Sα, 98.36% Eξ, 
and 89.37% Fβ on the EORSSD dataset.</span>
<p align="center"> <img src="Images/Network.png" width=90%"></p>

If our code is helpful to you, please cite:

```

```
# Saliency maps
   We provide saliency maps of our and compared methods at [here](https://pan.baidu.com/s/1Xp8TSt1UBiaKwQrGjgJtDg?pwd=o4dz) (code:o4dz) on two datasets (ORSSD and EORSSD).
      
# DateSets
ORSSD download  at [here](ttps://github.com/rmcong/ORSSD-dataset)

EORSSD download at [here](https://github.com/rmcong/EORSSD-dataset)

The structure of the dataset is as follows:
```python
GSANet
├── EORSSD
│   ├── train
│   │   ├── images
│   │   │   ├── 0001.jpg
│   │   │   ├── 0002.jpg
│   │   │   ├── .....
│   │   ├── lables
│   │   │   ├── 0001.png
│   │   │   ├── 0002.png
│   │   │   ├── .....
│   │   
│   ├── test
│   │   ├── images
│   │   │   ├── 0004.jpg
│   │   │   ├── 0005.jpg
│   │   │   ├── .....
│   │   ├── lables
│   │   │   ├── 0004.png
│   │   │   ├── 0005.png
│   │   │   ├── .....
```

# Train
1. Download the dataset.
2. Use data_aug.m to augment the training set of the dataset.

3. Download [uniformer_base_ls_in1k.pth](https://pan.baidu.com/s/1Xp8TSt1UBiaKwQrGjgJtDg?pwd=o4dz) (code: o4dz), and put it in './pretrain/'. 

4. Modify paths of datasets, then run train_MyNet.py.


# Test
1. Download the pre-trained models of our network at [here](https://pan.baidu.com/s/1Xp8TSt1UBiaKwQrGjgJtDg?pwd=o4dz) (code:o4dz)
2. Modify paths of pre-trained models  and datasets.
3. Run test_MyNet.py.

# Results
## Main results on ORSSD dataset

| **Methods** | **S<sub>α</sub>** |  **MAE**   | **adp E<sub>ξ</sub>** | **mean E<sub>ξ</sub>** | **max E<sub>ξ</sub>** | **adp F<sub>β</sub>** | **mean F<sub>β</sub>** | **max F<sub>β</sub>** |
|:-----------:|:-----------------:|:----------:|:---------------------:|:----------------------:|:---------------------:|:---------------------:|------------------------|-----------------------|
|   SAMNet    |      0.8761       |   0.0217   |        0.8656         |         0.8818         |        0.9478         |        0.6843         | 0.7531                 | 0.8137                |
|   HVPNet    |      0.8610       |   0.0225   |        0.8471         |         0.8737         |        0.9320         |        0.6726         | 0.7396                 | 0.7938                |
|   DAFNet    |      0.9191       |   0.0113   |        0.9360         |         0.9539         |        0.9771         |        0.7876         | 0.8511                 | 0.8928                |
|   HFANet    |      0.9399       |   0.0092   |        0.9722         |         0.9712         |        0.9770         |        0.8819         | 0.8981                 | 0.9112                |
|   MSCNet    |      0.9227       |   0.0129   |        0.9584         |         0.9653         |        0.9754         |        0.8350         | 0.8676                 | 0.8927                |
|    MJRBM    |      0.9204       |   0.0163   |        0.9328         |         0.9415         |        0.9623         |        0.8022         | 0.8566                 | 0.8842                |
|    PAFR     |      0.8938       |   0.0211   |        0.9315         |         0.9268         |        0.9467         |        0.8025         | 0.8275                 | 0.8438                |
|   CorrNet   |      0.9201       |   0.0158   |        0.9543         |         0.9487         |        0.9575         |        0.8605         | 0.8717                 | 0.8841                |
|   EMFINet   |      0.9380       |   0.0113   |        0.9637         |         0.9657         |        0.9733         |        0.8664         | 0.8873                 | 0.9019                |
|   MCCNet    |      0.9445       |   0.0091   |        0.9733         |         0.9740         |        0.9805         |        0.8925         | 0.9045                 | 0.9177                |
|   ACCoNet   |      0.9418       |   0.0095   |        0.9694         |         0.9684         |        0.9754         |        0.8614         | 0.8847                 | 0.9112                |
|   AESINet   |      0.9427       |   0.0090   |        0.9704         |         0.9736         |        0.9817         |        0.8667         | 0.8975                 | 0.9166                |
|   ERPNet    |      0.9254       |   0.0135   |        0.9520         |         0.8566         |        0.9710         |        0.8356         | 0.8745                 | 0.8974                |
|   GeleNet   |      0.9451       |   0.0092   |      **0.9816**       |         0.9799         |        0.9859         |      **0.9044**       | **0.9123**             | 0.9239                |
|   ADSTNet   |      0.9379       |   0.0086   |        0.9785         |         0.9740         |        0.9807         |        0.8979         | 0.9042                 | 0.9124                |
|    Ours     |    **0.9491**     | **0.0070** |        0.9807         |       **0.9815**       |      **0.9864**       |        0.8994         | 0.9095                 | **0.9253**            |
- Bold indicates the best performance.

## Main results on EORSSD dataset

| **Methods** | **S<sub>α</sub>** |  **MAE**   | **adp E<sub>ξ</sub>** | **mean E<sub>ξ</sub>** | **max E<sub>ξ</sub>** | **adp F<sub>β</sub>** | **mean F<sub>β</sub>** | **max F<sub>β</sub>** |
|:-----------:|:-----------------:|:----------:|:---------------------:|:----------------------:|:---------------------:|:---------------------:|------------------------|-----------------------|
|   SAMNet    |      0.8622       |   0.0132   |        0.8284         |         0.8700         |        0.9421         |        0.6114         | 0.7214                 | 0.7813                |
|   HVPNet    |      0.8734       |   0.0110   |        0.8270         |         0.8721         |        0.9482         |        0.6202         | 0.7377                 | 0.8036                |
|   DAFNet    |      0.9166       |   0.0060   |        0.8443         |         0.9290         |      **0.9859**       |        0.6423         | 0.7842                 | 0.8612                |
|   HFANet    |      0.9380       |   0.0070   |        0.9644         |         0.9679         |        0.9740         |        0.8365         | 0.8681                 | 0.8876                |
|   MSCNet    |      0.9071       |   0.0090   |        0.9329         |         0.9551         |        0.9689         |        0.7553         | 0.8151                 | 0.8539                |
|    MJRBM    |      0.9197       |   0.0099   |        0.8897         |         0.9350         |        0.9646         |        0.7066         | 0.8239                 | 0.8656                |
|    PAFR     |      0.8927       |   0.0119   |        0.8959         |         0.9210         |        0.9490         |        0.7123         | 0.7961                 | 0.8260                |
|   CorrNet   |      0.9153       |   0.0097   |        0.9514         |         0.9445         |        0.9553         |        0.8259         | 0.8450                 | 0.8597                |
|   EMFINet   |      0.9284       |   0.0087   |        0.9482         |         0.9542         |        0.9665         |        0.8049         | 0.8494                 | 0.8735                |
|   MCCNet    |      0.9340       |   0.0073   |        0.9609         |         0.9676         |        0.9758         |        0.8302         | 0.8656                 | 0.8884                |
|   ACCoNet   |      0.9346       |   0.0081   |        0.9559         |         0.9622         |        0.9707         |        0.8248         | 0.8628                 | 0.8846                |
|   AESINet   |      0.9362       |   0.0072   |        0.9443         |         0.9618         |        0.9734         |        0.7908         | 0.8507                 | 0.8820                |
|   ERPNet    |      0.9210       |   0.0089   |        0.9228         |         0.9401         |        0.9603         |        0.7554         | 0.8304                 | 0.8632                |
|   GeleNet   |      0.9373       |   0.0075   |        0.9728         |         0.9740         |        0.9810         |        0.8648         | 0.8781                 | 0.8910                |
|   ADSTNet   |      0.9311       |   0.0065   |        0.9681         |         0.9709         |        0.9769         |        0.8532         | 0.8716                 | 0.8804                |
|    Ours     |    **0.9391**     | **0.0053** |      **0.9743**       |       **0.9784**       |        0.9836         |      **0.8657**       | **0.8790**             | **0.8937**            |
- Bold indicates the best performance.


# Visualization of results
<p align="center"> <img src="Images/Result1.png" width=95%"></p>

# Evaluation Tool
   You can use the [evaluation tool (MATLAB version)](https://github.com/MathLee/MatlabEvaluationTools) to evaluate the above saliency maps.


# ORSI-SOD Summary
Salient Object Detection in Optical Remote Sensing Images Read List at [here](https://github.com/MathLee/ORSI-SOD_Summary)

# Acknowledgements
This code is built on [PyTorch](https://pytorch.org).
# Contact
If you have any questions, please submit an issue on GitHub or contact me by email (cxh1638843923@gmail.com).
       
                
