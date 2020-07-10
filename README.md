## Thank you for everyone nice and kindly waiting!

## News：My friend [nzhq](https://github.com/nzhq) open the SCN code and reproduce the experiments result!!! Thank you Zhiqing!!!

## For the WebEmotion Dataset, I will open the search and clips generation code, everyone can download the videos from YouTube with my code. 

## Our manuscript has been accepted by CVPR2020! [link](https://arxiv.org/pdf/2002.10392.pdf)
## I really appreciate the contribution from my co-authors: Prof. Yu Qiao, Prof. Xiaojiang Peng, Jianfei Yang and Prof. Shijian Lu

# Based on our further exploring, SCN can be applied in many other topics.

# Suppressing Uncertainties for Large-Scale Facial Expression Recognition

                                  Kai Wang, Xiaojiang Peng, Jianfei Yang, Shijian Lu, and Yu Qiao
                              Shenzhen Institutes of Advanced Technology, Chinese Academy of Sciences
                                         Nanyang Technological University, Singapore
                                            {kai.wang, xj.peng, yu.qiao}@siat.ac.cn
				   Kai Wang and Xiaojiang Peng are equally-contributted authors
					 
![image](https://github.com/kaiwang960112/Self-Cure-Network/blob/master/imgs/scn-moti.png)

## Abstract

Annotating a qualitative large-scale facial expression dataset is extremely difficult due to the uncertainties caused by ambiguous facial expressions, low-quality facial images, and the subjectiveness of annotators. These uncertainties lead to a key challenge of large-scale Facial Expression Recognition (FER) in deep learning era. To address this problem, this paper proposes a simple yet efficient Self-Cure Network (SCN) which suppresses the uncertainties efficiently and prevents deep networks from over-fitting uncertain facial images. Specifically, SCN suppresses the uncertainty from two different aspects: 1) a self-attention mechanism over mini-batch to weight each training sample with a ranking regularization, and 2) a careful relabeling mechanism to modify the labels of these samples in the lowest-ranked group. Experiments on synthetic FER datasets and our collected WebEmotion dataset validate the effectiveness of our method. Results on public benchmarks demonstrate that our SCN outperforms current state-of-the-art methods with 88.14% on RAF-DB, 60.23% on AffectNet, and 89.35% on FERPlus.
	
## Self-Cure Network

![image](https://github.com/kaiwang960112/Self-Cure-Network/blob/master/imgs/SCNpipeline.png)

Our SCN is built upon traditional CNNs and consists of three crucial modules: i) self-attention importance weighting, ii) ranking regularization, and iii) relabeling, as shown in Figure 2. Given a batch of face images with some uncertain samples, we first extract the deep features by a backbone network. The self-attention importance weighting module assigns an importance weight for each image using a fully-connected (FC) layer and the sigmoid function. These weights are multiplied by the logits for a sample re-weighting scheme. To explicitly reduce the importance of uncertain samples, a rank regularization module is further introduced to regularize the attention weights. In the rank regularization module, we first rank the learned attention weights and then split them into two groups, i.e. high and low importance groups. We then add a constraint between the mean weights of these groups by a margin-based loss, which is called rank regularization loss (RR-Loss). To further improve our SCN, the relabeling module is added to modify some of the uncertain samples in the low importance group. This relabeling operation aims to hunt more clean samples and then to enhance the final model. The whole SCN can be trained in an end-to-end manner and easily added into any CNN backbones.

## Visualization

![image](https://github.com/kaiwang960112/Self-Cure-Network/blob/master/imgs/visularization2.png)


## Train
- Pytorch

  Torch 1.2.0 or higher and torchvision 0.4.0 or higher are required.
- Data Preparation

  Download basic emotions dataset of [RAF-DB](http://www.whdeng.cn/RAF/model1.html#dataset), and make sure it have a structure like following:
 
```
- datasets/raf-basic/
         EmoLabel/
             list_patition_label.txt
         Image/aligned/
	     train_00001_aligned.jpg
             test_0001_aligned.jpg
             ...
```
- Start Training

​```
      python train.py --margin_1=0.07
​```
  
  --margin_1 denotes the margin in Rank Regularization which is set to 0.15 with batch size 1024 in the paper. Here --margin_1=0.07 with smaller batch size 64[default] in train.py can get similar results.
- Result

![image](https://github.com/nzhq/Self-Cure-Network/blob/master/imgs/result-raf.jpg)
  
  Accuracy on test set should hit 87.03%, as the paper shows, when training with RAF-DB only.
