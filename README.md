## Our manuscript has been accepted by CVPR! 

## Suppressing Uncertainties for Large-Scale Facial Expression Recognition

                                  Kai Wang, Xiaojiang Peng, Jianfei Yang, Shijian Lu, and Yu Qiao
                              Shenzhen Institutes of Advanced Technology, Chinese Academy of Sciences
                                         Nanyang Technological University, Singapore
                                         {kai.wang, xj.peng, yu.qiao}@siat.ac.cn
![image](https://github.com/kaiwang960112/Self-Cure-Network/blob/master/imgs/SCNpipeline.png)

### Abstract

Annotating a qualitative large-scale facial expression dataset is extremely difficult due to the uncertainties caused by ambiguous facial expressions, low-quality facial images, and the subjectiveness of annotators. These uncertainties lead to a key challenge of large-scale Facial Expression Recognition (FER) in deep learning era. To address this problem, this paper proposes a simple yet efficient Self-Cure Network (SCN) which suppresses the uncertainties efficiently and prevents deep networks from over-fitting uncertain facial images.
	Specifically, SCN suppresses the uncertainty from two different aspects: 1) a self-attention mechanism over mini-batch to weight each training sample with a ranking regularization, and 2) a careful relabeling mechanism to modify the labels of these samples in the lowest-ranked group. Experiments on synthetic FER datasets and our collected WebEmotion dataset validate the effectiveness of our method. 
	Results on public benchmarks demonstrate that our SCN outperforms current state-of-the-art methods with 88.14% on RAF-DB, 60.23% on AffectNet, and 89.35% on FERPlus.



