# FFANet
Abstract

Medical images are an important means to assist doctors in making judgments. For the problem that it is difficult to segment various complex tissue structures in medical images, we propose a segmentation method based on feature aggregation. Firstly, our method adopts VoVNet as the backbone and outputs multi-scale features. Secondly, we use the multi-scale features aggregation module to extract context information fully. Finally, we adopt an attention module to consider the relevance of each spatial and channel. Through the experiments conduct on two datasets, the proposed model scores a dice coefficient of 90.90% and 86.10%. Results show that our network can segment the target area well in the gray image and RGB image and outperforms the existing methodologies.

Experimental environment and settings

We adopt PyTorch to implement our method. All experiments are performed on the same server, which is equipped with one Nvidia 1080Ti and Intel i7-8700K CPU. In experiments, the Adam optimizer is adopted to train our method. We employ the cross-entropy loss as the training loss function. The initial learning rate is install to 0.001. We set the learning rate attenuation adaptive attenuation too. The training epochs is set to 250. The batch is set to 8.
