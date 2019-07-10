# Two-stream

[K. Simonyan and A. Zisserman. Two-stream convolutional networks for action recognition in videos. NIPS'14.](https://arxiv.org/abs/1406.2199)

![](images/0022.jpg)

采用两个分支。一个分支输入单帧图像，用于提取图像信息，即在做图像分类。另一个分支输入连续10帧的光流(optical flow)运动场，用于提取帧之间的运动信息。由于一个视频片段中的光流可能会沿某个特别方向位移的支配，所以在训练时光流减去所有光流向量的平均值。两个分支网络结构相同，分别用softmax进行预测，最后用直接平均或SVM两种方式融合两分支结果。

此外，为了加速训练，Simonyan和Zisserman预先计算出光流并保存到硬盘中。为了减小存储大小，他们将光流缩放到[0, 255]后用JPEG压缩，这会使UCF101的光流数据大小由1.5TB减小到27GB。

[L. Wang, et al. Action recognition with trajectory-pooled deep-convolutional descriptors. CVPR'15.](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Wang_Action_Recognition_With_2015_CVPR_paper.pdf)

![](images/0023.jpg)

Wang等人结合了经典iDT手工特征和two-stream深度特征，提出TDD。经典手工特征计算时通常分两步：检测图像中显著和有信息量的区域，并在运动显著的区域提取特征。TDD将预训练的two-stream网络当作固定的特征提取器。得到两者特征之后，TDD使用时空规范化以保证每个通道的数值范围近似一致，使用通道规范化以保证每个时空位置的描述向量的数值范围近似一致，之后用trajectory pooling并用Fisher向量构建TDD特征，最后用SVM分类。

[C. Feichtenhofer, et al. Convolutional two-stream network fusion for video action recognition. CVPR'16.](https://arxiv.org/pdf/1604.06573.pdf)

![](images/0024.jpg)

Feichtenhofer等人研究如何融合两分支的深度卷积特征。他们发现级联两个特征到2D维再用1×1卷积到D维的融合方法效果最好，之后再经过3D卷积和3D汇合后输出。

[Spatiotemporal Residual Networks for Video Action Recognition](https://arxiv.org/abs/1611.02155)

![](images/0025.jpg)

Feichtenhofer将ResNet作为two-stream的基础网络架构，用预训练网络的权重初始化新的3D网络：w(d, t, i, j) = w(d, i, j) / T。此外，有从光流分支到图像分支的信息传递。此外，网络输入不是连续的，而是步长5到15帧。

[C. Feichtenhofer, et al. Spatio-temporal multiplier networks for video action recognition. CVPR'17.](http://openaccess.thecvf.com/content_cvpr_2017/papers/Feichtenhofer_Spatiotemporal_Multiplier_Networks_CVPR_2017_paper.pdf)