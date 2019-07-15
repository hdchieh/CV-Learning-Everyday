# Video classification & recognition

## 介绍

视频分类主流方法:

1. 2-stream，结合光流和RGB，RGB支路可以是2D CNN 也可以是I3D
2. 3D CNN，卷积核多出时序上的维度，spatial-temporal 建模，变形是时空分离的伪3d、(2+1)D等
3. 时序信息用RNN建模
4. 传统方法，先进行密集跟踪点采样（角点提取/背景去除），对密集采样点进行光流计算获取一定帧长的轨迹，沿着轨迹进行一些如SIFT/HOG的特征提取，NIPS2018有一篇轨迹卷积将以上过程NN化。

## 论文

## 参考

[【知乎】简评 | Video Action Recognition 的近期进展](https://zhuanlan.zhihu.com/p/59915784)

[【知乎】Video Analysis相关领域解读之Action Recognition(行为识别)](https://zhuanlan.zhihu.com/p/26460437)

[【CSDN】3D CNN框架结构各层计算](https://blog.csdn.net/auto1993/article/details/70948249)
