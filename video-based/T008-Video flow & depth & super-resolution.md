# Video flow & depth & super-resolution

## 介绍

图像超分辨率就是提高图像的空间分辨率，例如将一幅图片的分辨率由352x288扩大到704x576，方便用户在大尺寸的显示设备上观看;在数字成像技术，视频编码通信技术，深空卫星遥感技术，目标识别分析技术和医学影像分析技术等方面，视频图像超分辨率技术都能够应对显示设备分辨率大于图像源分辨率的问题。

超分辨率技术可以分为以下两种：

1. 只参考当前低分辨率图像，不依赖其他相关图像的超分辨率技术，称之为单幅图像的超分辨率（single image super resolution），也可以称之为图像插值（image interpolation）；

2. 参考多幅图像或多个视频帧的超分辨率技术，称之为多帧视频/多图的超分辨率（multi-frame super resolution）。

这两类技术中，一般来讲后者相比于前者具有更多的可参考信息，并具有更好的高分辨率视频图像的重建质量，但是其更高的计算复杂度也限制了其应用。

## 论文

## 参考

[【CSDN】super resolution 论文阅读简略笔记](https://blog.csdn.net/Zealoe/article/details/78550444)

[【CSDN】深度学习（二十二）——ESPCN, FSRCNN, VESPCN, SRGAN, DemosaicNet, MemNet, RDN, ShuffleSeg](https://blog.csdn.net/antkillerfarm/article/details/79956241)

[【CSDN】CVPR2019中关于超分辨率算法的16篇论文](https://blog.csdn.net/leviopku/article/details/90634994)
