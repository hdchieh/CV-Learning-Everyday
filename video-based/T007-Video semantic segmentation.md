# Video semantic segmentation

**[Blazingly Fast Video Object Segmentation with Pixel-Wise Metric Learning, CVPR'18](http://openaccess.thecvf.com/content_cvpr_2018/papers/Chen_Blazingly_Fast_Video_CVPR_2018_paper.pdf)**

![](images/0054.png)

**Abstract**

This paper tackles the problem of video object segmentation, given some user annotation which indicates the object of interest. The problem is formulated as pixel-wise　etrieval in a learned embedding space: we embed pixels of the same object instance into the vicinity of each other, using a fully convolutional network trained by a modified triplet loss as the embedding model. Then the annotated pixels are set as reference and the rest of the pixels are classified using a nearest-neighbor approach. The proposed method supports different kinds of user input such as segmentation mask in the first frame (semi-supervised scenario), or a sparse set of clicked points (interactive scenario). In the semi-supervised scenario, we achieve results competitive with the state of the art but at a fraction of computation cost (275 milliseconds per frame). In the interactive scenario where the user is able to refine their input iteratively, the proposed method provides instant response to each input, and reaches comparable quality to competing methods with much less interaction.

这项工作为视频对象分割提供了一个概念上简单但非常有效的方法。这个问题是通过修改专门为视频对象分割而设计的三元组损失来学习嵌入空间中的像素方式检索。这样，视频上的注释像素（通过涂鸦，第一个掩模上的分割，点击等）就是参考样本，其余像素通过简单且快速的最近邻近方法进行分类。

[参考博客](https://blog.csdn.net/qq_16761599/article/details/80821007)



## 参考

[【cnblogs】应用于语义分割问题的深度学习技术综述](https://blog.csdn.net/bailing910/article/details/82625918)

[【知乎】CVPR 2018 | 弱监督语义分割简评](https://zhuanlan.zhihu.com/p/42058498)

[【知乎】视频语义分割介绍](https://zhuanlan.zhihu.com/p/52014957)