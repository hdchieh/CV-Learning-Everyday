# Video caption

## 介绍

video captioning的任务是给视频生成文字描述，和image captioning（图片生成文字描述）有点像，区别主要在于视频还包含了时序的信息。

video captioning任务可以理解为视频图像序列到文本序列的seq2seq任务。在近年的方法中，大部分文章都使用了LSTM来构造encoder-decoder结构，即使用lstm encoder来编码视频图像序列的特征，再用lstm decoder解码出文本信息。这样的video captioning模型结构最早在ICCV2015的”Sequence to Sequence – Video to Text”一文中提出，如下图所示。

![](images/0066.png)

基于上图中的结构，构造一个encoder-decoder结构的模型主要包括几个关键点：

1. 输入特征：即如何提取视频中的特征信息，在很多篇文章中都使用了多模态的特征。主要包括如下几种：

    1)基于视频图像的信息：包括简单的用CNN（VGGNet, ResNet等）提取图像(spatial)特征，用action recognition的模型(如C3D)提取视频动态(spatial+temporal)特征

    2)基于声音的特征：对声音进行编码，包括BOAW（Bag-of-Audio-Words)和FV(Fisher Vector)等

    3)先验特征：比如视频的类别，这种特征能提供很强的先验信息

    4)基于文本的特征：此处基于文本的特征是指先从视频中提取一些文本的描述，再將这些描述作为特征，来进行video captioning。这类特征我看到过两类，一类是先对单帧视频进行image captioning,将image captioning的结果作为video captioning的输入特征，另外一类是做video tagging，将得到的标签作为特征。

2. encoder-decoder构造：虽然大部分工作都是用lstm做encoder-decoder，但各个方法的具体配置还是存在着一定的差异。

3. 输出词汇的表达：主要包括两类，一类是使用Word2Vec这种词向量表示，另外就是直接使用词袋表示。

4. 其它部分：比如训练策略，多任务训练之类的。

## 论文

## 参考

[【知乎】Video Analysis 相关领域介绍之Video Captioning(视频to文字描述)](https://zhuanlan.zhihu.com/p/26730181)
