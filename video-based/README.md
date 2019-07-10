# 视频类任务

## 1. 数据集

视频分类主要有两种数据集，剪辑过(trimmed)的视频和未经剪辑的视频。剪辑的视频中包含一段明确的动作，时间较短标记唯一，而未剪辑的视频还包含了很多无用信息。

|数据集|文章|介绍|
|:--: |:--: |:--: |
|[HMDB-51](http://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/)|【ICCV2011】[HMDB: A large video database for human motion recognition.](https://dspace.mit.edu/handle/1721.1/69981)|51类、6,766剪辑视频、每个视频不超过10秒、分辨率320 *240、共2 GB。视频源于YouTube和谷歌视频，内容包括人面部、肢体、和物体交互的动作这几大类。|
|[UCF-101](https://www.crcv.ucf.edu/data/UCF101.php)|【CoRR2012】[UCF101: A dataset of 101 human action classes from videos in the wild.](https://arxiv.org/abs/1212.0402)|101类、13,320视频剪辑、每个视频不超过10秒、共27小时、分辨率320*240、共6.5 GB。视频源于YouTube，内容包含化妆刷牙、爬行、理发、弹奏乐器、体育运动五大类。每类动作由25个人做动作，每人做4-7组|
|[Sports-1M](https://cs.stanford.edu/people/karpathy/deepvideo/classes.html)|【CVPR2014】[Large-scale video classification with convolutional neural networks. ](http://vision.stanford.edu/pdf/karpathy14.pdf)|487类、1,100,000视频（70%训练、20%验证、10%测试）。内容包含各种体育运动。|
|[Charades](https://allenai.org/plato/charades/)|【ECCV2016】[Hollywood in Homes: Crowdsourcing Data Collection for Activity Understanding](https://arxiv.org/abs/1604.01753)|57类、9,848未剪辑视频（7,985训练、1,863测试）、每个视频大约30秒。每个视频有多个标记，以及每个动作的开始和结束时间。|
|[ActivityNet](http://activity-net.org/)|【CVPR2015】[ A large-scale video benchmark for human activity understanding.](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Heilbron_ActivityNet_A_Large-Scale_2015_CVPR_paper.pdf)|200类、19,994未剪辑视频（10,024训练，4,926验证，5,044测试）、共648小时。内容包括饮食、运动、家庭活动等。|
|[Kinetics](https://deepmind.com/research/open-source/open-source-datasets/kinetics/)|【CoRR2017】[The Kinetics human action video dataset](https://arxiv.org/pdf/1705.06950.pdf)|400类、246k训练视频、20k验证视频、每个视频大约10秒。视频源于YouTube。Kinetics是一个大规模数据集，其在视频理解中的作用有些类似于ImageNet在图像识别中的作用，有些工作用Kinetics预训练模型迁移到其他视频数据集。|
|[YouTube-8M](https://research.google.com/youtube8m/)|【CoRR2016】[YouTube-8M: A large-scale video classification benchmark](https://arxiv.org/pdf/1609.08675.pdf)|4716类、7M视频、共450,000小时。不论是下载还是训练都很困难。|
|[Something-something](https://20bn.com/datasets/something-something)|【ICCV2017】[The "something something" video database for learning and evaluating visual common sense](https://arxiv.org/abs/1706.04261)|174类、108,000视频、每个视频2到6秒。和Kinetics不同，Something-something数据集需要更加细粒度、更加底层交互动作的区分，例如“从左向右推”和“从右向左推”。|

# 2. 相关研究

## 2.1 经典方法

DT和iDT方法是深度学习方法成熟之前效果最好的经典方法。

[H. Wang, et al. Dense trajectories and motion boundary descriptors for action recognition. IJCV'13.](https://hal.inria.fr/hal-00803241/PDF/IJCV.pdf)

[H. Wang and C. Schmid. Action recognition with improved trajectories. ICCV'13.](http://lear.inrialpes.fr/people/wang/download/iccv13_poster_final.pdf)

## 2.2 逐帧处理融合

这类方法把视频看作一系列图像的集合，每帧图像单独提取特征，再融合它们的深度特征。

[A. Karpathy, et al. Large-scale video classification with convolutional neural networks. CVPR'14.](http://lear.inrialpes.fr/people/wang/download/iccv13_poster_final.pdf)

[Le, et al. Learning hierarchical invariant spatio-temporal features for action recognition with independent subspace analysis. CVPR'11.](http://ai.stanford.edu/~quocle/LeZouYeungNg11.pdf)

[J. Y.-H. Ng, et al. Beyond short snippets: Deep networks for video classification. CVPR'15.](https://arxiv.org/abs/1503.08909)

[B. Fernando and S. Gould. Learning end-to-end video classification with rank-pooling. ICML'16.](https://users.cecs.anu.edu.au/~sgould/papers/icml16-vidClassification.pdf)

[X.-S. Wei, et al. Deep bimodal regression of apparent personality traits from short video sequences. TAC'17.](https://www.researchgate.net/publication/320366199_Deep_Bimodal_Regression_of_Apparent_Personality_Traits_from_Short_Video_Sequences)

[A. Kar, et al. AdaScan: Adaptive scan pooling in deep convolutional neural networks for human action recognition in videos. CVPR'17.](https://arxiv.org/pdf/1611.08240.pdf)

[M. Zolfaghari, et al. ECO: Efficient Convolutional network for Online video understanding. arXiv:1804.09066.](http://openaccess.thecvf.com/content_ECCV_2018/papers/Mohammadreza_Zolfaghari_ECO_Efficient_Convolutional_ECCV_2018_paper.pdf)

## 2.3 ConvLSTM

## 参考

以上内容参考资料：

[【知乎】张皓：视频理解近期研究进展](https://zhuanlan.zhihu.com/p/36330561)