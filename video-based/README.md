# 视频理解类任务

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

## 2. 相关研究

### [2.1 经典方法](https://github.com/huuuuusy/CV-Learning-Everyday/blob/master/video-based/001%E7%BB%8F%E5%85%B8%E6%96%B9%E6%B3%95.md)

DT和iDT方法是深度学习方法成熟之前效果最好的经典方法。

[H. Wang, et al. Dense trajectories and motion boundary descriptors for action recognition. IJCV'13.](https://hal.inria.fr/hal-00803241/PDF/IJCV.pdf)

[H. Wang and C. Schmid. Action recognition with improved trajectories. ICCV'13.](http://lear.inrialpes.fr/people/wang/download/iccv13_poster_final.pdf)

### [2.2 逐帧处理融合](https://github.com/huuuuusy/CV-Learning-Everyday/blob/master/video-based/002%E9%80%90%E5%B8%A7%E5%A4%84%E7%90%86%E8%9E%8D%E5%90%88.md)

这类方法把视频看作一系列图像的集合，每帧图像单独提取特征，再融合它们的深度特征。

[A. Karpathy, et al. Large-scale video classification with convolutional neural networks. CVPR'14.](http://lear.inrialpes.fr/people/wang/download/iccv13_poster_final.pdf)

[Le, et al. Learning hierarchical invariant spatio-temporal features for action recognition with independent subspace analysis. CVPR'11.](http://ai.stanford.edu/~quocle/LeZouYeungNg11.pdf)

[J. Y.-H. Ng, et al. Beyond short snippets: Deep networks for video classification. CVPR'15.](https://arxiv.org/abs/1503.08909)

[B. Fernando and S. Gould. Learning end-to-end video classification with rank-pooling. ICML'16.](https://users.cecs.anu.edu.au/~sgould/papers/icml16-vidClassification.pdf)

[X.-S. Wei, et al. Deep bimodal regression of apparent personality traits from short video sequences. TAC'17.](https://www.researchgate.net/publication/320366199_Deep_Bimodal_Regression_of_Apparent_Personality_Traits_from_Short_Video_Sequences)

[A. Kar, et al. AdaScan: Adaptive scan pooling in deep convolutional neural networks for human action recognition in videos. CVPR'17.](https://arxiv.org/pdf/1611.08240.pdf)

[M. Zolfaghari, et al. ECO: Efficient Convolutional network for Online video understanding. arXiv:1804.09066.](http://openaccess.thecvf.com/content_ECCV_2018/papers/Mohammadreza_Zolfaghari_ECO_Efficient_Convolutional_ECCV_2018_paper.pdf)

### [2.3 ConvLSTM](https://github.com/huuuuusy/CV-Learning-Everyday/blob/master/video-based/003ConvLSTM.md)

这类方法是用CNN提取每帧图像的特征，之后用LSTM挖掘它们之间的时序关系。

[J. Y.-H. Ng, et al. Beyond short snippets: Deep networks for video classification. CVPR'15.](https://arxiv.org/abs/1503.08909)

[J. Donahue, et al. Long-term recurrent convolutional networks for visual recognition and description. CVPR'15.](https://arxiv.org/abs/1411.4389)

[W. Du, et al. RPAN: An end-to-end recurrent pose-attention network for action recognition in videos. ICCV'17.](https://www.sciencedirect.com/science/article/pii/S0031320319301098)

### [2.4 3D卷积](https://github.com/huuuuusy/CV-Learning-Everyday/blob/master/video-based/0043D%E5%8D%B7%E7%A7%AF.md)

把视频划分成很多固定长度的片段(clip)，相比2D卷积，3D卷积可以提取连续帧之间的运动信息。

[M. Baccouche, et al. Sequential deep learning for human action recognition. HBU Workshop'11.](https://liris.cnrs.fr/Documents/Liris-5228.pdf)

[S. Ji, et al. 3D convolutional neural networks for human action recognition. TPAMI'13.](https://ieeexplore.ieee.org/document/6165309/)

[D. Tran, et al. Learning spatio-temporal features with 3D convolutional networks. ICCV'15.](https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Tran_Learning_Spatiotemporal_Features_ICCV_2015_paper.pdf)

[L. Sun, et al. Human action recognition using factorized spatio-temporal convolutional networks. ICCV'15.](https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Sun_Human_Action_Recognition_ICCV_2015_paper.pdf)

[J. Carreira and A. Zisserman. Quo vadis, action recognition? A new model and the Kinetics dataset. CVPR'17.](https://arxiv.org/pdf/1705.07750.pdf)

[Z. Qiu, et al. Learning spatio-temporal representation with pseudo-3D residual networks. ICCV'17.](http://openaccess.thecvf.com/content_ICCV_2017/papers/Qiu_Learning_Spatio-Temporal_Representation_ICCV_2017_paper.pdf)

[D. Tran, et al. A closer look at spatio-temporal convolutions for action recognition. CVPR'18.](https://arxiv.org/pdf/1711.11248.pdf)

[C. Lea, et al. Temporal convolutional networks for action segmentation and detection. CVPR'17.](http://openaccess.thecvf.com/content_cvpr_2017/papers/Lea_Temporal_Convolutional_Networks_CVPR_2017_paper.pdf)

[L. Wang, et al. Appearance-and-relation networks for video classfication. CVPR'18.](http://openaccess.thecvf.com/content_cvpr_2018/papers/Wang_Appearance-and-Relation_Networks_for_CVPR_2018_paper.pdf)

[K. Hara, et al. Can spatio-temporal 3D CNNs retrace the history of 2D CNNs and ImageNet? CVPR'18.](http://openaccess.thecvf.com/content_cvpr_2018/papers/Hara_Can_Spatiotemporal_3D_CVPR_2018_paper.pdf)

[X. Wang, et al. Non-local neural networks. CVPR'18.](http://openaccess.thecvf.com/content_cvpr_2018/papers/Wang_Non-Local_Neural_Networks_CVPR_2018_paper.pdf)

### [2.5 Two-stream](https://github.com/huuuuusy/CV-Learning-Everyday/blob/master/video-based/005Two-stream.md)

[K. Simonyan and A. Zisserman. Two-stream convolutional networks for action recognition in videos. NIPS'14.](https://arxiv.org/abs/1406.2199)

[L. Wang, et al. Action recognition with trajectory-pooled deep-convolutional descriptors. CVPR'15.](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Wang_Action_Recognition_With_2015_CVPR_paper.pdf)

[C. Feichtenhofer, et al. Convolutional two-stream network fusion for video action recognition. CVPR'16.](https://arxiv.org/pdf/1604.06573.pdf)

[Spatiotemporal Residual Networks for Video Action Recognition](https://arxiv.org/abs/1611.02155)

[C. Feichtenhofer, et al. Spatio-temporal multiplier networks for video action recognition. CVPR'17.](http://openaccess.thecvf.com/content_cvpr_2017/papers/Feichtenhofer_Spatiotemporal_Multiplier_Networks_CVPR_2017_paper.pdf)

[L. Wang, et al. Temporal segment networks: Towards good practices for deep action recognition. ECCV'16.](https://wanglimin.github.io/papers/WangXWQLTV_ECCV16.pdf)

[Z. Lan, et al. Deep local video feature for action recognition. CVPR'17.](http://openaccess.thecvf.com/content_cvpr_2017_workshops/w14/papers/Lan_Deep_Local_Video_CVPR_2017_paper.pdf)

[R. Girdhar, et al. ActionVLAD: Learning spatio-temporal aggregation for action recognition. CVPR'17.](http://openaccess.thecvf.com/content_cvpr_2017/papers/Girdhar_ActionVLAD_Learning_Spatio-Temporal_CVPR_2017_paper.pdf)

[G. A. Sigurdsson, et al. Asynchronous temporal fields for action recognition. CVPR'17.](http://openaccess.thecvf.com/content_cvpr_2017/papers/Sigurdsson_Asynchronous_Temporal_Fields_CVPR_2017_paper.pdf)

[W. Zhu, et al. A key volume mining deep framework for action recognition. CVPR'16.](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Zhu_A_Key_Volume_CVPR_2016_paper.pdf)

[Y. Wang, et al. Spatio-temporal pyramid network for video action recognition. CVPR'16.](http://openaccess.thecvf.com/content_cvpr_2017/papers/Wang_Spatiotemporal_Pyramid_Network_CVPR_2017_paper.pdf)

[A. Diba, et al. Deep temporal linear encoding networks. CVPR'17.](http://openaccess.thecvf.com/content_cvpr_2017/papers/Diba_Deep_Temporal_Linear_CVPR_2017_paper.pdf)

[R. Girdhar and D. Ramanan. Attentional pooling for action recognition. NIPS'17.](https://rohitgirdhar.github.io/AttentionalPoolingAction/)

[I. C. Duta, et al. Spatio-temporal vector of locally max-pooled features for action recognition in videos. CVPR'17.](https://www.researchgate.net/publication/315841539_Spatio-Temporal_Vector_of_Locally_Max_Pooled_Features_for_Action_Recognition_in_Videos)

[C.-Y. Wu, et al. Compressed video action recognition. CVPR'18.](http://openaccess.thecvf.com/content_cvpr_2018/papers/Wu_Compressed_Video_Action_CVPR_2018_paper.pdf)

[P. Weinzaepfel, et al. DeepFlow: Large displacement optical flow with deep matching. ICCV'13.](https://hal.inria.fr/hal-00873592/document/)

[A. Dosovitskiy, et al. FlowNet: Learning optical flow with convolutional networks. ICCV'15.](http://openaccess.thecvf.com/content_iccv_2015/papers/Dosovitskiy_FlowNet_Learning_Optical_ICCV_2015_paper.pdf)

[E. Ilg, et al. FlowNet 2.0: Evolution of optical flow estimation with deep networks. CVPR'17.](http://openaccess.thecvf.com/content_cvpr_2017/papers/Ilg_FlowNet_2.0_Evolution_CVPR_2017_paper.pdf)

### 2.6 其他视频理解任务

## 参考

以上内容参考资料：

[【知乎】张皓：视频理解近期研究进展](https://zhuanlan.zhihu.com/p/36330561)
