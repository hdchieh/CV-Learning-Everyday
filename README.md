# CV-Learning-Everyday

## 1. Video

### [1.1 Dataset](https://github.com/huuuuusy/CV-Learning-Everyday/blob/master/video-based/000-dataset.md)

|Dataset|Paper|
|:--: |:--: |
|[HMDB-51](http://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/)|【ICCV2011】[HMDB: A large video database for human motion recognition.](https://dspace.mit.edu/handle/1721.1/69981)|
|[UCF-101](https://www.crcv.ucf.edu/data/UCF101.php)|【CoRR2012】[UCF101: A dataset of 101 human action classes from videos in the wild.](https://arxiv.org/abs/1212.0402)|
|[Sports-1M](https://cs.stanford.edu/people/karpathy/deepvideo/classes.html)|【CVPR2014】[Large-scale video classification with convolutional neural networks.](http://vision.stanford.edu/pdf/karpathy14.pdf)|
|[Charades](https://allenai.org/plato/charades/)|【ECCV2016】[Hollywood in Homes: Crowdsourcing Data Collection for Activity Understanding](https://arxiv.org/abs/1604.01753)|
|[ActivityNet](http://activity-net.org/)|【CVPR2015】[A large-scale video benchmark for human activity understanding.](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Heilbron_ActivityNet_A_Large-Scale_2015_CVPR_paper.pdf)|
|[Kinetics](https://deepmind.com/research/open-source/open-source-datasets/kinetics/)|【CoRR2017】[The Kinetics human action video dataset](https://arxiv.org/pdf/1705.06950.pdf)|
|[YouTube-8M](https://research.google.com/youtube8m/)|【CoRR2016】[YouTube-8M: A large-scale video classification benchmark](https://arxiv.org/pdf/1609.08675.pdf)|
|[Something-something](https://20bn.com/datasets/something-something)|【ICCV2017】[The "something something" video database for learning and evaluating visual common sense](https://arxiv.org/abs/1706.04261)|

### 1.2 Method

#### [1.2.1 Classical methods](https://github.com/huuuuusy/CV-Learning-Everyday/blob/master/video-based/F001-Classical%20methods.md)

|Paper|Extra Link|
| :--: |:--: |
|[H. Wang, et al. Dense trajectories and motion boundary descriptors for action recognition. IJCV'13.](https://hal.inria.fr/hal-00803241/PDF/IJCV.pdf)||
|[H. Wang and C. Schmid. Action recognition with improved trajectories. ICCV'13.](http://lear.inrialpes.fr/people/wang/download/iccv13_poster_final.pdf)||

#### [1.2.2 Frame-by-frame processing fusion](https://github.com/huuuuusy/CV-Learning-Everyday/blob/master/video-based/F002-Frame-by-frame%20processing%20fusion.md)

|Paper|Extra Link|
| :--: |:--: |
|[A. Karpathy, et al. Large-scale video classification with convolutional neural networks. CVPR'14.](http://lear.inrialpes.fr/people/wang/download/iccv13_poster_final.pdf)||
|[Le, et al. Learning hierarchical invariant spatio-temporal features for action recognition with independent subspace analysis. CVPR'11.](http://ai.stanford.edu/~quocle/LeZouYeungNg11.pdf)||
|[J. Y.-H. Ng, et al. Beyond short snippets: Deep networks for video classification. CVPR'15.](https://arxiv.org/abs/1503.08909)||
|[B. Fernando and S. Gould. Learning end-to-end video classification with rank-pooling. ICML'16.](https://users.cecs.anu.edu.au/~sgould/papers/icml16-vidClassification.pdf)||
|[X.-S. Wei, et al. Deep bimodal regression of apparent personality traits from short video sequences. TAC'17.](https://www.researchgate.net/publication/320366199_Deep_Bimodal_Regression_of_Apparent_Personality_Traits_from_Short_Video_Sequences)||
|[A. Kar, et al. AdaScan: Adaptive scan pooling in deep convolutional neural networks for human action recognition in videos. CVPR'17.](https://arxiv.org/pdf/1611.08240.pdf)||
|[M. Zolfaghari, et al. ECO: Efficient Convolutional network for Online video understanding. arXiv:1804.09066.](http://openaccess.thecvf.com/content_ECCV_2018/papers/Mohammadreza_Zolfaghari_ECO_Efficient_Convolutional_ECCV_2018_paper.pdf)||

#### [1.2.3 ConvLSTM](https://github.com/huuuuusy/CV-Learning-Everyday/blob/master/video-based/F003-ConvLSTM.md)

|Paper|Extra Link|
| :--: |:--: |
|[J. Y.-H. Ng, et al. Beyond short snippets: Deep networks for video classification. CVPR'15.](https://arxiv.org/abs/1503.08909)||
|[J. Donahue, et al. Long-term recurrent convolutional networks for visual recognition and description. CVPR'15.](https://arxiv.org/abs/1411.4389)||
|[W. Du, et al. RPAN: An end-to-end recurrent pose-attention network for action recognition in videos. ICCV'17.](https://www.sciencedirect.com/science/article/pii/S0031320319301098)||

#### [1.2.4 3D convolution](https://github.com/huuuuusy/CV-Learning-Everyday/blob/master/video-based/F004-3D%20convolution.md)

|Paper|Extra Link|
| :--: |:--: |
|[M. Baccouche, et al. Sequential deep learning for human action recognition. HBU Workshop'11.](https://liris.cnrs.fr/Documents/Liris-5228.pdf)||
|[S. Ji, et al. 3D convolutional neural networks for human action recognition. TPAMI'13.](https://ieeexplore.ieee.org/document/6165309/)||
|[D. Tran, et al. Learning spatio-temporal features with 3D convolutional networks. ICCV'15.](https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Tran_Learning_Spatiotemporal_Features_ICCV_2015_paper.pdf)||
|[L. Sun, et al. Human action recognition using factorized spatio-temporal convolutional networks. ICCV'15.](https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Sun_Human_Action_Recognition_ICCV_2015_paper.pdf)||
|[J. Carreira and A. Zisserman. Quo vadis, action recognition? A new model and the Kinetics dataset. CVPR'17.](https://arxiv.org/pdf/1705.07750.pdf)||
|[Z. Qiu, et al. Learning spatio-temporal representation with pseudo-3D residual networks. ICCV'17.](http://openaccess.thecvf.com/content_ICCV_2017/papers/Qiu_Learning_Spatio-Temporal_Representation_ICCV_2017_paper.pdf)||
|[D. Tran, et al. A closer look at spatio-temporal convolutions for action recognition. CVPR'18.](https://arxiv.org/pdf/1711.11248.pdf)||
|[C. Lea, et al. Temporal convolutional networks for action segmentation and detection. CVPR'17.](http://openaccess.thecvf.com/content_cvpr_2017/papers/Lea_Temporal_Convolutional_Networks_CVPR_2017_paper.pdf)||
|[L. Wang, et al. Appearance-and-relation networks for video classfication. CVPR'18.](http://openaccess.thecvf.com/content_cvpr_2018/papers/Wang_Appearance-and-Relation_Networks_for_CVPR_2018_paper.pdf)||
|[K. Hara, et al. Can spatio-temporal 3D CNNs retrace the history of 2D CNNs and ImageNet? CVPR'18.](http://openaccess.thecvf.com/content_cvpr_2018/papers/Hara_Can_Spatiotemporal_3D_CVPR_2018_paper.pdf)||
|[X. Wang, et al. Non-local neural networks. CVPR'18.](http://openaccess.thecvf.com/content_cvpr_2018/papers/Wang_Non-Local_Neural_Networks_CVPR_2018_paper.pdf)||

#### [1.2.5 Two-stream](https://github.com/huuuuusy/CV-Learning-Everyday/blob/master/video-based/F005-Two-stream.md)

|Paper|Extra Link|
| :--: |:--: |
|[K. Simonyan and A. Zisserman. Two-stream convolutional networks for action recognition in videos. NIPS'14.](https://arxiv.org/abs/1406.2199)||
|[L. Wang, et al. Action recognition with trajectory-pooled deep-convolutional descriptors. CVPR'15.](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Wang_Action_Recognition_With_2015_CVPR_paper.pdf)||
|[C. Feichtenhofer, et al. Convolutional two-stream network fusion for video action recognition. CVPR'16.](https://arxiv.org/pdf/1604.06573.pdf)||
|[Spatiotemporal Residual Networks for Video Action Recognition, NIPS'16](https://arxiv.org/abs/1611.02155)||
|[C. Feichtenhofer, et al. Spatio-temporal multiplier networks for video action recognition. CVPR'17.](http://openaccess.thecvf.com/content_cvpr_2017/papers/Feichtenhofer_Spatiotemporal_Multiplier_Networks_CVPR_2017_paper.pdf)||
|[L. Wang, et al. Temporal segment networks: Towards good practices for deep action recognition. ECCV'16.](https://wanglimin.github.io/papers/WangXWQLTV_ECCV16.pdf)||
|[Z. Lan, et al. Deep local video feature for action recognition. CVPR'17.](http://openaccess.thecvf.com/content_cvpr_2017_workshops/w14/papers/Lan_Deep_Local_Video_CVPR_2017_paper.pdf)||
|[R. Girdhar, et al. ActionVLAD: Learning spatio-temporal aggregation for action recognition. CVPR'17.](http://openaccess.thecvf.com/content_cvpr_2017/papers/Girdhar_ActionVLAD_Learning_Spatio-Temporal_CVPR_2017_paper.pdf)||
|[G. A. Sigurdsson, et al. Asynchronous temporal fields for action recognition. CVPR'17.](http://openaccess.thecvf.com/content_cvpr_2017/papers/Sigurdsson_Asynchronous_Temporal_Fields_CVPR_2017_paper.pdf)||
|[W. Zhu, et al. A key volume mining deep framework for action recognition. CVPR'16.](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Zhu_A_Key_Volume_CVPR_2016_paper.pdf)||
|[Y. Wang, et al. Spatio-temporal pyramid network for video action recognition. CVPR'16.](http://openaccess.thecvf.com/content_cvpr_2017/papers/Wang_Spatiotemporal_Pyramid_Network_CVPR_2017_paper.pdf)||
|[A. Diba, et al. Deep temporal linear encoding networks. CVPR'17.](http://openaccess.thecvf.com/content_cvpr_2017/papers/Diba_Deep_Temporal_Linear_CVPR_2017_paper.pdf)||
|[R. Girdhar and D. Ramanan. Attentional pooling for action recognition. NIPS'17.](https://rohitgirdhar.github.io/AttentionalPoolingAction/)||
|[I. C. Duta, et al. Spatio-temporal vector of locally max-pooled features for action recognition in videos. CVPR'17.](https://www.researchgate.net/publication/315841539_Spatio-Temporal_Vector_of_Locally_Max_Pooled_Features_for_Action_Recognition_in_Videos)||
|[C.-Y. Wu, et al. Compressed video action recognition. CVPR'18.](http://openaccess.thecvf.com/content_cvpr_2018/papers/Wu_Compressed_Video_Action_CVPR_2018_paper.pdf)||
|[P. Weinzaepfel, et al. DeepFlow: Large displacement optical flow with deep matching. ICCV'13.](https://hal.inria.fr/hal-00873592/document/)||
|[A. Dosovitskiy, et al. FlowNet: Learning optical flow with convolutional networks. ICCV'15.](http://openaccess.thecvf.com/content_iccv_2015/papers/Dosovitskiy_FlowNet_Learning_Optical_ICCV_2015_paper.pdf)||
|[E. Ilg, et al. FlowNet 2.0: Evolution of optical flow estimation with deep networks. CVPR'17.](http://openaccess.thecvf.com/content_cvpr_2017/papers/Ilg_FlowNet_2.0_Evolution_CVPR_2017_paper.pdf)||

#### 1.2.6 Reference

[【知乎】视频理解近期研究进展](https://zhuanlan.zhihu.com/p/36330561)

[【知乎】计算机视觉中video understanding领域有什么研究方向和比较重要的成果？](https://www.zhihu.com/question/64021205)

### 1.3 Task

#### [1.3.1 Multitask learning](https://github.com/huuuuusy/CV-Learning-Everyday/blob/master/video-based/T002-Multitask%20Learning.md)

|Paper|Extra Link|
| :--: |:--: |
|[Visual to Sound: Generating Natural Sound for Videos in the Wild, CVPR'18](https://arxiv.org/abs/1712.01393)|[【Project Page】](http://bvision11.cs.unc.edu/bigpen/yipin/visual2sound_webpage/visual2sound.html)[【Blog(Chinese)】](http://www.sohu.com/a/209593882_610300)|

**Reference**

[【知乎】模型汇总-14 多任务学习-Multitask Learning概述](https://zhuanlan.zhihu.com/p/27421983)

[【知乎】一箭N雕：多任务深度学习实战](https://zhuanlan.zhihu.com/p/22190532)

[【机器之心】共享相关任务表征，一文读懂深度神经网络多任务学习](https://www.jiqizhixin.com/articles/2017-06-23-5)

#### [1.3.2 Video summarization](https://github.com/huuuuusy/CV-Learning-Everyday/blob/master/video-based/T003-Video%20summarization.md)

|Paper|Extra Link|
| :--: |:--: |
|[Viewpoint-aware Video Summarization, CVPR'18](https://arxiv.org/abs/1804.02843)||
|[HSA-RNN: Hierarchical Structure-Adaptive RNN for Video Summarization, CVPR'18](http://openaccess.thecvf.com/content_cvpr_2018/papers/Zhao_HSA-RNN_Hierarchical_Structure-Adaptive_CVPR_2018_paper.pdf)|[【Blog(Chinese)】](https://zhuanlan.zhihu.com/p/67510393)|
|[FFNet: Video Fast-Forwarding via Reinforcement Learning, CVPR'18](https://arxiv.org/abs/1805.02792)||
|[A Memory Network Approach for Story-based Temporal Summarization of 360° Videos, CVPR'18](https://arxiv.org/abs/1805.02838)||

**Reference**

[【机器之心】深度学习之视频摘要简述](https://blog.csdn.net/Uwr44UOuQcNsUQb60zk2/article/details/78869193)

#### [1.3.3 Video object detection](https://github.com/huuuuusy/CV-Learning-Everyday/blob/master/video-based/T004-Video%20object%20detection.md)

|Paper|Extra Link|
| :--: |:--: |
|[Mobile Video Object Detection with Temporally-Aware Feature Maps, CVPR'18](https://arxiv.org/abs/1711.06368)||
|[Deep Feature Flow for Video Recognition, CVPR'17](https://arxiv.org/abs/1611.07715)|[【Project Page】](https://github.com/msracver/Deep-Feature-Flow)[【Blog(Chinese)】](https://blog.csdn.net/lxt1994/article/details/79952310)|
|[Flow-Guided Feature Aggregation for Video Object Detection, ICCV'17](https://arxiv.org/abs/1703.10025)|[【Project Page】](https://github.com/msracver/Flow-Guided-Feature-Aggregation)[【Blog(Chinese)】](https://blog.csdn.net/lxt1994/article/details/79953401)|
|[Towards High Performance Video Object Detection, CVPR'18](https://arxiv.org/abs/1711.11577)|[【Blog(Chinese)】](https://zhuanlan.zhihu.com/p/37068429)|

**Reference**

[【GitHub】基于视频的目标检测算法研究](https://github.com/guanfuchen/video_obj)

[【知乎】视频中的目标检测与图像中的目标检测具体有什么区别？](https://www.zhihu.com/question/52185576)

#### [Video prediction](https://github.com/huuuuusy/CV-Learning-Everyday/blob/master/video-based/T005-Video%20prediction.md)

#### [Video question answer & retrieval & search & reasoning](https://github.com/huuuuusy/CV-Learning-Everyday/blob/master/video-based/T006-Video%20question%20answer%20%26%20retrieval%20%26%20search%20%26%20reasoning.md)

#### [Video semantic segmentation](https://github.com/huuuuusy/CV-Learning-Everyday/blob/master/video-based/T007-Video%20semantic%20segmentation.md)

#### [Video flow & depth & super-resolution](https://github.com/huuuuusy/CV-Learning-Everyday/blob/master/video-based/T008-Video%20flow%20%26%20depth%20%26%20super-resolution.md)

#### [Video classification & recognition](https://github.com/huuuuusy/CV-Learning-Everyday/blob/master/video-based/T009-Video%20classification%20%26%20recognition.md)

#### [Video caption]

#### [Video generation (GAN)]

#### [Video saliency & gaze prediction]

#### [Video pedestrian tasks]

#### [Video frame interpolation]

### 1.2 图像类

## 2. 工具类

### 2.1 OpenCV
