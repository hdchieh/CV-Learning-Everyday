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
|[M. Zolfaghari, et al. ECO: Efficient Convolutional network for Online video understanding.](http://openaccess.thecvf.com/content_ECCV_2018/papers/Mohammadreza_Zolfaghari_ECO_Efficient_Convolutional_ECCV_2018_paper.pdf)||

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

#### [1.3.4 Video prediction](https://github.com/huuuuusy/CV-Learning-Everyday/blob/master/video-based/T005-Video%20prediction.md)

|Paper|Extra Link|
| :--: |:--: |
|[Structure Preserving Video Prediction, CVPR'18](http://openaccess.thecvf.com/content_cvpr_2018/papers/Xu_Structure_Preserving_Video_CVPR_2018_paper.pdf)|[【PPT】](http://www.icst.pku.edu.cn/struct/Seminar/YuzhangHu_181209/YuzhangHu_181209.pdf)|
|[Learning to Extract a Video Sequence from a Single Motion-Blurred Image, CVPR'18](http://openaccess.thecvf.com/content_cvpr_2018/papers/Jin_Learning_to_Extract_CVPR_2018_paper.pdf)|[【Project Page】](https://github.com/MeiguangJin/Learning-to-Extract-a-Video-Sequence-from-a-Single-Motion-Blurred-Image)|

**Reference**

[【CSDN】基于深度学习的视频预测文献综述](https://blog.csdn.net/weixin_41024483/article/details/88366989)

#### [1.3.5 Video question answer & retrieval & search & reasoning](https://github.com/huuuuusy/CV-Learning-Everyday/blob/master/video-based/T006-Video%20question%20answer%20%26%20retrieval%20%26%20search%20%26%20reasoning.md)

|Paper|Extra Link|
| :--: |:--: |
|[Motion-Appearance Co-Memory Networks for Video Question Answering, CVPR'18](https://arxiv.org/abs/1803.10906)||
|[Finding "It": Weakly-Supervised Reference-Aware Visual Grounding in Instructional Videos, CVPR'18 Oral](http://openaccess.thecvf.com/content_cvpr_2018/papers/Huang_Finding_It_Weakly-Supervised_CVPR_2018_paper.pdf)|[【Project Page】](https://finding-it.github.io/)[【Blog(Chinese)】](https://zhuanlan.zhihu.com/p/36374060)|
|[Attend and Interact: Higher-Order Object Interactions for Video Understanding, CVPR'18](https://arxiv.org/abs/1711.06330)|[【Blog(Chinese)】](https://blog.csdn.net/u014230646/article/details/80878109)|
|[MovieGraphs: Towards Understanding Human-Centric Situations from Videos, CVPR'18](http://www.cs.toronto.edu/~makarand/papers/CVPR2018_MovieGraphs.pdf)|[【Project Page】](http://moviegraphs.cs.toronto.edu/)|

**Reference**

[【专知】【论文推荐】最新7篇视觉问答（VQA）相关论文—解释、读写记忆网络、逆视觉问答、视觉推理、可解释性、注意力机制、计数](https://cloud.tencent.com/developer/article/1086325)

#### [1.3.6 Video semantic segmentation](https://github.com/huuuuusy/CV-Learning-Everyday/blob/master/video-based/T007-Video%20semantic%20segmentation.md)

|Paper|Extra Link|
| :--: |:--: |
|[Blazingly Fast Video Object Segmentation with Pixel-Wise Metric Learning, CVPR'18](http://openaccess.thecvf.com/content_cvpr_2018/papers/Chen_Blazingly_Fast_Video_CVPR_2018_paper.pdf)|[【Blog(Chinese)】](https://blog.csdn.net/qq_16761599/article/details/80821007)|
|[MoNet: Deep Motion Exploitation for Video Object Segmentation, CVPR'18](http://openaccess.thecvf.com/content_cvpr_2018/papers/Xiao_MoNet_Deep_Motion_CVPR_2018_paper.pdf)|[【Blog(Chinese)】](https://blog.csdn.net/zxx827407369/article/details/84950833)|
|[Motion-Guided Cascaded Refinement Network for Video Object Segmentation, CVPR'18](http://openaccess.thecvf.com/content_cvpr_2018/papers/Hu_Motion-Guided_Cascaded_Refinement_CVPR_2018_paper.pdf)|[【Project Page】](https://github.com/feinanshan/Motion-Guided-CRN)[【Blog(Chinese)】](https://blog.csdn.net/qq_34914551/article/details/88096247)|
|[Dynamic Video Segmentation Network, CVPR'18](https://arxiv.org/abs/1804.00931)|[【Project Page】](https://github.com/XUSean0118/DVSNet)|
|[Efficient Video Object Segmentation via Network Modulation, CVPR'18](http://openaccess.thecvf.com/content_cvpr_2018/papers_backup/Yang_Efficient_Video_Object_CVPR_2018_paper.pdf)|[【Project Page】](https://github.com/linjieyangsc/video_seg)[【Blog(Chinese)1】](https://www.jianshu.com/p/a7def2b306ff)[【Blog(Chinese)2】](https://zhuanlan.zhihu.com/p/36139460)|
|[Low-Latency Video Semantic Segmentation, CVPR'18 Spotlight](https://arxiv.org/abs/1804.00389)|[【Blog(Chinese)】](https://zhuanlan.zhihu.com/p/36549785)|
|[CNN in MRF: Video Object Segmentation via Inference in A CNN-Based Higher-Order Spatio-Temporal MRF, CVPR'18](https://arxiv.org/abs/1803.09453)||
|[[Actor and Action Video Segmentation from a Sentence, CVPR'18 Oral](http://openaccess.thecvf.com/content_cvpr_2018/papers/Gavrilyuk_Actor_and_Action_CVPR_2018_paper.pdf)]|[[【Project Page】](https://kgavrilyuk.github.io/publication/actor_action/)
[【Blog(Chinese)】](https://blog.csdn.net/fuxin607/article/details/79955912)]|
|[Fast and Accurate Online Video Object Segmentation via Tracking Parts, CVPR'18](http://openaccess.thecvf.com/content_cvpr_2018/papers/Cheng_Fast_and_Accurate_CVPR_2018_paper.pdf)|[【Project Page】](https://github.com/JingchunCheng/FAVOS)[【Blog(Chinese)】](https://blog.csdn.net/weixin_39347054/article/details/83414251)|
|[[Semantic Video Segmentation by Gated Recurrent Flow Propagation, CVPR'18](https://arxiv.org/abs/1612.08871)]|[【Project Page】](https://github.com/D-Nilsson/GRFP)|
|[Reinforcement Cutting-Agent Learning for Video Object Segmentation, CVPR'18](http://openaccess.thecvf.com/content_cvpr_2018/papers_backup/Han_Reinforcement_Cutting-Agent_Learning_CVPR_2018_paper.pdf)||
|[Deep Spatio-Temporal Random Fields for Efficient Video Segmentation, CVPR'18](https://arxiv.org/abs/1807.03148)||

**Reference**

[【cnblogs】应用于语义分割问题的深度学习技术综述](https://blog.csdn.net/bailing910/article/details/82625918)

[【知乎】CVPR 2018 | 弱监督语义分割简评](https://zhuanlan.zhihu.com/p/42058498)

[【知乎】视频语义分割介绍](https://zhuanlan.zhihu.com/p/52014957)

[【CSDN】CVPR2018-Segmentation相关论文整理](https://blog.csdn.net/qq_16761599/article/details/80727466)

[【GitHub】Awesome Semantic Segmentation](https://github.com/mrgloom/awesome-semantic-segmentation)

#### [1.3.7 Video flow & depth & super-resolution](https://github.com/huuuuusy/CV-Learning-Everyday/blob/master/video-based/T008-Video%20flow%20%26%20depth%20%26%20super-resolution.md)

|Paper|Extra Link|
| :--: |:--: |
|[Frame-Recurrent Video Super-Resolution, CVPR'18](https://arxiv.org/abs/1801.04590)|[【Project Page】](https://github.com/msmsajjadi/frvsr)[【Blog(Chinese)】](https://blog.csdn.net/qq_33590958/article/details/89654853)|
|[PoseFlow: A Deep Motion Representation for Understanding Human Behaviors in Videos, CVPR'18](http://openaccess.thecvf.com/content_cvpr_2018/CameraReady/3170.pdf)||
|[LEGO: Learning Edge with Geometry all at Once by Watching Videos, CVPR'18](https://arxiv.org/abs/1803.05648)|[【Project Page】](https://github.com/zhenheny/LEGO)[【Blog(Chinese)】](https://zhuanlan.zhihu.com/p/50729039)|
|[Learning Depth from Monocular Videos using Direct Methods, CVPR'18](https://arxiv.org/abs/1712.00175)|[【Project Page】](https://github.com/MightyChaos/LKVOLearner)[【Blog(Chinese)】](https://blog.csdn.net/yueleileili/article/details/82946910)|
|[End-to-End Learning of Motion Representation for Video Understanding](https://arxiv.org/abs/1804.00413)|[【Project Page】](https://github.com/LijieFan/tvnet)[【Blog(Chinese)】](https://blog.csdn.net/weixin_42164269/article/details/80651752)|
|[Unsupervised Learning of Depth and Ego-Motion from Monocular Video Using 3D Geometric Constraints](https://arxiv.org/abs/1802.05522)||

**Reference**

[【CSDN】super resolution 论文阅读简略笔记](https://blog.csdn.net/Zealoe/article/details/78550444)

[【CSDN】深度学习（二十二）——ESPCN, FSRCNN, VESPCN, SRGAN, DemosaicNet, MemNet, RDN, ShuffleSeg](https://blog.csdn.net/antkillerfarm/article/details/79956241)

[【CSDN】CVPR2019中关于超分辨率算法的16篇论文](https://blog.csdn.net/leviopku/article/details/90634994)

[【GitHub】Video-Super-Resolution](https://github.com/flyywh/Video-Super-Resolution)

[【GitHub】Mind Mapping for Depth Estimation](https://github.com/sxfduter/monocular-depth-estimation)

[【知乎】CVPR2018 人体姿态相关](https://zhuanlan.zhihu.com/p/38328177)

[【知乎】深度学习在图像超分辨率重建中的应用](https://zhuanlan.zhihu.com/p/25532538)

#### [1.3.8 Video classification & recognition](https://github.com/huuuuusy/CV-Learning-Everyday/blob/master/video-based/T009-Video%20classification%20%26%20recognition.md)

|Paper|Extra Link|
| :--: |:--: |
|[Appearance-and-Relation Networks for Video Classification, CVPR'18](https://arxiv.org/abs/1711.09125)|[【Project Page】](https://github.com/wanglimin/ARTNet)[【Blog(Chinese)】](https://zhuanlan.zhihu.com/p/32197854)|
|[Recurrent Residual Module for Fast Inference in Videos, CVPR'18](https://arxiv.org/abs/1802.09723)||
|[Memory Based Online Learning of Deep Representations from Video Streams, CVPR'18](https://arxiv.org/abs/1711.07368)||
|[Geometry Guided Convolutional Neural Networks for Self-Supervised Video Representation Learning, CVPR'18](https://cseweb.ucsd.edu/~haosu/papers/cvpr18_geometry_predictive_learning.pdf)||
|[Learning Latent Super-Events to Detect Multiple Activities in Videos, CVPR'18](http://openaccess.thecvf.com/content_cvpr_2018/papers/Piergiovanni_Learning_Latent_Super-Events_CVPR_2018_paper.pdf)|[【Project Page】](https://github.com/piergiaj/super-events-cvpr18)|
|[Compressed Video Action Recognition, CVPR'18](http://openaccess.thecvf.com/content_cvpr_2018/papers/Wu_Compressed_Video_Action_CVPR_2018_paper.pdf)|[【Project Page】](https://github.com/chaoyuaw/pytorch-coviar)[【Blog(Chinese)1】](https://blog.csdn.net/perfects110/article/details/84329491)[【Blog(Chinese)2】](https://blog.csdn.net/Dongjiuqing/article/details/84678962)|
|[Video Representation Learning Using Discriminative Pooling, CVPR'18](http://openaccess.thecvf.com/content_cvpr_2018/papers/Wang_Video_Representation_Learning_CVPR_2018_paper.pdf)|[【Project Page】](https://github.com/3xwangDot/SVMP)|
|[Optical Flow Guided Feature: A Fast and Robust Motion Representation for Video Action Recognition, CVPR'18](https://arxiv.org/abs/1711.11152)|[【Project Page】](https://github.com/kevin-ssy/Optical-Flow-Guided-Feature)|
|[NeuralNetwork-Viterbi: A Framework for Weakly Supervised Video Learning, CVPR'18 Spotlight](https://arxiv.org/abs/1805.06875)|[【Project Page】](https://github.com/alexanderrichard/NeuralNetwork-Viterbi)|
|[Temporal Deformable Residual Networks for Action Segmentation in Videos, CVPR'18](http://openaccess.thecvf.com/content_cvpr_2018/papers/Lei_Temporal_Deformable_Residual_CVPR_2018_paper.pdf)||

**Reference**

[【知乎】简评 | Video Action Recognition 的近期进展](https://zhuanlan.zhihu.com/p/59915784)

[【知乎】Video Analysis相关领域解读之Action Recognition(行为识别)](https://zhuanlan.zhihu.com/p/26460437)

[【CSDN】3D CNN框架结构各层计算](https://blog.csdn.net/auto1993/article/details/70948249)

[【CSDN】Temporal Action Detection (时序动作检测)综述](https://blog.csdn.net/qq_33278461/article/details/80720104)

#### [1.3.9 Video caption]

|Paper|Extra Link|
| :--: |:--: |
|[Video Captioning via Hierarchical Reinforcement Learning, CVPR'18](https://arxiv.org/abs/1711.11135)|[【Blog(Chinese)】](https://cloud.tencent.com/developer/article/1092810)|
|[Fine-grained Video Captioning for Sports Narrative, CVPR'18](http://openaccess.thecvf.com/content_cvpr_2018/papers/Yu_Fine-Grained_Video_Captioning_CVPR_2018_paper.pdf)|[【Blog(Chinese)】](https://zhuanlan.zhihu.com/p/38292295)|
|[Jointly Localizing and Describing Events for Dense Video Captioning, CVPR'18](https://arxiv.org/abs/1804.08274)||
|[Bidirectional Attentive Fusion with Context Gating for Dense Video Captioning, CVPR'18](http://openaccess.thecvf.com/content_cvpr_2018/papers/Wang_Bidirectional_Attentive_Fusion_CVPR_2018_paper.pdf)|[【Project Page】](https://github.com/JaywongWang/DenseVideoCaptioning)[【Blog(Chinese)】](https://zhuanlan.zhihu.com/p/50924797)|
|[Interpretable Video Captioning via Trajectory Structured Localization, CVPR'18](http://openaccess.thecvf.com/content_cvpr_2018/papers/Wu_Interpretable_Video_Captioning_CVPR_2018_paper.pdf)||
|[End-to-End Dense Video Captioning with Masked Transformer, CVPR'18](https://arxiv.org/pdf/1804.00819.pdf)|[【Project Page】](https://github.com/salesforce/densecap)|
|[Reconstruction Network for Video Captioning, CVPR'18](https://arxiv.org/abs/1803.11438)|[【Project Page(Unofficial)】](https://github.com/hobincar/reconstruction-network-for-video-captioning)[【Blog(Chinese)】](https://zhuanlan.zhihu.com/p/50784504)|
|[Multimodal Memory Modelling for Video Captioning, CVPR'18](https://arxiv.org/abs/1611.05592)||

**Reference**

[【知乎】Video Analysis 相关领域介绍之Video Captioning(视频to文字描述)](https://zhuanlan.zhihu.com/p/26730181)

#### [1.3.10 Video generation (GAN)]

|Paper|Extra Link|
| :--: |:--: |
|[MoCoGAN: Decomposing Motion and Content for Video Generation, CVPR'18](http://openaccess.thecvf.com/content_cvpr_2018/papers/Tulyakov_MoCoGAN_Decomposing_Motion_CVPR_2018_paper.pdf)|[【Project Page】](https://github.com/sergeytulyakov/mocogan)|
|[Learning to Generate Time-Lapse Videos Using Multi-Stage Dynamic Generative Adversarial Networks, CVPR'18](https://arxiv.org/abs/1709.07592)||
|[Controllable Video Generation with Sparse Trajectories, CVPR'18](http://openaccess.thecvf.com/content_cvpr_2018/papers/Hao_Controllable_Video_Generation_CVPR_2018_paper.pdf)|[【Project Page】](https://github.com/zekunhao1995/ControllableVideoGen)|

#### [1.3.11 Video saliency & gaze prediction](https://github.com/huuuuusy/CV-Learning-Everyday/blob/master/video-based/T012-Video%20saliency%20%26%20gaze%20prediction.md)

#### [1.3.12 Video pedestrian tasks](https://github.com/huuuuusy/CV-Learning-Everyday/blob/master/video-based/T013-Video%20pedestrian%20tasks.md)

#### [1.3.13 Video frame interpolation](https://github.com/huuuuusy/CV-Learning-Everyday/blob/master/video-based/T014-Video%20frame%20interpolation.md)

|Paper|Extra Link|
| :--: |:--: |
|[Super SloMo: High Quality Estimation of Multiple Intermediate Frames for Video Interpolation, CVPR'18](https://arxiv.org/abs/1712.00080)|[【Project Page】](https://people.cs.umass.edu/~hzjiang/projects/superslomo/)[【Project Page(Unofficial)】](https://github.com/TheFairBear/Super-SlowMo)[【Blog(Chinese)】](https://zhuanlan.zhihu.com/p/69538032)|
|[PhaseNet for Video Frame Interpolation, CVPR'18](https://arxiv.org/abs/1804.00884)||
|[Context-aware Synthesis for Video Frame Interpolation, CVPR'18](https://arxiv.org/abs/1803.10967)||

**Reference**

[【GitHub】Video-Enhancement](https://github.com/yulunzhang/video-enhancement)
