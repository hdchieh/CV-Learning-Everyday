# CV-Learning-Everyday

The README file mainly contains papers and related links.

The reading notes of some papers will be published in the [issue section](https://github.com/huuuuusy/CV-Learning-Everyday/issues) and marked with different labels. You are welcome to discuss with me under the issue secion if you have interest in this paper.

## 1. Video

### [1.1 Dataset](https://github.com/huuuuusy/CV-Learning-Everyday/blob/master/video-based/000-dataset.md)

|Dataset|Paper|
|:--: |:--: |
|[HMDB-51](http://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/)|【ICCV'11】[HMDB: A large video database for human motion recognition.](https://dspace.mit.edu/handle/1721.1/69981)|
|[UCF-101](https://www.crcv.ucf.edu/data/UCF101.php)|【CoRR'12】[UCF101: A dataset of 101 human action classes from videos in the wild.](https://arxiv.org/abs/1212.0402)|
|[Sports-1M](https://cs.stanford.edu/people/karpathy/deepvideo/classes.html)|【CVPR'14】[Large-scale video classification with convolutional neural networks.](http://vision.stanford.edu/pdf/karpathy14.pdf)|
|[Charades](https://allenai.org/plato/charades/)|【ECCV'16】[Hollywood in Homes: Crowdsourcing Data Collection for Activity Understanding](https://arxiv.org/abs/1604.01753)|
|[ActivityNet](http://activity-net.org/)|【CVPR'15】[A large-scale video benchmark for human activity understanding.](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Heilbron_ActivityNet_A_Large-Scale_2015_CVPR_paper.pdf)|
|[Kinetics](https://deepmind.com/research/open-source/open-source-datasets/kinetics/)|【CoRR'17】[The Kinetics human action video dataset](https://arxiv.org/pdf/1705.06950.pdf)|
|[YouTube-8M](https://research.google.com/youtube8m/)|【CoRR'16】[YouTube-8M: A large-scale video classification benchmark](https://arxiv.org/pdf/1609.08675.pdf)|
|[Something-something](https://20bn.com/datasets/something-something)|【ICCV'17】[The "something something" video database for learning and evaluating visual common sense](https://arxiv.org/abs/1706.04261)|

### 1.2 Method

#### [1.2.1 Classical methods](https://github.com/huuuuusy/CV-Learning-Everyday/blob/master/video-based/F001-Classical%20methods.md)

|Paper|Extra Link|
| :--: |:--: |
|【IJCV'13】[H. Wang, et al. Dense trajectories and motion boundary descriptors for action recognition](https://hal.inria.fr/hal-00803241/PDF/IJCV.pdf)||
|【ICCV'13】[H. Wang and C. Schmid. Action recognition with improved trajectories](http://lear.inrialpes.fr/people/wang/download/iccv13_poster_final.pdf)||

#### [1.2.2 Frame-by-frame processing fusion](https://github.com/huuuuusy/CV-Learning-Everyday/blob/master/video-based/F002-Frame-by-frame%20processing%20fusion.md)

|Paper|Extra Link|
| :--: |:--: |
|【CVPR'14】[A. Karpathy, et al. Large-scale video classification with convolutional neural networks.](http://lear.inrialpes.fr/people/wang/download/iccv13_poster_final.pdf)||
|【CVPR'11】[Le, et al. Learning hierarchical invariant spatio-temporal features for action recognition with independent subspace analysis.](http://ai.stanford.edu/~quocle/LeZouYeungNg11.pdf)||
|【CVPR'15】[J. Y.-H. Ng, et al. Beyond short snippets: Deep networks for video classification.](https://arxiv.org/abs/1503.08909)||
|【ICML'16.】[B. Fernando and S. Gould. Learning end-to-end video classification with rank-pooling.](https://users.cecs.anu.edu.au/~sgould/papers/icml16-vidClassification.pdf)||
|【TAC'17】[X.-S. Wei, et al. Deep bimodal regression of apparent personality traits from short video sequences.](https://www.researchgate.net/publication/320366199_Deep_Bimodal_Regression_of_Apparent_Personality_Traits_from_Short_Video_Sequences)||
|【CVPR'17】[A. Kar, et al. AdaScan: Adaptive scan pooling in deep convolutional neural networks for human action recognition in videos.](https://arxiv.org/pdf/1611.08240.pdf)||
|[M. Zolfaghari, et al. ECO: Efficient Convolutional network for Online video understanding.](http://openaccess.thecvf.com/content_ECCV_2018/papers/Mohammadreza_Zolfaghari_ECO_Efficient_Convolutional_ECCV_2018_paper.pdf)||

#### [1.2.3 ConvLSTM](https://github.com/huuuuusy/CV-Learning-Everyday/blob/master/video-based/F003-ConvLSTM.md)

|Paper|Extra Link|
| :--: |:--: |
|【CVPR'15】[J. Y.-H. Ng, et al. Beyond short snippets: Deep networks for video classification.](https://arxiv.org/abs/1503.08909)||
|【CVPR'15】[J. Donahue, et al. Long-term recurrent convolutional networks for visual recognition and description.](https://arxiv.org/abs/1411.4389)||
|【ICCV'17】[W. Du, et al. RPAN: An end-to-end recurrent pose-attention network for action recognition in videos.](https://www.sciencedirect.com/science/article/pii/S0031320319301098)||

#### [1.2.4 3D convolution](https://github.com/huuuuusy/CV-Learning-Everyday/blob/master/video-based/F004-3D%20convolution.md)

|Paper|Extra Link|
| :--: |:--: |
|【HBU Workshop'11】[M. Baccouche, et al. Sequential deep learning for human action recognition.](https://liris.cnrs.fr/Documents/Liris-5228.pdf)||
|【TPAMI'13】[S. Ji, et al. 3D convolutional neural networks for human action recognition.](https://ieeexplore.ieee.org/document/6165309/)||
|【ICCV'15】[D. Tran, et al. Learning spatio-temporal features with 3D convolutional networks.](https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Tran_Learning_Spatiotemporal_Features_ICCV_2015_paper.pdf)||
|【ICCV'15】[L. Sun, et al. Human action recognition using factorized spatio-temporal convolutional networks.](https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Sun_Human_Action_Recognition_ICCV_2015_paper.pdf)||
|【CVPR'17】[J. Carreira and A. Zisserman. Quo vadis, action recognition? A new model and the Kinetics dataset.](https://arxiv.org/pdf/1705.07750.pdf)||
|【ICCV'17】[Z. Qiu, et al. Learning spatio-temporal representation with pseudo-3D residual networks.](http://openaccess.thecvf.com/content_ICCV_2017/papers/Qiu_Learning_Spatio-Temporal_Representation_ICCV_2017_paper.pdf)||
|【CVPR'18】[D. Tran, et al. A closer look at spatio-temporal convolutions for action recognition.](https://arxiv.org/pdf/1711.11248.pdf)||
|【CVPR'17】[C. Lea, et al. Temporal convolutional networks for action segmentation and detection.](http://openaccess.thecvf.com/content_cvpr_2017/papers/Lea_Temporal_Convolutional_Networks_CVPR_2017_paper.pdf)||
|【CVPR'18】[L. Wang, et al. Appearance-and-relation networks for video classfication.](http://openaccess.thecvf.com/content_cvpr_2018/papers/Wang_Appearance-and-Relation_Networks_for_CVPR_2018_paper.pdf)||
|【CVPR'18】[K. Hara, et al. Can spatio-temporal 3D CNNs retrace the history of 2D CNNs and ImageNet?](http://openaccess.thecvf.com/content_cvpr_2018/papers/Hara_Can_Spatiotemporal_3D_CVPR_2018_paper.pdf)||
|【CVPR'18】[X. Wang, et al. Non-local neural networks.](http://openaccess.thecvf.com/content_cvpr_2018/papers/Wang_Non-Local_Neural_Networks_CVPR_2018_paper.pdf)||

#### [1.2.5 Two-stream](https://github.com/huuuuusy/CV-Learning-Everyday/blob/master/video-based/F005-Two-stream.md)

|Paper|Extra Link|
| :--: |:--: |
|【NIPS'14】[K. Simonyan and A. Zisserman. Two-stream convolutional networks for action recognition in videos.](https://arxiv.org/abs/1406.2199)||
|【CVPR'15】[L. Wang, et al. Action recognition with trajectory-pooled deep-convolutional descriptors.](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Wang_Action_Recognition_With_2015_CVPR_paper.pdf)||
|【CVPR'16】[C. Feichtenhofer, et al. Convolutional two-stream network fusion for video action recognition.](https://arxiv.org/pdf/1604.06573.pdf)||
|【NIPS'16】[Spatiotemporal Residual Networks for Video Action Recognition](https://arxiv.org/abs/1611.02155)||
|【CVPR'17】[C. Feichtenhofer, et al. Spatio-temporal multiplier networks for video action recognition.](http://openaccess.thecvf.com/content_cvpr_2017/papers/Feichtenhofer_Spatiotemporal_Multiplier_Networks_CVPR_2017_paper.pdf)||
|【ECCV'16】[L. Wang, et al. Temporal segment networks: Towards good practices for deep action recognition.](https://wanglimin.github.io/papers/WangXWQLTV_ECCV16.pdf)||
|【CVPR'17】[Z. Lan, et al. Deep local video feature for action recognition.](http://openaccess.thecvf.com/content_cvpr_2017_workshops/w14/papers/Lan_Deep_Local_Video_CVPR_2017_paper.pdf)||
|【CVPR'17】[R. Girdhar, et al. ActionVLAD: Learning spatio-temporal aggregation for action recognition.](http://openaccess.thecvf.com/content_cvpr_2017/papers/Girdhar_ActionVLAD_Learning_Spatio-Temporal_CVPR_2017_paper.pdf)||
|【CVPR'17】[G. A. Sigurdsson, et al. Asynchronous temporal fields for action recognition.](http://openaccess.thecvf.com/content_cvpr_2017/papers/Sigurdsson_Asynchronous_Temporal_Fields_CVPR_2017_paper.pdf)||
|【CVPR'16】[W. Zhu, et al. A key volume mining deep framework for action recognition.](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Zhu_A_Key_Volume_CVPR_2016_paper.pdf)||
|【CVPR'16】[Y. Wang, et al. Spatio-temporal pyramid network for video action recognition.](http://openaccess.thecvf.com/content_cvpr_2017/papers/Wang_Spatiotemporal_Pyramid_Network_CVPR_2017_paper.pdf)||
|【CVPR'17】[A. Diba, et al. Deep temporal linear encoding networks.](http://openaccess.thecvf.com/content_cvpr_2017/papers/Diba_Deep_Temporal_Linear_CVPR_2017_paper.pdf)||
|【NIPS'17】[R. Girdhar and D. Ramanan. Attentional pooling for action recognition.](https://rohitgirdhar.github.io/AttentionalPoolingAction/)||
|【CVPR'17】[I. C. Duta, et al. Spatio-temporal vector of locally max-pooled features for action recognition in videos.](https://www.researchgate.net/publication/315841539_Spatio-Temporal_Vector_of_Locally_Max_Pooled_Features_for_Action_Recognition_in_Videos)||
|【CVPR'18】[C.-Y. Wu, et al. Compressed video action recognition.](http://openaccess.thecvf.com/content_cvpr_2018/papers/Wu_Compressed_Video_Action_CVPR_2018_paper.pdf)||
|【ICCV'13】[P. Weinzaepfel, et al. DeepFlow: Large displacement optical flow with deep matching.](https://hal.inria.fr/hal-00873592/document/)||
|【ICCV'15】[A. Dosovitskiy, et al. FlowNet: Learning optical flow with convolutional networks.](http://openaccess.thecvf.com/content_iccv_2015/papers/Dosovitskiy_FlowNet_Learning_Optical_ICCV_2015_paper.pdf)||
|【CVPR'17】[E. Ilg, et al. FlowNet 2.0: Evolution of optical flow estimation with deep networks.](http://openaccess.thecvf.com/content_cvpr_2017/papers/Ilg_FlowNet_2.0_Evolution_CVPR_2017_paper.pdf)||

#### 1.2.6 Reference

|Reference|
| :--: |
|【知乎】[视频理解近期研究进展](https://zhuanlan.zhihu.com/p/36330561)|
|【知乎】[计算机视觉中video understanding领域有什么研究方向和比较重要的成果？](https://www.zhihu.com/question/64021205)|

### 1.3 Task

#### [1.3.1 Multitask learning](https://github.com/huuuuusy/CV-Learning-Everyday/blob/master/video-based/T002-Multitask%20Learning.md)

|Paper|Extra Link|
| :--: |:--: |
|【CVPR'18】[Visual to Sound: Generating Natural Sound for Videos in the Wild](https://arxiv.org/abs/1712.01393)|[【Project Page】](http://bvision11.cs.unc.edu/bigpen/yipin/visual2sound_webpage/visual2sound.html)[【Blog(Chinese)】](http://www.sohu.com/a/209593882_610300)|

|Reference|
| :--: |
|【知乎】[模型汇总-14 多任务学习-Multitask Learning概述](https://zhuanlan.zhihu.com/p/27421983)|
|【知乎】[一箭N雕：多任务深度学习实战](https://zhuanlan.zhihu.com/p/22190532)|
|【机器之心】[共享相关任务表征，一文读懂深度神经网络多任务学习](https://www.jiqizhixin.com/articles/2017-06-23-5)|

#### [1.3.2 Video summarization](https://github.com/huuuuusy/CV-Learning-Everyday/blob/master/video-based/T003-Video%20summarization.md)

|Paper|Extra Link|
| :--: |:--: |
|【CVPR'18】[Viewpoint-aware Video Summarization](https://arxiv.org/abs/1804.02843)||
|【CVPR'18】[HSA-RNN: Hierarchical Structure-Adaptive RNN for Video Summarization](http://openaccess.thecvf.com/content_cvpr_2018/papers/Zhao_HSA-RNN_Hierarchical_Structure-Adaptive_CVPR_2018_paper.pdf)|[【Blog(Chinese)】](https://zhuanlan.zhihu.com/p/67510393)|
|【CVPR'18】[FFNet: Video Fast-Forwarding via Reinforcement Learning](https://arxiv.org/abs/1805.02792)||
|【CVPR'18】[A Memory Network Approach for Story-based Temporal Summarization of 360° Videos](https://arxiv.org/abs/1805.02838)||

|Reference|
| :--: |
|【机器之心】[深度学习之视频摘要简述](https://blog.csdn.net/Uwr44UOuQcNsUQb60zk2/article/details/78869193)|

#### [1.3.3 Video object detection](https://github.com/huuuuusy/CV-Learning-Everyday/blob/master/video-based/T004-Video%20object%20detection.md)

|Paper|Extra Link|
| :--: |:--: |
|【CVPR'18】[Mobile Video Object Detection with Temporally-Aware Feature Maps](https://arxiv.org/abs/1711.06368)||
|【CVPR'17】[Deep Feature Flow for Video Recognition](https://arxiv.org/abs/1611.07715)|[【Project Page】](https://github.com/msracver/Deep-Feature-Flow)[【Blog(Chinese)】](https://blog.csdn.net/lxt1994/article/details/79952310)|
|【ICCV'17】[Flow-Guided Feature Aggregation for Video Object Detection](https://arxiv.org/abs/1703.10025)|[【Project Page】](https://github.com/msracver/Flow-Guided-Feature-Aggregation)[【Blog(Chinese)】](https://blog.csdn.net/lxt1994/article/details/79953401)|
|【CVPR'18】[Towards High Performance Video Object Detection](https://arxiv.org/abs/1711.11577)|[【Blog(Chinese)】](https://zhuanlan.zhihu.com/p/37068429)|

|Reference|
| :--: |
|【GitHub】[基于视频的目标检测算法研究](https://github.com/guanfuchen/video_obj)|
|【知乎】[视频中的目标检测与图像中的目标检测具体有什么区别？](https://www.zhihu.com/question/52185576)|

#### [1.3.4 Video prediction](https://github.com/huuuuusy/CV-Learning-Everyday/blob/master/video-based/T005-Video%20prediction.md)

|Paper|Extra Link|
| :--: |:--: |
|【CVPR'18】[Structure Preserving Video Prediction](http://openaccess.thecvf.com/content_cvpr_2018/papers/Xu_Structure_Preserving_Video_CVPR_2018_paper.pdf)|[【PPT】](http://www.icst.pku.edu.cn/struct/Seminar/YuzhangHu_181209/YuzhangHu_181209.pdf)|
|【CVPR'18】[Learning to Extract a Video Sequence from a Single Motion-Blurred Image](http://openaccess.thecvf.com/content_cvpr_2018/papers/Jin_Learning_to_Extract_CVPR_2018_paper.pdf)|[【Project Page】](https://github.com/MeiguangJin/Learning-to-Extract-a-Video-Sequence-from-a-Single-Motion-Blurred-Image)|

|Reference|
| :--: |
|【CSDN】[基于深度学习的视频预测文献综述](https://blog.csdn.net/weixin_41024483/article/details/88366989)|

#### [1.3.5 Video question answer & retrieval & search & reasoning](https://github.com/huuuuusy/CV-Learning-Everyday/blob/master/video-based/T006-Video%20question%20answer%20%26%20retrieval%20%26%20search%20%26%20reasoning.md)

|Paper|Extra Link|
| :--: |:--: |
|【CVPR'18】[Motion-Appearance Co-Memory Networks for Video Question Answering](https://arxiv.org/abs/1803.10906)||
|【CVPR'18 Oral】[Finding "It": Weakly-Supervised Reference-Aware Visual Grounding in Instructional Videos](http://openaccess.thecvf.com/content_cvpr_2018/papers/Huang_Finding_It_Weakly-Supervised_CVPR_2018_paper.pdf)|[【Project Page】](https://finding-it.github.io/)[【Blog(Chinese)】](https://zhuanlan.zhihu.com/p/36374060)|
|【CVPR'18】[Attend and Interact: Higher-Order Object Interactions for Video Understanding](https://arxiv.org/abs/1711.06330)|[【Blog(Chinese)】](https://blog.csdn.net/u014230646/article/details/80878109)|
|【CVPR'18】[MovieGraphs: Towards Understanding Human-Centric Situations from Videos](http://www.cs.toronto.edu/~makarand/papers/CVPR2018_MovieGraphs.pdf)|[【Project Page】](http://moviegraphs.cs.toronto.edu/)|

|Reference|
| :--: |
|【专知】[【论文推荐】最新7篇视觉问答（VQA）相关论文—解释、读写记忆网络、逆视觉问答、视觉推理、可解释性、注意力机制、计数](https://cloud.tencent.com/developer/article/1086325)|

#### [1.3.6 Video semantic segmentation](https://github.com/huuuuusy/CV-Learning-Everyday/blob/master/video-based/T007-Video%20semantic%20segmentation.md)

|Paper|Extra Link|
| :--: |:--: |
|【CVPR'18】[Blazingly Fast Video Object Segmentation with Pixel-Wise Metric Learning](http://openaccess.thecvf.com/content_cvpr_2018/papers/Chen_Blazingly_Fast_Video_CVPR_2018_paper.pdf)|[【Blog(Chinese)】](https://blog.csdn.net/qq_16761599/article/details/80821007)|
|【CVPR'18】[MoNet: Deep Motion Exploitation for Video Object Segmentation](http://openaccess.thecvf.com/content_cvpr_2018/papers/Xiao_MoNet_Deep_Motion_CVPR_2018_paper.pdf)|[【Blog(Chinese)】](https://blog.csdn.net/zxx827407369/article/details/84950833)|
|【CVPR'18】[Motion-Guided Cascaded Refinement Network for Video Object Segmentation](http://openaccess.thecvf.com/content_cvpr_2018/papers/Hu_Motion-Guided_Cascaded_Refinement_CVPR_2018_paper.pdf)|[【Project Page】](https://github.com/feinanshan/Motion-Guided-CRN)[【Blog(Chinese)】](https://blog.csdn.net/qq_34914551/article/details/88096247)|
|【CVPR'18】[Dynamic Video Segmentation Network](https://arxiv.org/abs/1804.00931)|[【Project Page】](https://github.com/XUSean0118/DVSNet)|
|【CVPR'18】[Efficient Video Object Segmentation via Network Modulation](http://openaccess.thecvf.com/content_cvpr_2018/papers_backup/Yang_Efficient_Video_Object_CVPR_2018_paper.pdf)|[【Project Page】](https://github.com/linjieyangsc/video_seg)[【Blog(Chinese)1】](https://www.jianshu.com/p/a7def2b306ff)[【Blog(Chinese)2】](https://zhuanlan.zhihu.com/p/36139460)|
|【CVPR'18】[Low-Latency Video Semantic Segmentation Spotlight](https://arxiv.org/abs/1804.00389)|[【Blog(Chinese)】](https://zhuanlan.zhihu.com/p/36549785)|
|【CVPR'18】[CNN in MRF: Video Object Segmentation via Inference in A CNN-Based Higher-Order Spatio-Temporal MRF](https://arxiv.org/abs/1803.09453)||
|【CVPR'18 Oral】[Actor and Action Video Segmentation from a Sentence](http://openaccess.thecvf.com/content_cvpr_2018/papers/Gavrilyuk_Actor_and_Action_CVPR_2018_paper.pdf)|[【Project Page】](https://kgavrilyuk.github.io/publication/actor_action/)[【Blog(Chinese)】](https://blog.csdn.net/fuxin607/article/details/79955912)|
|【CVPR'18】[Fast and Accurate Online Video Object Segmentation via Tracking Parts](http://openaccess.thecvf.com/content_cvpr_2018/papers/Cheng_Fast_and_Accurate_CVPR_2018_paper.pdf)|[【Project Page】](https://github.com/JingchunCheng/FAVOS)[【Blog(Chinese)】](https://blog.csdn.net/weixin_39347054/article/details/83414251)|
|【CVPR'18】[Semantic Video Segmentation by Gated Recurrent Flow Propagation](https://arxiv.org/abs/1612.08871)|[【Project Page】](https://github.com/D-Nilsson/GRFP)|
|【CVPR'18】[Reinforcement Cutting-Agent Learning for Video Object Segmentation](http://openaccess.thecvf.com/content_cvpr_2018/papers_backup/Han_Reinforcement_Cutting-Agent_Learning_CVPR_2018_paper.pdf)||
|【CVPR'18】[Deep Spatio-Temporal Random Fields for Efficient Video Segmentation](https://arxiv.org/abs/1807.03148)||

|Reference|
| :--: |
|【cnblogs】[应用于语义分割问题的深度学习技术综述](https://blog.csdn.net/bailing910/article/details/82625918)|
|【知乎】[CVPR 2018 弱监督语义分割简评](https://zhuanlan.zhihu.com/p/42058498)|
|【知乎】[视频语义分割介绍](https://zhuanlan.zhihu.com/p/52014957)|
|【CSDN】[CVPR2018-Segmentation相关论文整理](https://blog.csdn.net/qq_16761599/article/details/80727466)|
|【GitHub】[Awesome Semantic Segmentation](https://github.com/mrgloom/awesome-semantic-segmentation)|

#### [1.3.7 Video flow & depth & super-resolution](https://github.com/huuuuusy/CV-Learning-Everyday/blob/master/video-based/T008-Video%20flow%20%26%20depth%20%26%20super-resolution.md)

|Paper|Extra Link|
| :--: |:--: |
|【CVPR'18】[Frame-Recurrent Video Super-Resolution](https://arxiv.org/abs/1801.04590)|[【Project Page】](https://github.com/msmsajjadi/frvsr)[【Blog(Chinese)】](https://blog.csdn.net/qq_33590958/article/details/89654853)|
|【CVPR'18】[PoseFlow: A Deep Motion Representation for Understanding Human Behaviors in Videos](http://openaccess.thecvf.com/content_cvpr_2018/CameraReady/3170.pdf)||
|【CVPR'18】[LEGO: Learning Edge with Geometry all at Once by Watching Videos](https://arxiv.org/abs/1803.05648)|[【Project Page】](https://github.com/zhenheny/LEGO)[【Blog(Chinese)】](https://zhuanlan.zhihu.com/p/50729039)|
|【CVPR'18】[Learning Depth from Monocular Videos using Direct Methods](https://arxiv.org/abs/1712.00175)|[【Project Page】](https://github.com/MightyChaos/LKVOLearner)[【Blog(Chinese)】](https://blog.csdn.net/yueleileili/article/details/82946910)|
|【CVPR'18】[End-to-End Learning of Motion Representation for Video Understanding](https://arxiv.org/abs/1804.00413)|[【Project Page】](https://github.com/LijieFan/tvnet)[【Blog(Chinese)】](https://blog.csdn.net/weixin_42164269/article/details/80651752)|
|【CVPR'18】[Unsupervised Learning of Depth and Ego-Motion from Monocular Video Using 3D Geometric Constraints](https://arxiv.org/abs/1802.05522)||

|Reference|
| :--: |
|【CSDN】[super resolution 论文阅读简略笔记](https://blog.csdn.net/Zealoe/article/details/78550444)|
|【CSDN】[深度学习（二十二）——ESPCN, FSRCNN, VESPCN, SRGAN, DemosaicNet, MemNet, RDN, ShuffleSeg](https://blog.csdn.net/antkillerfarm/article/details/79956241)|
|【CSDN】[CVPR2019中关于超分辨率算法的16篇论文](https://blog.csdn.net/leviopku/article/details/90634994)|
|【GitHub】[Video-Super-Resolution](https://github.com/flyywh/Video-Super-Resolution)|
|【GitHub】[Mind Mapping for Depth Estimation](https://github.com/sxfduter/monocular-depth-estimation)|
|【知乎】[CVPR2018 人体姿态相关](https://zhuanlan.zhihu.com/p/38328177)|
|【知乎】[深度学习在图像超分辨率重建中的应用](https://zhuanlan.zhihu.com/p/25532538)|

#### [1.3.8 Video classification & recognition](https://github.com/huuuuusy/CV-Learning-Everyday/blob/master/video-based/T009-Video%20classification%20%26%20recognition.md)

|Paper|Extra Link|
| :--: |:--: |
|【CVPR'18】[Appearance-and-Relation Networks for Video Classification](https://arxiv.org/abs/1711.09125)|[【Project Page】](https://github.com/wanglimin/ARTNet)[【Blog(Chinese)】](https://zhuanlan.zhihu.com/p/32197854)|
|【CVPR'18】[Recurrent Residual Module for Fast Inference in Videos](https://arxiv.org/abs/1802.09723)||
|【CVPR'18】[Memory Based Online Learning of Deep Representations from Video Streams](https://arxiv.org/abs/1711.07368)||
|【CVPR'18】[Geometry Guided Convolutional Neural Networks for Self-Supervised Video Representation Learning](https://cseweb.ucsd.edu/~haosu/papers/cvpr18_geometry_predictive_learning.pdf)||
|【CVPR'18】[Learning Latent Super-Events to Detect Multiple Activities in Videos](http://openaccess.thecvf.com/content_cvpr_2018/papers/Piergiovanni_Learning_Latent_Super-Events_CVPR_2018_paper.pdf)|[【Project Page】](https://github.com/piergiaj/super-events-cvpr18)|
|【CVPR'18】[Compressed Video Action Recognition](http://openaccess.thecvf.com/content_cvpr_2018/papers/Wu_Compressed_Video_Action_CVPR_2018_paper.pdf)|[【Project Page】](https://github.com/chaoyuaw/pytorch-coviar)[【Blog(Chinese)1】](https://blog.csdn.net/perfects110/article/details/84329491)[【Blog(Chinese)2】](https://blog.csdn.net/Dongjiuqing/article/details/84678962)|
|【CVPR'18】[Video Representation Learning Using Discriminative Pooling](http://openaccess.thecvf.com/content_cvpr_2018/papers/Wang_Video_Representation_Learning_CVPR_2018_paper.pdf)|[【Project Page】](https://github.com/3xwangDot/SVMP)|
|【CVPR'18】[Optical Flow Guided Feature: A Fast and Robust Motion Representation for Video Action Recognition](https://arxiv.org/abs/1711.11152)|[【Project Page】](https://github.com/kevin-ssy/Optical-Flow-Guided-Feature)|
|【CVPR'18】[NeuralNetwork-Viterbi: A Framework for Weakly Supervised Video Learning Spotlight](https://arxiv.org/abs/1805.06875)|[【Project Page】](https://github.com/alexanderrichard/NeuralNetwork-Viterbi)|
|【CVPR'18】[Temporal Deformable Residual Networks for Action Segmentation in Videos](http://openaccess.thecvf.com/content_cvpr_2018/papers/Lei_Temporal_Deformable_Residual_CVPR_2018_paper.pdf)||

|Reference|
| :--: |
|【知乎】[简评Video Action Recognition 的近期进展](https://zhuanlan.zhihu.com/p/59915784)|
|【知乎】[Video Analysis相关领域解读之Action Recognition(行为识别)](https://zhuanlan.zhihu.com/p/26460437)|
|【CSDN】[3D CNN框架结构各层计算](https://blog.csdn.net/auto1993/article/details/70948249)|
|【CSDN】[Temporal Action Detection (时序动作检测)综述](https://blog.csdn.net/qq_33278461/article/details/80720104)|

#### [1.3.9 Video caption](https://github.com/huuuuusy/CV-Learning-Everyday/blob/master/video-based/T010-Video%20caption.md)

|Paper|Extra Link|
| :--: |:--: |
|【CVPR'18】[Video Captioning via Hierarchical Reinforcement Learning](https://arxiv.org/abs/1711.11135)|[【Blog(Chinese)】](https://cloud.tencent.com/developer/article/1092810)|
|【CVPR'18】[Fine-grained Video Captioning for Sports Narrative](http://openaccess.thecvf.com/content_cvpr_2018/papers/Yu_Fine-Grained_Video_Captioning_CVPR_2018_paper.pdf)|[【Blog(Chinese)】](https://zhuanlan.zhihu.com/p/38292295)|
|【CVPR'18】[Jointly Localizing and Describing Events for Dense Video Captioning](https://arxiv.org/abs/1804.08274)||
|【CVPR'18】[Bidirectional Attentive Fusion with Context Gating for Dense Video Captioning](http://openaccess.thecvf.com/content_cvpr_2018/papers/Wang_Bidirectional_Attentive_Fusion_CVPR_2018_paper.pdf)|[【Project Page】](https://github.com/JaywongWang/DenseVideoCaptioning)[【Blog(Chinese)】](https://zhuanlan.zhihu.com/p/50924797)|
|【CVPR'18】[Interpretable Video Captioning via Trajectory Structured Localization](http://openaccess.thecvf.com/content_cvpr_2018/papers/Wu_Interpretable_Video_Captioning_CVPR_2018_paper.pdf)||
|【CVPR'18】[End-to-End Dense Video Captioning with Masked Transformer](https://arxiv.org/pdf/1804.00819.pdf)|[【Project Page】](https://github.com/salesforce/densecap)|
|【CVPR'18】[Reconstruction Network for Video Captioning](https://arxiv.org/abs/1803.11438)|[【Project Page(Unofficial)】](https://github.com/hobincar/reconstruction-network-for-video-captioning)[【Blog(Chinese)】](https://zhuanlan.zhihu.com/p/50784504)|
|【CVPR'18】[Multimodal Memory Modelling for Video Captioning](https://arxiv.org/abs/1611.05592)||

|Reference|
| :--: |
|【知乎】[Video Analysis 相关领域介绍之Video Captioning(视频to文字描述)](https://zhuanlan.zhihu.com/p/26730181)|

#### [1.3.10 Video generation (GAN)](https://github.com/huuuuusy/CV-Learning-Everyday/blob/master/video-based/T011-Video%20generation%20(GAN).md)

|Paper|Extra Link|
| :--: |:--: |
|【CVPR'18】[MoCoGAN: Decomposing Motion and Content for Video Generation](http://openaccess.thecvf.com/content_cvpr_2018/papers/Tulyakov_MoCoGAN_Decomposing_Motion_CVPR_2018_paper.pdf)|[【Project Page】](https://github.com/sergeytulyakov/mocogan)|
|【CVPR'18】[Learning to Generate Time-Lapse Videos Using Multi-Stage Dynamic Generative Adversarial Networks](https://arxiv.org/abs/1709.07592)||
|【CVPR'18】[Controllable Video Generation with Sparse Trajectories](http://openaccess.thecvf.com/content_cvpr_2018/papers/Hao_Controllable_Video_Generation_CVPR_2018_paper.pdf)|[【Project Page】](https://github.com/zekunhao1995/ControllableVideoGen)|

#### 1.3.11 Video saliency & gaze prediction

|Paper|Extra Link|
| :--: |:--: |
|【CVPR'18】[Cube Padding for Weakly-Supervised Saliency Prediction in 360° Videos](https://arxiv.org/abs/1806.01320)|[【Project Page】](https://github.com/hsientzucheng/CP-360-Weakly-Supervised-Saliency)|
|【CVPR'18】[Gaze Prediction in Dynamic 360deg Immersive Videos](http://openaccess.thecvf.com/content_cvpr_2018/CameraReady/2529.pdf)|[【Project Page】](https://github.com/xuyanyu-shh/VR-EyeTracking)|
|【CVPR'18】[Revisiting Video Saliency: A Large-scale Benchmark and a New Model](http://openaccess.thecvf.com/content_cvpr_2018/papers/Wang_Revisiting_Video_Saliency_CVPR_2018_paper.pdf)|[【Project Page】](https://mmcheng.net/zh/videosal/)[【Blog(Chinese)】](https://blog.csdn.net/weixin_38682454/article/details/88530234)|
|【CVPR'18】[Flow Guided Recurrent Neural Encoder for Video Salient Object Detection](http://openaccess.thecvf.com/content_cvpr_2018/papers/Li_Flow_Guided_Recurrent_CVPR_2018_paper.pdf)|[【Blog(Chinese)1】](https://blog.csdn.net/Dorothy_Xue/article/details/83042331)[【Blog(Chinese)2】](https://www.jianshu.com/p/056353117ea4)|
|【CVPR'18】[Going from Image to Video Saliency: Augmenting Image Salience with Dynamic Attentional Push](http://openaccess.thecvf.com/content_cvpr_2018/papers/Gorji_Going_From_Image_CVPR_2018_paper.pdf)|[【Blog(Chinese)】](https://blog.csdn.net/Dorothy_Xue/article/details/82750300)|

|Reference|
| :--: |
|【CSDN】[视觉显著性检测](https://blog.csdn.net/u012507022/article/details/52863461)|

#### 1.3.12 Video pedestrian tasks

|Paper|Extra Link|
| :--: |:--: |
|【CVPR'18】[Detect-and-Track: Efficient Pose Estimation in Videos](https://arxiv.org/abs/1712.09184)|[【Project Page】](https://github.com/facebookresearch/DetectAndTrack)[【Blog(Chinese)】](https://blog.csdn.net/m0_37644085/article/details/82924949)|
|【CVPR'18】[Diversity Regularized Spatiotemporal Attention for Video-based Person Re-identification](http://openaccess.thecvf.com/content_cvpr_2018/papers/Li_Diversity_Regularized_Spatiotemporal_CVPR_2018_paper.pdf)|[【Project Page(Uncompleted)】](https://github.com/ShuangLI59/Diversity-Regularized-Spatiotemporal-Attention)[【Blog(Chinese)1】](https://blog.csdn.net/baidu_39622935/article/details/84349093)[【Blog(Chinese)2】](https://zhuanlan.zhihu.com/p/35460367)|
|【CVPR'18】[Video Person Re-identification with Competitive Snippet-similarity Aggregation and Co-attentive Snippet Embedding](http://openaccess.thecvf.com/content_cvpr_2018/papers/Chen_Video_Person_Re-Identification_CVPR_2018_paper.pdf)|[【Project Page】](https://github.com/dapengchen123/video_reid)[【Blog(Chinese)】](https://blog.csdn.net/Coder_XiaoHui/article/details/81122373)|
|【CVPR'18】[Exploit the Unknown Gradually: One-Shot Video-Based Person Re-Identification by Stepwise Learning](http://openaccess.thecvf.com/content_cvpr_2018/papers/Wu_Exploit_the_Unknown_CVPR_2018_paper.pdf)|[【Project Page】](https://github.com/Yu-Wu/Exploit-Unknown-Gradually)[【Blog(Chinese)】](https://blog.csdn.net/NGUever15/article/details/88930864)|

|Reference|
| :--: |
|【知乎】[基于深度学习的行人重识别研究综述](https://zhuanlan.zhihu.com/p/31921944)|

#### 1.3.13 Video frame interpolation

|Paper|Extra Link|
| :--: |:--: |
|【CVPR'18】[Super SloMo: High Quality Estimation of Multiple Intermediate Frames for Video Interpolation](https://arxiv.org/abs/1712.00080)|[【Project Page】](https://people.cs.umass.edu/~hzjiang/projects/superslomo/)[【Project Page(Unofficial)】](https://github.com/TheFairBear/Super-SlowMo)[【Blog(Chinese)】](https://zhuanlan.zhihu.com/p/69538032)|
|【CVPR'18】[PhaseNet for Video Frame Interpolation](https://arxiv.org/abs/1804.00884)||
|【CVPR'18】[Context-aware Synthesis for Video Frame Interpolation](https://arxiv.org/abs/1803.10967)||

|Reference|
| :--: |
|【GitHub】[Video-Enhancement](https://github.com/yulunzhang/video-enhancement)|

