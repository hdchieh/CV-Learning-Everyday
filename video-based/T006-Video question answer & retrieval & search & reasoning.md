# Video question answer & retrieval & search & reasoning

**[Motion-Appearance Co-Memory Networks for Video Question Answering](https://arxiv.org/abs/1803.10906)**

**Abstract**

Video Question Answering (QA) is an important task in understanding video temporal structure. We observe that there are three unique attributes of video QA compared with image QA: (1) it deals with long sequences of images containing richer information not only in quantity but also in variety; (2) motion and appearance information are usually correlated with each other and able to provide useful attention cues to the other; (3) different questions require different number of frames to infer the answer. Based these observations, we propose a motion-appearance comemory network for video QA. Our networks are built on concepts from Dynamic Memory Network (DMN) and introduces new mechanisms for video QA. Specifically, there are three salient aspects: (1) a co-memory attention mechanism that utilizes cues from both motion and appearance to generate attention; (2) a temporal conv-deconv network to generate multi-level contextual facts; (3) a dynamic fact ensemble method to construct temporal representation dynamically for different questions. We evaluate our method on TGIF-QA dataset, and the results outperform state-of-the-art significantly on all four tasks of TGIF-QA.

视频问答（QA）是理解视频时间结构的重要任务。视频QA与图像QA相比有三个独特的属性：（1）它处理的是包含更多信息的长序列图像，不仅数量多，而且种类多; （2）运动和外观信息通常相互关联，能够为对方提供有用的注意线索; （3）不同的问题需要不同数量的帧来推断答案。

![](images/0048.png)

![](images/0049.png)

基于这些观察，文章提出了用于视频QA的运动外观存储网络。网络基于动态内存网络（DMN）并引入视频质量保证的新机制。具体来说，有三个突出的方面：（1）共同记忆注意机制，利用运动和外观的提示来产生注意力; （2）生成多层上下文事实的时间conv-deconv网络; （3）动态事实集合方法，动态地构造不同问题的时间表示。文章在TGIF-QA数据集所有四个任务上都优于现有技术。

