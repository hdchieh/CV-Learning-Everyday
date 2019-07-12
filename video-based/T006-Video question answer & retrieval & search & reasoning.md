# Video question answer & retrieval & search & reasoning

**[Motion-Appearance Co-Memory Networks for Video Question Answering, CVPR'18](https://arxiv.org/abs/1803.10906)**

**Abstract**

Video Question Answering (QA) is an important task in understanding video temporal structure. We observe that there are three unique attributes of video QA compared with image QA: (1) it deals with long sequences of images containing richer information not only in quantity but also in variety; (2) motion and appearance information are usually correlated with each other and able to provide useful attention cues to the other; (3) different questions require different number of frames to infer the answer. Based these observations, we propose a motion-appearance comemory network for video QA. Our networks are built on concepts from Dynamic Memory Network (DMN) and introduces new mechanisms for video QA. Specifically, there are three salient aspects: (1) a co-memory attention mechanism that utilizes cues from both motion and appearance to generate attention; (2) a temporal conv-deconv network to generate multi-level contextual facts; (3) a dynamic fact ensemble method to construct temporal representation dynamically for different questions. We evaluate our method on TGIF-QA dataset, and the results outperform state-of-the-art significantly on all four tasks of TGIF-QA.

视频问答（QA）是理解视频时间结构的重要任务。视频QA与图像QA相比有三个独特的属性：（1）它处理的是包含更多信息的长序列图像，不仅数量多，而且种类多; （2）运动和外观信息通常相互关联，能够为对方提供有用的注意线索; （3）不同的问题需要不同数量的帧来推断答案。

![](images/0048.png)

![](images/0049.png)

基于这些观察，文章提出了用于视频QA的运动外观存储网络。网络基于动态内存网络（DMN）并引入视频质量保证的新机制。具体来说，有三个突出的方面：（1）共同记忆注意机制，利用运动和外观的提示来产生注意力; （2）生成多层上下文事实的时间conv-deconv网络; （3）动态事实集合方法，动态地构造不同问题的时间表示。文章在TGIF-QA数据集所有四个任务上都优于现有技术。

**[Finding "It": Weakly-Supervised Reference-Aware Visual Grounding in Instructional Videos, CVPR'18 Oral](http://openaccess.thecvf.com/content_cvpr_2018/papers/Huang_Finding_It_Weakly-Supervised_CVPR_2018_paper.pdf)**

**Abstract**

Grounding textual phrases in visual content with standalone image-sentence pairs is a challenging task. When we consider grounding in instructional videos, this problem becomes profoundly more complex: the latent temporal structure of instructional videos breaks independence assumptions and necessitates contextual understanding for resolving ambiguous visual-linguistic cues. Furthermore, dense annotations and video data scale mean supervised approaches are prohibitively costly. In this work, we propose to tackle this new task with a weakly-supervised framework for reference-aware visual grounding in instructional videos, where only the temporal alignment between the transcription and the video segment are available for supervision. We introduce the visually grounded action graph, a structured representation capturing the latent dependency between grounding and references in video. For optimization, we propose a new reference-aware multiple instance learning (RA-MIL) objective for weak supervision of grounding in videos. We evaluate our approach over unconstrained videos from YouCookII and RoboWatch, augmented with new reference-grounding test set annotations. We demonstrate that our jointly optimized, reference-aware approach simultaneously improves visual grounding, reference-resolution, and generalization to unseen instructional video categories.

使用独立的图像-句子将可视内容中的文本短语grounding是一项具有挑战性的任务。在教学视频中进行visual grounding时，问题变得非常复杂：教学视频的潜在时间结构打破了独立性假设，并且需要用于解决模糊视觉语言线索的语境理解。此外，密集注释和视频数据规模意味着监督方法成本过高。在这项工作中，我们建议用一个弱监督的教学视频中的参考感知视觉基础框架来解决这个新任务，其中只有转录和视频片段之间的时间对齐可用于监督。我们介绍了视觉上接地的动作图，这是一种结构化表示，捕捉了视频中接地和参考之间的潜在依赖关系。为了优化，我们提出了一种新的参考感知多实例学习（RA-MIL）目标，用于弱视频监控。我们评估了来自YouCookII和RoboWatch的无约束视频的方法，并增加了新的参考接地测试集注释。我们证明了我们的联合优化的参考感知方法同时改善了视觉基础，参考分辨率和对看不见的教学视频类别的概括。
CVPR 2018（口头）

[项目地址](https://finding-it.github.io/)