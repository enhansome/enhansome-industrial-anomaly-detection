# Awesome Industrial Anomaly Detection with stars

We discuss public datasets and related studies in detail. Welcome to read our paper and make comments.

[Deep Industrial Image Anomaly Detection: A Survey (Machine Intelligence Research)](https://link.springer.com/article/10.1007/s11633-023-1459-z)

[IM-IAD: Industrial Image Anomaly Detection Benchmark in Manufacturing \[TCYB 2024\]](https://arxiv.org/abs/2301.13359)[\[code\]](https://github.com/M-3LAB/open-iad) ⭐ 171 | 🐛 3 | 🌐 Python | 📅 2025-02-20[\[中文\]](https://blog.csdn.net/m0_63828250/article/details/136891730)

We will keep focusing on this field and updating relevant information.

Keywords: anomaly detection, anomaly segmentation, industrial image, defect detection

[\[Main Page\]](https://github.com/M-3LAB) [\[Survey\]](https://github.com/M-3LAB/awesome-industrial-anomaly-detection) ⭐ 3,752 | 🐛 1 | 📅 2026-08-28 [\[Benchmark\]](https://github.com/M-3LAB/open-iad) ⭐ 171 | 🐛 3 | 🌐 Python | 📅 2025-02-20 [\[Result\]](https://github.com/M-3LAB/IM-IAD) ⭐ 25 | 🐛 0 | 📅 2023-07-04

🔥🔥🔥 Contributions to our repository are welcome. Feel free to categorize the papers and [pull requests](https://github.com/M-3LAB/awesome-industrial-anomaly-detection/pulls) ⭐ 3,752 | 🐛 1 | 📅 2026-08-28.

***

🔥🔥🔥 We have released AD-Copilot, an end-to-end trained MLLM for industrial anomaly detection. Most impressively, AD-Copilot surpasses humans on real industrial inspection tasks! Try it at [\[Code\]](https://github.com/jam-cc/AD-Copilot) ⭐ 64 | 🐛 5 | 🌐 Python | 📅 2026-04-25[\[Demo\]](https://huggingface.co/spaces/jiang-cc/AD-Copilot)

🔥🔥🔥 How well are current MLLMs performing as industrial quality inspectors? Which MLLM performs best in industrial anomaly detection? Please refer to our recent research. [\[ICLR 2025\]](https://arxiv.org/abs/2410.09453)[\[Github\]](https://github.com/jam-cc/MMAD) ⭐ 271 | 🐛 1 | 🌐 Python | 📅 2026-01-14

🔥🔥🔥 We compare different types of anomaly synthesis methods in detail. Welcome to make comments.

ASBench: Image Anomalies Synthesis Benchmark for Anomaly Detection [\[paper\]](https://arxiv.org/abs/2510.07927)

A Survey on Industrial Anomalies Synthesis [\[paper\]](https://arxiv.org/abs/2502.16412)[\[github\]](https://github.com/M-3LAB/awesome-anomaly-synthesis) ⭐ 80 | 🐛 0 | 📅 2025-11-18

🔥🔥🔥 3D Anomaly Detection: A Survey [\[paper\]](https://www.researchgate.net/publication/398334588_3D_Anomaly_Detection_A_Survey?_tp=eyJjb250ZXh0Ijp7InBhZ2UiOiJzcG90bGlnaHQiLCJwcmV2aW91c1BhZ2UiOiJwcm9maWxlIiwicG9zaXRpb24iOiJwYWdlQ29udGVudCJ9fQ) [\[github\]](https://github.com/M-3LAB/awesome-3d-anomaly-detection) ⭐ 135 | 🐛 0 | 📅 2026-08-29

***

## Table of Contents

* [Awesome Industrial Anomaly Detection ](#awesome-industrial-anomaly-detection-)
  * [Table of Contents](#table-of-contents)
* [SOTA methods with code](#sota-methods-with-code)
* [Recommended Benchmarks](#recommended-benchmarks)
* [Recent research](#recent-research)
  * [ECCV 2026](#eccv-2026)
  * [ICML 2026](#icml-2026)
  * [CVPR 2026](#cvpr-2026)
  * [ICLR 2026](#iclr-2026)
  * [AAAI 2026](#aaai-2026)
  * [NeurIPS 2025](#neurips-2025)
  * [KDD 2025](#kdd-2025)
  * [ICCV 2025](#iccv-2025)
  * [ICML 2025](#icml-2025)
  * [CVPR 2025](#cvpr-2025)
* [Paper Tree (Classification of representative methods)](#paper-tree-classification-of-representative-methods)
* [Timeline](#timeline)
* [Paper list for industrial image anomaly detection](#paper-list-for-industrial-image-anomaly-detection)
* [Related Survey, Benchmark, and Framework](#related-survey-benchmark-and-framework)
* [2 Unsupervised AD](#2-unsupervised-ad)
  * [2.1 Feature-Embedding-based Methods](#21-feature-embedding-based-methods)
    * [2.1.1 Teacher-Student](#211-teacher-student)
    * [2.1.2 One-Class Classification (OCC)](#212-one-class-classification-occ)
    * [2.1.3 Distribution-Map](#213-distribution-map)
    * [2.1.4 Memory Bank](#214-memory-bank)
    * [2.1.5 Vison Language AD](#215-vison-language-ad)
  * [2.2 Reconstruction-Based Methods](#22-reconstruction-based-methods)
    * [2.2.1 Autoencoder (AE)](#221-autoencoder-ae)
    * [2.2.2 Generative Adversarial Networks (GANs)](#222-generative-adversarial-networks-gans)
    * [2.2.3 Transformer](#223-transformer)
    * [2.2.4 Diffusion Model](#224-diffusion-model)
    * [2.2.5 Others](#225-others)
  * [2.3 Supervised AD](#23-supervised-ad)
    * [More Normal Samples With (Less Abnormal Samples or Weak Labels)](#more-normal-samples-with-less-abnormal-samples-or-weak-labels)
    * [More Abnormal Samples](#more-abnormal-samples)
* [3 Other Research Direction](#3-other-research-direction)
  * [3.1 Zero/Few-Shot AD](#31-zerofew-shot-ad)
    * [Zero-Shot AD](#zero-shot-ad)
    * [Few-Shot AD](#few-shot-ad)
  * [3.2 Noisy AD](#32-noisy-ad)
  * [3.3 Anomaly Synthesis \[awesome-anomaly-synthesis\]](#33-anomaly-synthesis-awesome-anomaly-synthesis)
  * [3.4 RGBD AD](#34-rgbd-ad)
  * [3.5 3D AD](#35-3d-ad)
  * [3.6 Continual AD](#36-continual-ad)
  * [3.7 Uniform/Multi-Class AD](#37-uniformmulti-class-ad)
  * [3.8 Logical AD](#38-logical-ad)
  * [3.9 MLLM-based AD](#39-mllm-based-ad)
  * [3.10 Video IAD](#310-video-iad)
  * [Other settings](#other-settings)
    * [TTT binary segmentation](#ttt-binary-segmentation)
    * [MoE with TTA](#moe-with-tta)
    * [Adversary Attack](#adversary-attack)
    * [Defect Classification](#defect-classification)
    * [Rubustness](#rubustness)
    * [Universal Task](#universal-task)
* [4 Dataset](#4-dataset)
  * [BibTex Citation](#bibtex-citation)
  * [Star History](#star-history)

# SOTA methods with code

| Title                                                                                                                                                                                                                                                                                                                                                                                         |  Venue  | Date |                                                                  Code                                                                  |                      topic                     |
| :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :-----: | :--: | :------------------------------------------------------------------------------------------------------------------------------------: | :--------------------------------------------: |
| ![Star](https://img.shields.io/github/stars/hq-deng/RD4AD.svg?style=social\&label=Star) <br> [**Anomaly Detection via Reverse Distillation from One-Class Embedding**](https://openaccess.thecvf.com/content/CVPR2022/html/Deng_Anomaly_Detection_via_Reverse_Distillation_From_One-Class_Embedding_CVPR_2022_paper.html) <br>                                                                |   CVPR  | 2022 |                         [Github](https://github.com/hq-deng/RD4AD) ⭐ 252 \| 🐛 21 \| 🌐 Python \| 📅 2023-06-23                        |                 Teacher-Student                |
| ![Star](https://img.shields.io/github/stars/guojiajeremy/Dinomaly.svg?style=social\&label=Star) <br> [**Dinomaly: The Less Is More Philosophy in Multi-Class Unsupervised Anomaly Detection**](https://openaccess.thecvf.com/content/CVPR2025/html/Guo_Dinomaly_The_Less_Is_More_Philosophy_in_Multi-Class_Unsupervised_Anomaly_CVPR_2025_paper.html) <br>                                    |   CVPR  | 2025 |                     [Github](https://github.com/guojiajeremy/Dinomaly) ⭐ 516 \| 🐛 30 \| 🌐 Python \| 📅 2026-05-21                    |               Multi-Class Unified              |
| ![Star](https://img.shields.io/github/stars/guojiajeremy/Dinomaly2.svg?style=social\&label=Star) <br> [**One Dinomaly2 Detect Them All: A Unified Framework for Full-Spectrum Unsupervised Anomaly Detection**](https://arxiv.org/abs/2510.17611) <br>                                                                                                                                        |  Arxiv  | 2025 |                     [Github](https://github.com/guojiajeremy/Dinomaly2) ⭐ 102 \| 🐛 4 \| 🌐 Python \| 📅 2026-05-30                    | Multi-Class, Multi-View, Multi-Modal, Few-shot |
| ![Star](https://img.shields.io/github/stars/tientrandinh/Revisiting-Reverse-Distillation.svg?style=social\&label=Star) <br> [**Revisiting Reverse Distillation for Anomaly Detection**](https://openaccess.thecvf.com/content/CVPR2023/html/Tien_Revisiting_Reverse_Distillation_for_Anomaly_Detection_CVPR_2023_paper.html) <br>                                                             |   CVPR  | 2023 |          [Github](https://github.com/tientrandinh/Revisiting-Reverse-Distillation) ⭐ 170 \| 🐛 9 \| 🌐 Python \| 📅 2023-12-28         |                 Teacher-Student                |
| ![Star](https://img.shields.io/github/stars/DonaldRR/SimpleNet.svg?style=social\&label=Star) <br> [**SimpleNet: A Simple Network for Image Anomaly Detection and Localization**](https://openaccess.thecvf.com/content/CVPR2023/html/Liu_SimpleNet_A_Simple_Network_for_Image_Anomaly_Detection_and_Localization_CVPR_2023_paper.html) <br>                                                   |   CVPR  | 2023 |                      [Github](https://github.com/DonaldRR/SimpleNet) ⭐ 603 \| 🐛 30 \| 🌐 Python \| 📅 2024-08-12                      |            One-Class-Classification            |
| ![Star](https://img.shields.io/github/stars/gudovskiy/cflow-ad.svg?style=social\&label=Star) <br> [**Real-time unsupervised anomaly detection with localization via conditional normalizing flows**](https://openaccess.thecvf.com/content/WACV2022/html/Gudovskiy_CFLOW-AD_Real-Time_Unsupervised_Anomaly_Detection_With_Localization_via_Conditional_Normalizing_WACV_2022_paper.html) <br> |   WACV  | 2022 |                      [Github](https://github.com/gudovskiy/cflow-ad) ⭐ 270 \| 🐛 25 \| 🌐 Python \| 📅 2023-08-18                      |                Distribution Map                |
| ![Star](https://img.shields.io/github/stars/amazon-science/patchcore-inspection.svg?style=social\&label=Star) <br> [**Towards total recall in industrial anomaly detection**](https://openaccess.thecvf.com/content/CVPR2022/html/Roth_Towards_Total_Recall_in_Industrial_Anomaly_Detection_CVPR_2022_paper.html) <br>                                                                        |   CVPR  | 2022 |             [Github](https://github.com/amazon-science/patchcore-inspection) ⭐ 1,378 \| 🐛 83 \| 🌐 Python \| 📅 2024-07-10            |                   Memory-bank                  |
| ![Star](https://img.shields.io/github/stars/vitjanz/draem.svg?style=social\&label=Star) <br> [**Draem-a discriminatively trained reconstruction embedding for surface anomaly detection**](https://openaccess.thecvf.com/content/ICCV2021/html/Zavrtanik_DRAEM_-_A_Discriminatively_Trained_Reconstruction_Embedding_for_Surface_Anomaly_ICCV_2021_paper.html) <br>                           |   ICCV  | 2021 |                         [Github](https://github.com/vitjanz/draem) ⭐ 291 \| 🐛 18 \| 🌐 Python \| 📅 2023-01-02                        |              Reconstruction-based              |
| ![Star](https://img.shields.io/github/stars/VitjanZ/DSR_anomaly_detection.svg?style=social\&label=Star) <br> [**DSR: A dual subspace re-projection network for surface anomaly detection**](https://link.springer.com/chapter/10.1007/978-3-031-19821-2_31) <br>                                                                                                                              |   ECCV  | 2022 |                  [Github](https://github.com/VitjanZ/DSR_anomaly_detection) ⭐ 56 \| 🐛 4 \| 🌐 Python \| 📅 2022-07-18                 |              Reconstruction-based              |
| ![Star](https://img.shields.io/github/stars/zhangzjn/ocr-gan.svg?style=social\&label=Star) <br> [**Omni-frequency Channel-selection Representations for Unsupervised Anomaly Detection**](https://ieeexplore.ieee.org/abstract/document/10192551/) <br>                                                                                                                                       |   TIP   | 2023 |                        [Github](https://github.com/zhangzjn/ocr-gan) ⭐ 52 \| 🐛 4 \| 🌐 Python \| 📅 2022-07-14                        |              Reconstruction-based              |
| ![Star](https://img.shields.io/github/stars/cnulab/RealNet.svg?style=social\&label=Star) <br> [**RealNet: A Feature Selection Network with Realistic Synthetic Anomaly for Anomaly Detection**](https://arxiv.org/abs/2403.05897) <br>                                                                                                                                                        |   CVPR  | 2024 |                        [Github](https://github.com/cnulab/RealNet) ⭐ 426 \| 🐛 79 \| 🌐 Python \| 📅 2025-02-12                        |              Reconstruction-based              |
| ![Star](https://img.shields.io/github/stars/MediaBrain-SJTU/RegAD.svg?style=social\&label=Star) <br> [**Registration based few-shot anomaly detection**](https://link.springer.com/chapter/10.1007/978-3-031-20053-3_18) <br>                                                                                                                                                                 |   ECCV  | 2022 |                     [Github](https://github.com/MediaBrain-SJTU/RegAD) ⭐ 323 \| 🐛 14 \| 🌐 Python \| 📅 2022-09-07                    |                    Few Shot                    |
| ![Star](https://img.shields.io/github/stars/CASIA-IVA-Lab/AnomalyGPT.svg?style=social\&label=Star) <br> [**AnomalyGPT: Detecting Industrial Anomalies using Large Vision-Language Models**](https://arxiv.org/abs/2308.15366) <br>                                                                                                                                                            |   AAAI  | 2024 |                  [Github](https://github.com/CASIA-IVA-Lab/AnomalyGPT) ⭐ 1,132 \| 🐛 61 \| 🌐 Python \| 📅 2023-12-20                  |                    Few Shot                    |
| ![Star](https://img.shields.io/github/stars/jam-cc/AD-Copilot.svg?style=social\&label=Star) <br> [**AD-Copilot: A Vision-Language Assistant for Industrial Anomaly Detection via Visual In-context Comparison**](https://arxiv.org/abs/2603.13779) <br>                                                                                                                                       |  arxiv  | 2026 |                        [Github](https://github.com/jam-cc/AD-Copilot) ⭐ 64 \| 🐛 5 \| 🌐 Python \| 📅 2026-04-25                       |                      MLLM                      |
| ![Star](https://img.shields.io/github/stars/Choubo/DRA.svg?style=social\&label=Star) <br> [**Catching Both Gray and Black Swans: Open-set Supervised Anomaly Detection**](https://openaccess.thecvf.com/content/CVPR2022/html/Ding_Catching_Both_Gray_and_Black_Swans_Open-Set_Supervised_Anomaly_Detection_CVPR_2022_paper.html) <br>                                                        |   CVPR  | 2022 |                           [Github](https://github.com/Choubo/DRA) ⭐ 95 \| 🐛 7 \| 🌐 Python \| 📅 2022-04-11                           |              Few abnormal samples              |
| ![Star](https://img.shields.io/github/stars/xcyao00/BGAD.svg?style=social\&label=Star) <br> [**Explicit Boundary Guided Semi-Push-Pull Contrastive Learning for Supervised Anomaly Detection**](https://openaccess.thecvf.com/content/CVPR2023/html/Yao_Explicit_Boundary_Guided_Semi-Push-Pull_Contrastive_Learning_for_Supervised_Anomaly_Detection_CVPR_2023_paper.html) <br>              |   CVPR  | 2023 |                          [Github](https://github.com/xcyao00/BGAD) ⭐ 98 \| 🐛 19 \| 🌐 Python \| 📅 2026-06-02                         |              Few abnormal samples              |
| ![Star](https://img.shields.io/github/stars/tianyu0207/IGD.svg?style=social\&label=Star) <br> [**Deep one-class classification via interpolated gaussian descriptor**](https://ojs.aaai.org/index.php/AAAI/article/view/19915) <br>                                                                                                                                                           |   AAAI  | 2022 |                         [Github](https://github.com/tianyu0207/IGD) ⭐ 69 \| 🐛 5 \| 🌐 Python \| 📅 2025-10-29                         |                    Noisy AD                    |
| ![Star](https://img.shields.io/github/stars/TencentYoutuResearch/AnomalyDetection-SoftPatch.svg?style=social\&label=Star) <br> [**SoftPatch: Unsupervised Anomaly Detection with Noisy Data**](https://proceedings.neurips.cc/paper_files/paper/2022/hash/637a456d89289769ac1ab29617ef7213-Abstract-Conference.html) <br>                                                                     | NeurIPS | 2022 |         [Github](https://github.com/TencentYoutuResearch/AnomalyDetection-SoftPatch) ⭐ 96 \| 🐛 5 \| 🌐 Python \| 📅 2024-07-13        |                    Noisy AD                    |
| ![Star](https://img.shields.io/github/stars/DeclanMcIntosh/InReaCh.svg?style=social\&label=Star) <br> [**Inter-Realization Channels: Unsupervised Anomaly Detection Beyond One-Class Classification**](https://openaccess.thecvf.com/content/ICCV2023/html/McIntosh_Inter-Realization_Channels_Unsupervised_Anomaly_Detection_Beyond_One-Class_Classification_ICCV_2023_paper.html) <br>      |   ICCV  | 2023 |                     [Github](https://github.com/DeclanMcIntosh/InReaCh) ⭐ 15 \| 🐛 0 \| 🌐 Python \| 📅 2026-01-12                     |                    Noisy AD                    |
| ![Star](https://img.shields.io/github/stars/shirowalker/UCAD.svg?style=social\&label=Star) <br> [**Unsupervised Continual Anomaly Detection with Contrastively-learned Prompt**](https://ojs.aaai.org/index.php/AAAI/article/view/28153) <br>                                                                                                                                                 |   AAAI  | 2024 |                        [Github](https://github.com/shirowalker/UCAD) ⭐ 115 \| 🐛 0 \| 🌐 Python \| 📅 2024-08-06                       |                  Continual AD                  |
| ![Star](https://img.shields.io/github/stars/zhiyuanyou/UniAD.svg?style=social\&label=Star) <br> [**A Unified Model for Multi-class Anomaly Detection**](https://proceedings.neurips.cc/paper_files/paper/2022/hash/1d774c112926348c3e25ea47d87c835b-Abstract-Conference.html) <br>                                                                                                            | NeurIPS | 2022 |                        [Github](https://github.com/zhiyuanyou/UniAD) ⭐ 331 \| 🐛 1 \| 🌐 Python \| 📅 2022-11-22                       |               Multi-class unified              |
| ![Star](https://img.shields.io/github/stars/RuiyingLu/HVQ-Trans.svg?style=social\&label=Star) <br> [**Hierarchical Vector Quantized Transformer for Multi-class Unsupervised Anomaly Detection**](https://openreview.net/pdf?id=clJTNssgn6) <br>                                                                                                                                              | NeurIPS | 2023 |                       [Github](https://github.com/RuiyingLu/HVQ-Trans) ⭐ 49 \| 🐛 6 \| 🌐 Python \| 📅 2024-04-20                      |               Multi-class unified              |
| ![Star](https://img.shields.io/github/stars/nomewang/M3DM.svg?style=social\&label=Star) <br> [**Multimodal Industrial Anomaly Detection via Hybrid Fusion**](https://openaccess.thecvf.com/content/CVPR2023/html/Wang_Multimodal_Industrial_Anomaly_Detection_via_Hybrid_Fusion_CVPR_2023_paper.html) <br>                                                                                    |   CVPR  | 2023 |                         [Github](https://github.com/nomewang/M3DM) ⭐ 215 \| 🐛 22 \| 🌐 Python \| 📅 2023-09-08                        |                      RGBD                      |
| ![Star](https://img.shields.io/github/stars/M-3LAB/Real3D-AD.svg?style=social\&label=Star) <br> [**Real3D-AD: A Dataset of Point Cloud Anomaly Detection**](https://openreview.net/pdf?id=zGthDp4yYe) <br>                                                                                                                                                                                    | NeurIPS | 2023 |                        [Github](https://github.com/M-3LAB/Real3D-AD) ⭐ 162 \| 🐛 0 \| 🌐 Python \| 📅 2024-03-13                       |                   Point Cloud                  |
| ![Star](https://img.shields.io/github/stars/hq-deng/AnoVL.svg?style=social\&label=Star) <br> [**AnoVL: Adapting Vision-Language Models for Unified Zero-shot Anomaly Localization**](https://arxiv.org/abs/2308.15939) <br>                                                                                                                                                                   |  arxiv  | 2023 |                          [Github](https://github.com/hq-deng/AnoVL) ⭐ 57 \| 🐛 9 \| 🌐 Python \| 📅 2023-09-07                         |                    Zero Shot                   |
| ![Star](https://img.shields.io/github/stars/caoyunkang/GroundedSAM-zero-shot-anomaly-detection.svg?style=social\&label=Star) <br> [**Segment Any Anomaly without Training via Hybrid Prompt Regularization**](https://arxiv.org/abs/2305.10724) <br>                                                                                                                                          |  arxiv  | 2023 | [Github](https://github.com/caoyunkang/GroundedSAM-zero-shot-anomaly-detection) ⭐ 844 \| 🐛 12 \| 🌐 Jupyter Notebook \| 📅 2025-02-22 |                    Zero Shot                   |
| ![Star](https://img.shields.io/github/stars/oopil/PSAD_logical_anomaly_detection.svg?style=social\&label=Star) <br> [**PSAD: Few Shot Part Segmentation Reveals Compositional Logic for Industrial Anomaly Detection**](https://ojs.aaai.org/index.php/AAAI/article/view/28703) <br>                                                                                                          |   AAAI  | 2024 |              [Github](https://github.com/oopil/PSAD_logical_anomaly_detection) ⭐ 65 \| 🐛 10 \| 🌐 Python \| 📅 2025-05-09             |                Logical/Few Shot                |
| ![Star](https://img.shields.io/github/stars/MaticFuc/SALAD.svg?style=social\&label=Star) <br> [**SALAD -- Semantics-Aware Logical Anomaly Detection**](https://arxiv.org/abs/2509.02101) <br>                                                                                                                                                                                                 |   ICCV  | 2025 |                         [Github](https://github.com/MaticFuc/SALAD) ⭐ 49 \| 🐛 10 \| 🌐 Python \| 📅 2025-10-03                        |                     Logical                    |
| ![Star](https://img.shields.io/github/stars/YoojLee/Uniformaly.svg?style=social\&label=Star) <br> [**UniFormaly: Towards Task-Agnostic Unified Framework for Visual Anomaly Detection**](https://arxiv.org/abs/2307.12540) <br>                                                                                                                                                               |  arxiv  | 2023 |                       [Github](https://github.com/YoojLee/Uniformaly) ⭐ 16 \| 🐛 5 \| 🌐 Python \| 📅 2023-09-15                       |               Multi-class unified              |

# Recommended Benchmarks

| Title                                                                                                                                                                                                                                                                                 | Venue | Date |                                                   Code                                                  |   topic   |
| :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | :---: | :--: | :-----------------------------------------------------------------------------------------------------: | :-------: |
| ![Star](https://img.shields.io/github/stars/open-edge-platform/anomalib.svg?style=social\&label=Star) <br> [**Anomalib: A Deep Learning Library for Anomaly Detection**](https://ieeexplore.ieee.org/abstract/document/9897283/) <br>                                                 |  ICIP | 2022 | [Github](https://github.com/open-edge-platform/anomalib) ⭐ 6,106 \| 🐛 75 \| 🌐 Python \| 📅 2026-09-01 | Benchmark |
| ![Star](https://img.shields.io/github/stars/M-3LAB/open-iad.svg?style=social\&label=Star) <br> [**IM-IAD: Industrial Image Anomaly Detection Benchmark in Manufacturing**](https://arxiv.org/abs/2301.13359) <br>                                                                     |  TCYB | 2024 |         [Github](https://github.com/M-3LAB/open-iad) ⭐ 171 \| 🐛 3 \| 🌐 Python \| 📅 2025-02-20        | Benchmark |
| ![Star](https://img.shields.io/github/stars/zhangzjn/ader.svg?style=social\&label=Star) <br> [**ADer: A Comprehensive Benchmark for Multi-class Visual Anomaly Detection**](http://arxiv.org/pdf/2406.03262v1) <br>                                                                   | arxiv | 2024 |         [Github](https://github.com/zhangzjn/ader) ⭐ 350 \| 🐛 39 \| 🌐 Python \| 📅 2025-01-29         | Benchmark |
| ![Star](https://img.shields.io/github/stars/jam-cc/MMAD.svg?style=social\&label=Star) <br> [**MMAD: The First-Ever Comprehensive Benchmark for Multimodal Large Language Models in Industrial Anomaly Detection**](https://arxiv.org/abs/2410.09453) <br>                             |  ICLR | 2024 |           [Github](https://github.com/jam-cc/MMAD) ⭐ 271 \| 🐛 1 \| 🌐 Python \| 📅 2026-01-14          | Benchmark |
| ![Star](https://img.shields.io/github/stars/en-research/RobustMAD.svg?style=social\&label=Star) <br> [**RobustMAD: Evaluating Real-World Robustness of Multimodal Small Language Models for Deployable Anomaly Detection Assistants**](https://openreview.net/pdf?id=skrA9UYNIZ) <br> |  TMLR | 2026 |       [Github](https://github.com/en-research/RobustMAD) ⭐ 6 \| 🐛 1 \| 🌐 Python \| 📅 2026-08-13      | Benchmark |

# Recent research

## ECCV 2026

* ArcAD: Anomaly-Rectified Calibration for Cold-Start Supervised Anomaly Detection [\[ECCV 2026\]](https://arxiv.org/abs/2607.02252)[\[code\]](https://github.com/LGC-AD/ArcAD) ⭐ 14 | 🐛 1 | 🌐 Python | 📅 2026-07-04
* LogiCo: A Unified Framework for Logical and Structural Anomaly Detection [\[ECCV 2026\]](https://arxiv.org/abs/2606.28688)[\[code\]](https://github.com/cnulab/LogiCo) ⭐ 11 | 🐛 1 | 🌐 Python | 📅 2026-06-30
* CMDS-AD: Cross-Modal Dual-Stream Decoupling for Few-Shot Anomaly Detection [\[ECCV 2026\]](https://arxiv.org/abs/2606.20300)[\[code\]](https://github.com/Junhaocai27/CMDS-AD) ⭐ 10 | 🐛 0 | 🌐 Python | 📅 2026-09-01
* ReFP-AD: Rectified Flow Preconditioning for Energy-Based Anomaly Detection [\[ECCV 2026\]](https://arxiv.org/abs/2608.01793)[\[code\]](https://github.com/CLendering/ReFP-AD) ⭐ 8 | 🐛 0 | 🌐 Python | 📅 2026-08-17
* DeCo: Zero-Shot Industrial Anomaly Generation through Decoupling and Recoupling [\[ECCV 2026\]](https://arxiv.org/abs/2608.07904)[\[code\]](https://github.com/HUST-SLOW/DeCo) ⭐ 5 | 🐛 2 | 📅 2026-06-25
* UniScale: Arbitrary-Scale Industrial Anomaly Generation [\[ECCV 2026\]](https://arxiv.org/abs/2608.07864)[\[code\]](https://github.com/HUST-SLOW/UniScale) ⭐ 4 | 🐛 2 | 📅 2026-06-25
* IMMoE: Incomplete Multi-View Anomaly Detection via Mixture of View Experts Fusion [\[ECCV 2026\]](https://arxiv.org/abs/2607.19032)[\[code\]](https://github.com/HULEI7/IMMoE) ⭐ 4 | 🐛 1 | 🌐 Python | 📅 2026-08-16
* Rethinking Continual Anomaly Detection on the Edge: Benchmarking Under Realistic Industrial Conditions [\[ECCV 2026\]](https://arxiv.org/abs/2605.24251)[\[code\]](https://github.com/Continue-Edge-AI-Lab/Rethinking-Continual-AD) ⭐ 3 | 🐛 0 | 🌐 Python | 📅 2026-06-29
* Defect-aware Hybrid Prompt Optimization via Progressive Tuning for Zero-Shot Multi-type Anomaly Detection and Segmentation [\[ECCV 2026\]](https://arxiv.org/abs/2512.09446)
* PADFormer: Pose-agnostic Anomaly Detection from Sparse View Images [\[ECCV 2026 Oral\]](https://arxiv.org/abs/2608.04210)
* O-VAD: Industrial Video Anomaly Detection through Object-Centric Tracking and Reasoning [\[ECCV 2026\]](https://arxiv.org/abs/2607.18142)[\[code\]](https://o-vad.github.io/)
* Global Logic and Local Search: Dual-Stream Multimodal In-Context Learning for Verifiable Industrial Anomaly Detection [\[ECCV 2026\]](https://arxiv.org/abs/2607.03817)
* Robust Zero-shot Anomaly Detection under Limited Auxiliary Anomaly Priors [\[ECCV 2026\]](https://arxiv.org/abs/2606.29428)
* DeCoFlow: Structural Decomposition of Normalizing Flows for Continual Anomaly Detection [\[ECCV 2026\]](https://arxiv.org/abs/2606.26687)
* MATCH: Flow Matching for Multi-View Anomaly Detection [\[ECCV 2026\]](https://arxiv.org/abs/2606.24375)

## ICML 2026

* Is Training Necessary for Anomaly Detection? [\[ICML 2026\]](https://arxiv.org/abs/2601.22763)[\[code\]](https://github.com/longkukuhi/RAD) ⭐ 87 | 🐛 0 | 🌐 Python | 📅 2026-06-11
* Memory-Distilled Selection for Noise-Robust Anomaly Detection [\[ICML 2026\]](https://arxiv.org/abs/2605.26676)[\[code\]](https://github.com/SirojbekSafarov/MeDS) ⭐ 18 | 🐛 0 | 🌐 Python | 📅 2026-06-17
* Mixture Prototype Flow Matching for Open-Set Supervised Anomaly Detection [\[ICML 2026\]](https://arxiv.org/abs/2605.02438)[\[code\]](https://github.com/fuyunwang/MPFM-OSAD) ⭐ 5 | 🐛 1 | 🌐 Python | 📅 2026-05-19
* CoGeoAD: Hierarchical Color-Geometric Fusion with Multi-View Attention for Zero-Shot 3D Anomaly Detection [\[ICML 2026\]](https://icml.cc/virtual/2026/poster/62489)
* Formally Exploring Visual Anomaly Detection Evaluation Metrics [\[ICML 2026\]](https://icml.cc/virtual/2026/poster/65518)
* Remove the Ambiguity: Few-shot Multimodal Anomaly Detection Using Crossmodal Feature Replacers [\[ICML 2026\]](https://icml.cc/virtual/2026/poster/62880)
* Anomaly-Preference Image Generation [\[ICML 2026\]](https://arxiv.org/abs/2605.02439)

## CVPR 2026

* UniMMAD: Unified Multi-Modal and Multi-Class Anomaly Detection via MoE-Driven Feature Decompression [\[CVPR 2026\]](https://arxiv.org/abs/2509.25934)[\[code\]](https://github.com/yuanzhao-CVLAB/UniMMAD) ⭐ 245 | 🐛 4 | 🌐 Python | 📅 2026-05-20
* SubspaceAD: Training-Free Few-Shot Anomaly Detection via Subspace Modeling [\[CVPR 2026\]](https://arxiv.org/abs/2602.23013)[\[code\]](https://github.com/CLendering/SubspaceAD) ⭐ 241 | 🐛 1 | 🌐 Python | 📅 2026-08-17
* VisualAD: Language-Free Zero-Shot Anomaly Detection via Vision Transformer [\[CVPR 2026\]](https://arxiv.org/abs/2603.07952)[\[code\]](https://github.com/7HHHHH/VisualAD) ⭐ 126 | 🐛 3 | 🌐 Python | 📅 2026-06-07
* One-to-More: High-Fidelity Training-Free Anomaly Generation with Attention Control [\[CVPR 2026\]](https://arxiv.org/abs/2603.18093)[\[code\]](https://github.com/echrao/O2MAG) ⭐ 67 | 🐛 3 | 🌐 Jupyter Notebook | 📅 2026-03-26
* AnomalyVFM -- Transforming Vision Foundation Models into Zero-Shot Anomaly Detectors [\[CVPR 2026\]](https://arxiv.org/abs/2601.20524)[\[code\]](https://github.com/MaticFuc/AnomalyVFM) ⭐ 66 | 🐛 2 | 🌐 Python | 📅 2026-04-21
* InvAD: Inversion-based Reconstruction-Free Anomaly Detection with Diffusion Models [\[CVPR 2026\]](https://arxiv.org/abs/2504.05662)[\[code\]](https://github.com/SkyShunsuke/InversionAD) ⭐ 62 | 🐛 2 | 🌐 Python | 📅 2026-07-27
* CoPS: Conditional Prompt Synthesis for Zero-Shot Anomaly Detection [\[CVPR 2026 Findings\]](https://arxiv.org/abs/2508.03447)[\[code\]](https://github.com/cqylunlun/CoPS) ⭐ 52 | 🐛 1 | 🌐 Python | 📅 2026-06-02
* RAID: Retrieval-Augmented Anomaly Detection [\[CVPR 2026\]](https://arxiv.org/abs/2602.19611)[\[code\]](https://github.com/Mingxiu-Cai/RAID) ⭐ 46 | 🐛 6 | 🌐 Python | 📅 2026-04-20
* MoECLIP: Patch-Specialized Experts for Zero-shot Anomaly Detection [\[CVPR 2026\]](https://arxiv.org/abs/2603.03101)[\[code\]](https://github.com/CoCoRessa/MoECLIP) ⭐ 37 | 🐛 1 | 🌐 Python | 📅 2026-05-28
* GS-CLIP: Zero-shot 3D Anomaly Detection by Geometry-Aware Prompt and Synergistic View Representation Learning [\[CVPR 2026\]](https://arxiv.org/abs/2602.19206)[\[code\]](https://github.com/zhushengxinyue/GS-CLIP) ⭐ 25 | 🐛 5 | 🌐 Python | 📅 2026-03-07
* Back to Point: Exploring Point-Language Models for Zero-Shot 3D Anomaly Detection [\[CVPR 2026\]](https://arxiv.org/abs/2603.21511)[\[code\]](https://github.com/wistful-8029/BTP-3DAD) ⭐ 23 | 🐛 4 | 🌐 Python | 📅 2026-06-22
* FB-CLIP: Fine-Grained Zero-Shot Anomaly Detection with Foreground-Background Disentanglement [\[CVPR 2026\]](https://arxiv.org/abs/2603.19608)[\[code\]](https://github.com/Xi-Mu-Yu/FB-CLIP) ⭐ 19 | 🐛 0 | 🌐 Python | 📅 2026-07-03
* PDD: Manifold-Prior Diverse Distillation for Medical Anomaly Detection [\[CVPR 2026\]](https://arxiv.org/abs/2603.07142)[\[code\]](https://github.com/OxygenLu/PDD) ⭐ 18 | 🐛 1 | 🌐 Python | 📅 2026-08-25
* MAGIC: Few-Shot Mask-Guided Anomaly Inpainting with Prompt Perturbation, Spatially Adaptive Guidance, and Context Awareness [\[CVPR 2026 Findings\]](https://arxiv.org/abs/2507.02314)[\[code\]](https://github.com/SpatialAILab/MAGIC-Anomaly-generation) ⭐ 12 | 🐛 0 | 🌐 Python | 📅 2026-04-10
* AG-VAS: Anchor-Guided Zero-Shot Visual Anomaly Segmentation with Large Multimodal Models [\[CVPR 2026\]](https://arxiv.org/abs/2603.01305)[\[code\]](https://github.com/xiaozhen228/AG-VAS) ⭐ 11 | 🐛 2 | 📅 2026-06-30
* Reasoning-Driven Anomaly Detection and Localization with Image-Level Supervision [\[CVPR 2026\]](https://arxiv.org/abs/2603.27179)[\[code\]](https://github.com/YizhouJin313/ReADL) ⭐ 5 | 🐛 2 | 📅 2026-03-11
* Hierarchical Point-Patch Fusion with Adaptive Patch Codebook for 3D Shape Anomaly Detection [\[CVPR 2026\]](https://arxiv.org/abs/2604.03972)[\[code\]](https://github.com/alexandor91/Shape-Anomaly-Codebook) ⭐ 4 | 🐛 2 | 🌐 Python | 📅 2026-05-19
* Bidirectional Multimodal Prompt Learning with Scale-Aware Training for Few-Shot Multi-Class Anomaly Detection [\[CVPR 2026\]](https://arxiv.org/abs/2408.13516)[\[code\]](https://github.com/YoojLee/AnoPLe) ⭐ 1 | 🐛 2 | 📅 2026-03-31
* Towards an Incremental Unified Multimodal Anomaly Detection: Augmenting Multimodal [\[CVPR 2026\]](https://arxiv.org/abs/2603.02629)
* DLVP-CLIP: Enhancing Fine-Grained Zero-Shot Anomaly Detection via Dynamic Local Visual Prompting [\[CVPR 2026\]](https://cvpr.thecvf.com/virtual/2026/poster/39303)
* Wavelet-Driven 3D Anomaly Detection under Pose-Agnostic and Sparse-View [\[CVPR 2026\]](https://cvpr.thecvf.com/virtual/2026/poster/38024)
* FastRef:Fast Prototype Refinement for Few-Shot Industrial Anomaly Detection [\[CVPR 2026\]](https://arxiv.org/abs/2506.21398)
* Complementary Prototype Mapping for Efficient Multimodal Anomaly Detection [\[CVPR 2026\]](https://cvpr.thecvf.com/virtual/2026/poster/37601)
* GPFlow: Gaussian Prototype Probability Flow for Unsupervised Multi-Modal Anomaly Detection [\[CVPR 2026\]](https://cvpr.thecvf.com/virtual/2026/poster/37603)
* Geometry-Aligned and Anomaly-Aware Reconstruction for 3D Anomaly Detection [\[CVPR 2026\]](https://cvpr.thecvf.com/virtual/2026/poster/36699)
* MMR-AD: A Large-Scale Multimodal Dataset for Benchmarking General Anomaly Detection with Multimodal Large Language Models [\[CVPR 2026\]](https://arxiv.org/abs/2604.10971)[\[code\]](https://xcyao00.github.io/MMR-AD)
* ADSeeker: A Knowledge-Grounded Reasoning Framework for Industry Anomaly Detection and Reasoning [\[CVPR 2026\]](https://arxiv.org/abs/2508.03088)
* Multi-Prototype Compactness and Boundary-Aware Synthesis for Unsupervised Anomaly Detection [\[CVPR 2026\]](https://cvpr.thecvf.com/virtual/2026/poster/37868)
* Dual-Prototype-Guided Multi-task Learning for Unsupervised Anomaly Detection and Classification [\[CVPR 2026\]](https://cvpr.thecvf.com/virtual/2026/poster/38573)
* Defect Cue-Preserved Structural Feature Refinement for Few-Shot Anomaly Detection [\[CVPR 2026\]](https://cvpr.thecvf.com/virtual/2026/poster/39808)
* Omni-AD: A Large-scale and Versatile Benchmark for Industrial Anomaly Detection [\[CVPR 2026\]](https://cvpr.thecvf.com/virtual/2026/poster/39148)
* From Attraction to Equilibrium: Physics-Inspired Semantic Gravitons for Zero-Shot Anomaly Detection [\[CVPR 2026\]](https://cvpr.thecvf.com/virtual/2026/poster/39782)
* Anomaly as Non-Conformity via Training-Free Graph Laplacian Energy Minimization [\[CVPR 2026\]](https://cvpr.thecvf.com/virtual/2026/poster/38295)
* A Semantically Disentangled Unified Model for Multi-category 3D Anomaly Detection [\[CVPR 2026\]](https://arxiv.org/abs/2603.25159)[\[code\]](https://visualsciencelab-khu.github.io/SeDiR_project)
* Hyperbolic Defect Feature Synthesis for Few-Shot Defect Classification [\[CVPR 2026\]](https://cvpr.thecvf.com/virtual/2026/poster/40073)
* UniSpector: Towards Universal Open-set Defect Recognition via Spectral-Contrastive Visual Prompting [\[CVPR 2026\]](https://cvpr.thecvf.com/virtual/2026/poster/38081)
* Towards Open-Vocabulary Industrial Defect Understanding with a Large-Scale Multimodal Dataset [\[CVPR 2026\]](https://arxiv.org/abs/2512.24160)[\[data\]](https://ninaneon.github.io/projectpage)
* Real-IAD MVN: A Multi-View Normal Vector Dataset and Benchmark for High-Fidelity Industrial Anomaly Detection [\[CVPR 2026\]](https://arxiv.org/abs/2605.07149)

## ICLR 2026

* Foundation Visual Encoders Are Secretly Few-Shot Anomaly Detectors [\[ICLR 2026\]](https://arxiv.org/abs/2510.01934)[\[code\]](https://github.com/ymxlzgy/FoundAD) ⭐ 81 | 🐛 4 | 🌐 Python | 📅 2026-03-16
* MRAD: Zero-Shot Anomaly Detection with Memory-Driven Retrieval [\[ICLR 2026\]](https://openreview.net/forum?id=TQkFiW3AEX)[\[code\]](https://github.com/CROVO1026/MRAD) ⭐ 52 | 🐛 4 | 🌐 Python | 📅 2026-03-13
* MRAD: Zero-Shot Anomaly Detection with Memory-Driven Retrieval [\[ICLR 2026\]](https://arxiv.org/abs/2602.00522)
* PIRN: Prototypical-based Intra-modal Reconstruction with Normality Communication for Multi-modal Anomaly Detection [\[ICLR 2026\]](https://openreview.net/forum?id=7L7kmHHfgf)
* Dual Distillation for Few-Shot Anomaly Detection [\[ICLR 2026\]](https://openreview.net/forum?id=tRO6G20Qba)
* Judo: A Juxtaposed Domain-oriented Multimodal Reasoner for Industrial Anomaly QA [\[ICLR 2026\]](https://openreview.net/forum?id=XW4mROtaVb)

## AAAI 2026

* AdaptCLIP: Adapting CLIP for Universal Visual Anomaly Detection [\[AAAI 2026\]](https://arxiv.org/abs/2505.09926)[\[code\]](https://github.com/gaobb/AdaptCLIP) ⭐ 182 | 🐛 3 | 🌐 Python | 📅 2026-02-05
* Anomagic: Crossmodal Prompt-driven Zero-shot Anomaly Generation [\[AAAI 2026\]](https://arxiv.org/abs/2511.10020)[\[code\]](https://github.com/yuxin-jiang/Anomagic) ⭐ 158 | 🐛 0 | 🌐 Python | 📅 2026-04-25
* IAD-R1: Reinforcing Consistent Reasoning in Industrial Anomaly Detection [\[AAAI 2026\]](https://arxiv.org/abs/2508.09178)[\[code\]](https://github.com/Yanhui-Lee/IAD-R1) ⭐ 96 | 🐛 17 | 🌐 Python | 📅 2025-12-09
* AnomalyMoE: Towards a Language-free Generalist Model for Unified Visual Anomaly Detection [\[AAAI 2026\]](https://arxiv.org/abs/2508.06203)[\[code\]](https://github.com/CASIA-LMC-Lab/AnomalyMoE) ⭐ 32 | 🐛 2 | 🌐 Python | 📅 2025-11-17
* Commonality in Few: Few-Shot Multimodal Anomaly Detection via Hypergraph-Enhanced Memory [\[AAAI 2026\]](https://arxiv.org/abs/2511.05966)[\[code\]](https://github.com/Sunny5250/CIF) ⭐ 26 | 🐛 2 | 🌐 Python | 📅 2025-11-13
* Commonality in Few: Few-Shot Multimodal Anomaly Detection via Hypergraph-Enhanced Memory [\[AAAI 2026\]](https://arxiv.org/abs/2511.05966)[\[code\]](https://github.com/Sunny5250/CIF) ⭐ 26 | 🐛 2 | 🌐 Python | 📅 2025-11-13
* AnoStyler: Text-Driven Localized Anomaly Generation via Lightweight Style Transfer [\[AAAI 2026\]](https://arxiv.org/abs/2511.06687)[\[code\]](https://github.com/yulimso/AnoStyler) ⭐ 22 | 🐛 7 | 🌐 Python | 📅 2025-11-14
* CASL: Curvature-Augmented Self-supervised Learning for 3D Anomaly Detection [\[AAAI 2026\]](https://arxiv.org/abs/2511.12909)[\[code\]](https://github.com/zyh16143998882/CASL) ⭐ 12 | 🐛 2 | 🌐 Python | 📅 2025-12-15
* CHIMERA:Controllable High-quality Image-Mask Extraction for Reliable Diffusion-Based Anomaly Synthesis [\[AAAI 2026\]](https://ojs.aaai.org/index.php/AAAI/article/view/37511)[\[code\]](https://github.com/cvlab-kaist/CHIMERA) ⭐ 11 | 🐛 2 | 📅 2025-11-16
* Quality-Aware Language-Conditioned Local Auto-Regressive Anomaly Synthesis and Detection [\[AAAI 2026\]](https://arxiv.org/abs/2508.03539)[\[code\]](https://github.com/neymarql/QARAD) ⭐ 8 | 🐛 0 | 🌐 Python | 📅 2025-11-12
* FDP: A Frequency-Decomposition Preprocessing Pipeline for Unsupervised Anomaly Detection in Brain MRI [\[AAAI 2026\]](https://arxiv.org/abs/2511.12899)[\[code\]](https://github.com/ls1rius/MRI_FDP) ⭐ 4 | 🐛 0 | 🌐 Python | 📅 2026-05-23
* MaskAD: Parallel Masked Autoencoder for Multi-class Unsupervised Anomaly Detection [\[AAAI 2026\]](https://ojs.aaai.org/index.php/AAAI/article/download/38573/42535)[\[code\]](https://github.com/liugang-xd/MaskAD) ⭐ 3 | 🐛 0 | 🌐 Python | 📅 2026-05-31
* Towards High-Resolution 3D Anomaly Detection: A Scalable Dataset and Real-Time Framework for Subtle Industrial Defects [\[AAAI 2026 oral\]](https://arxiv.org/abs/2507.07435)[\[code\]](https://hustcyq.github.io/MiniShift-Simple3D/)
* PromptMoE: Generalizable Zero-Shot Anomaly Detection via Visually-Guided Prompt Mixtures [\[AAAI 2026\]](https://arxiv.org/abs/2511.18116)[\[code\]](https://github.com/yourusername/PromptMoE)
* Unsupervised Multi-View Visual Anomaly Detection via Progressive Homography-Guided Alignment [\[AAAI 2026\]](https://arxiv.org/abs/2511.18766)
* AnomalyPainter: Vision-Language-Diffusion Synergy for Zero-Shot Realistic and Diverse Industrial Anomaly Synthesis [\[AAAI 2026\]](https://arxiv.org/abs/2503.07253)
* Exploring High-order-aware Prompt Learning for Zero-shot Anomaly Detection [\[AAAI 2026\]](https://ojs.aaai.org/index.php/AAAI/article/view/38029)
* RcAE: Recursive Reconstruction Framework for Unsupervised Industrial Anomaly Detection [\[AAAI 2026\]](https://arxiv.org/abs/2512.11284)
* CADiff: Context-Aware Diffusion for Controllable Anomaly Generation in Anomaly Detection [\[AAAI 2026\]](https://ojs.aaai.org/index.php/AAAI/article/view/37917)
* MAU-GPT: Enhancing Multi-type Industrial Anomaly Understanding via Anomaly-aware and Generalist Experts Adaptation [\[AAAI 2026\]](https://arxiv.org/abs/2602.07011)
* SCoNE: Spherical Consistent Neighborhoods Ensemble for Effective and Efficient Multi-View Anomaly Detection [\[AAAI 2026\]](https://arxiv.org/abs/2512.05540)
* RPE-PAD: Relative Pose Estimation for Pose-agnostic Anomaly Detection [\[AAAI 2026\]](https://ojs.aaai.org/index.php/AAAI/article/view/38304)
* AD-FM: Multimodal LLMs for Anomaly Detection via Multi-Stage Reasoning and Fine-Grained Reward Optimization [\[AAAI 2026\]](https://arxiv.org/abs/2508.04175)

## NeurIPS 2025

* Normal-Abnormal Guided Generalist Anomaly Detection [\[NeurIPS 2025\]](https://arxiv.org/abs/2510.00495)[\[code\]](https://github.com/JasonKyng/NAGL) ⭐ 86 | 🐛 0 | 🌐 Python | 📅 2025-11-21
* ADPretrain: Advancing Industrial Anomaly Detection via Anomaly Representation Pretraining [\[NeurIPS 2025\]](https://arxiv.org/abs/2511.05245)[\[code\]](https://github.com/xcyao00/ADPretrain) ⭐ 76 | 🐛 6 | 🌐 Python | 📅 2026-06-02
* FAST: Foreground-aware Diffusion with Accelerated Sampling Trajectory for Segmentation-oriented Anomaly Synthesis [\[NeurIPS 2025\]](https://arxiv.org/abs/2509.20295)[\[code\]](https://anonymous.4open.science/r/NeurIPS-938/README1.md)
* Registration is a Powerful Rotation-Invariance Learner for 3D Anomaly Detection [\[NeurIPS 2025\]](https://arxiv.org/abs/2510.16865)

## KDD 2025

* Self-Tuning Self-Supervised Image Anomaly Detection [\[KDD 2025\]](https://arxiv.org/abs/2306.12033) [\[code\]](https://github.com/jaeminyoo/ST-SSAD) ⭐ 0 | 🐛 0 | 🌐 Python | 📅 2025-07-21
* Logical Anomaly Detection with Text-based Logic via Component-Aware Contrastive Language-Image Training [\[KDD 25\]](https://dl.acm.org/doi/abs/10.1145/3711896.3737032)

<!-- ## ACM MM 2025
+ AF-CLIP: Zero-Shot Anomaly Detection via Anomaly-Focused CLIP Adaptation [[ACM MM 2025]](https://arxiv.org/abs/2507.19949)[[code]](https://github.com/Faustinaqq/AF-CLIP)
+ AnomalyControl: Highly-Aligned Anomalous Image Generation with Controlled Diffusion Model [[ACM MM 2025]](https://dl.acm.org/doi/abs/10.1145/3746027.3755274)
+ Taming Anomalies with Down-Up Sampling Networks: Group Center Preserving Reconstruction for 3D Anomaly Detection [[ACM MM 2025]](https://arxiv.org/abs/2507.03903v1)
+ Robust Modality-Incomplete Anomaly Detection: A Modality-Instructive Framework with Benchmark [[ACM MM 2025]](https://dl.acm.org/doi/abs/10.1145/3746027.3754766)
+ Exploring Multimodal Prompts For Unsupervised Continuous Anomaly Detection [[ACM MM 2025]](https://dl.acm.org/doi/abs/10.1145/3746027.3755219)
+ Learning Invariant Discriminative Patterns for Unified Anomaly Detection [[ACM MM 2025]](https://dl.acm.org/doi/abs/10.1145/3746027.3755179)
+ Uniad: Integrating geometric and semantic cues for unified anomaly detection [[ACM MM 2025]](https://dl.acm.org/doi/abs/10.1145/3746027.3755422) -->

## ICCV 2025

* SeaS: Few-shot Industrial Anomaly Image Generation with Separation and Sharing Fine-tuning [\[ICCV 2025\]](https://arxiv.org/pdf/2410.14987)[\[code\]](https://github.com/HUST-SLOW/SeaS) ⭐ 160 | 🐛 21 | 🌐 Python | 📅 2025-08-04
* MultiADS: Defect-aware Supervision for Multi-type Anomaly Detection and Segmentation in Zero-Shot Learning [\[ICCV 2025\]](https://arxiv.org/abs/2504.06740)[\[code\]](https://github.com/boschresearch/MultiADS) ⭐ 68 | 🐛 3 | 🌐 Python | 📅 2026-08-28
* DictAS: A Framework for Class-Generalizable Few-Shot Anomaly Segmentation via Dictionary Lookup [\[ICCV 2025\]](https://www.arxiv.org/abs/2508.13560)[\[code\]](https://github.com/xiaozhen228/DictAS) ⭐ 58 | 🐛 1 | 🌐 Python | 📅 2025-12-13
* SALAD -- Semantics-Aware Logical Anomaly Detection [\[ICCV 2025\]](https://arxiv.org/abs/2509.02101)[\[code\]](https://github.com/MaticFuc/SALAD) ⭐ 49 | 🐛 10 | 🌐 Python | 📅 2025-10-03
* Fine-grained Abnormality Prompt Learning for Zero-shot Anomaly Detection [\[ICCV 2025\]](https://arxiv.org/abs/2410.10289)[\[code\]](https://github.com/mala-lab/FAPrompt) ⭐ 47 | 🐛 3 | 🌐 Python | 📅 2026-03-07
* Triad: Empowering LMM-based Anomaly Detection with Vision Expert-guided Visual Tokenizer and Manufacturing Process [\[ICCV 2025\]](https://arxiv.org/abs/2503.13184)[\[code\]](https://github.com/tzjtatata/Triad) ⭐ 42 | 🐛 0 | 🌐 Python | 📅 2025-08-15
* Training-Free Industrial Defect Generation with Diffusion Models [\[ICCV 2025\]](https://openaccess.thecvf.com/content/ICCV2025/papers/Xu_Training-Free_Industrial_Defect_Generation_with_Diffusion_Models_ICCV_2025_paper.pdf)[\[code\]](https://github.com/rubymiaomiao/TF-IDG) ⭐ 42 | 🐛 1 | 🌐 Python | 📅 2025-10-01
* RareCLIP: Rarity-aware Online Zero-shot Industrial Anomaly Detection [\[ICCV 2025\]](https://openaccess.thecvf.com/content/ICCV2025/papers/He_RareCLIP_Rarity-aware_Online_Zero-shot_Industrial_Anomaly_Detection_ICCV_2025_paper.pdf)[\[code\]](https://github.com/hjf02/RareCLIP) ⭐ 25 | 🐛 0 | 🌐 Python | 📅 2025-10-20
* Bridging 3D Anomaly Localization and Repair via High-Quality Continuous Geometric Representation [\[ICCV 2025\]](https://arxiv.org/abs/2505.24431)[\[code\]](https://github.com/ZZZBBBZZZ/PASDF) ⭐ 18 | 🐛 2 | 🌐 Python | 📅 2025-12-17
* G2SF: Geometry-Guided Score Fusion for Multimodal Industrial Anomaly Detection[\[ICCV 2025\]](https://openaccess.thecvf.com/content/ICCV2025/papers/Tao_G2SF_Geometry-Guided_Score_Fusion_for_Multimodal_Industrial_Anomaly_Detection_ICCV_2025_paper.pdf)[\[code\]](https://github.com/ctaoaa/G2SF) ⭐ 17 | 🐛 2 | 🌐 Python | 📅 2025-12-14
* Anomaly Detection of Integrated Circuits Package Substrates Using the Large Vision Model SAIC: Dataset Construction, Methodology, and Application [\[ICCV 2025\]](https://openaccess.thecvf.com/content/ICCV2025/papers/Yu_Anomaly_Detection_of_Integrated_Circuits_Package_Substrates_Using_the_Large_ICCV_2025_paper.pdf)[\[data\]](https://github.com/Bingyang0410/CPS2D-AD) ⭐ 16 | 🐛 4 | 📅 2025-07-09
* ReMP-AD: Retrieval-enhanced Multi-modal Prompt Fusion for Few-Shot Industrial Visual Anomaly Detection [\[ICCV 2025\]](https://openaccess.thecvf.com/content/ICCV2025/papers/Ma_ReMP-AD_Retrieval-enhanced_Multi-modal_Prompt_Fusion_for_Few-Shot_Industrial_Visual_Anomaly_ICCV_2025_paper.pdf)[\[code\]](https://github.com/cshcma/ReMP-AD) ⭐ 16 | 🐛 0 | 🌐 Python | 📅 2025-08-16
* Salvaging the Overlooked: Leveraging Class-Aware Contrastive Learning for Multi-Class Anomaly Detection [\[ICCV 2025\]](https://openaccess.thecvf.com/content/ICCV2025/papers/Fan_Salvaging_the_Overlooked_Leveraging_Class-Aware_Contrastive_Learning_for_Multi-Class_Anomaly_ICCV_2025_paper.pdf)[\[code\]](https://github.com/LGC-AD/AD-LGC) ⭐ 11 | 🐛 2 | 🌐 Python | 📅 2025-11-26
* Towards Real Unsupervised Anomaly Detection Via Confident Meta-Learning [\[ICCV 2025\]](https://arxiv.org/abs/2508.02293)
* Kaputt: A Large-Scale Dataset for Visual Defect Detection [\[ICCV 2025\]](https://arxiv.org/abs/2510.05903)[\[data\]](https://www.kaputt-dataset.com/)
* FE-CLIP: Frequency Enhanced CLIP Model for Zero-Shot Anomaly Detection and Segmentation [\[ICCV 2025\]](https://openaccess.thecvf.com/content/ICCV2025/papers/Gong_FE-CLIP_Frequency_Enhanced_CLIP_Model_for_Zero-Shot_Anomaly_Detection_and_ICCV_2025_paper.pdf)
* DecAD: Decoupling Anomalies in Latent Space for Multi-Class Unsupervised Anomaly Detection [\[ICCV 2025\]](https://openaccess.thecvf.com/content/ICCV2025/html/Wang_DecAD_Decoupling_Anomalies_in_Latent_Space_for_Multi-Class_Unsupervised_Anomaly_ICCV_2025_paper.html)
* Debiasing Trace Guidance: Top-down Trace Distillation and Bottom-up Velocity Alignment for Unsupervised Anomaly Detection [\[ICCV 2025\]](https://openaccess.thecvf.com/content/ICCV2025/papers/Wang_Debiasing_Trace_Guidance_Top-down_Trace_Distillation_and_Bottom-up_Velocity_Alignment_ICCV_2025_paper.pdf)
* FIND: Few-Shot Anomaly Inspection with Normal-Only Multi-Modal Data [\[ICCV 2025\]](https://openaccess.thecvf.com/content/ICCV2025/papers/Li_FIND_Few-Shot_Anomaly_Inspection_with_Normal-Only_Multi-Modal_Data_ICCV_2025_paper.pdf)
* SiM3D: Single-instance Multiview Multimodal and Multisetup 3D Anomaly Detection Benchmark [\[ICCV 2025\]](https://arxiv.org/abs/2506.21549)[\[data\]](https://alex-costanzino.github.io/SiM3D/)
* Toward Long-Tailed Online Anomaly Detection through Class-Agnostic Concepts [\[ICCV 2025\]](https://arxiv.org/abs/2507.16946)[\[data\]](https://zenodo.org/records/16283853)
* Wave-MambaAD: Wavelet-driven State Space Model for Multi-class Unsupervised Anomaly Detection [\[ICCV 2025\]](https://openaccess.thecvf.com/content/ICCV2025/papers/Zhang_Wave-MambaAD_Wavelet-driven_State_Space_Model_for_Multi-class_Unsupervised_Anomaly_Detection_ICCV_2025_paper.pdf)

## ICML 2025

* CostFilter-AD: Enhancing Anomaly Detection through Matching Cost Filtering [\[ICML2025\]](https://openreview.net/pdf?id=6p2wsBeYSs)[\[code\]](https://github.com/ZHE-SAPI/CostFilter-AD) ⭐ 93 | 🐛 1 | 🌐 Python | 📅 2026-07-03
* OmiAD: One-Step Adaptive Masked Diffusion Model for Multi-class Anomaly Detection via Adversarial Distillation [\[ICML2025\]](https://icml.cc/virtual/2025/poster/46291)
* Demeaned Sparse: Efficient Anomaly Detection by Residual Estimate [\[ICML2025\]](https://icml.cc/virtual/2025/poster/45914)

<!-- ## IJCAI 2025
+ MC3D-AD: A Unified Geometry-aware Reconstruction Model for Multi-category 3D Anomaly Detection [[IJCAI 2025]](https://arxiv.org/abs/2505.01969)
+ ReplayCAD: Generative Diffusion Replay for Continual Anomaly Detection [[IJCAI 2025]](https://arxiv.org/abs/2505.06603)[[code]](https://github.com/HULEI7/ReplayCAD) -->

## CVPR 2025

* Dinomaly: The Less is More Philosophy in Multi-Class Unsupervised Anomaly Detection [\[CVPR 2025\]](https://arxiv.org/abs/2405.14325)[\[code\]](https://github.com/guojiajeremy/Dinomaly) ⭐ 516 | 🐛 30 | 🌐 Python | 📅 2026-05-21
* Exploring Intrinsic Normal Prototypes within a Single Image for Universal Anomaly Detection [\[CVPR 2025\]](https://arxiv.org/abs/2503.02424)[\[code\]](https://github.com/luow23/INP-Former) ⭐ 317 | 🐛 45 | 🌐 Python | 📅 2026-05-26
* Anomaly Anything: Promptable Unseen Visual Anomaly Generation [\[CVPR 2025\]](https://arxiv.org/abs/2406.01078)[\[code\]](https://github.com/EPFL-IMOS/AnomalyAny) ⭐ 301 | 🐛 14 | 🌐 Jupyter Notebook | 📅 2025-06-30
* AA-CLIP: Enhancing Zero-shot Anomaly Detection via Anomaly-Aware CLIP [\[CVPR 2025\]](https://arxiv.org/pdf/2503.06661)[\[code\]](https://github.com/Mwxinnn/AA-CLIP) ⭐ 269 | 🐛 29 | 🌐 Python | 📅 2025-05-26
* AnomalyNCD: Towards Novel Anomaly Class Discovery in Industrial Scenarios [\[CVPR 2025\]](https://arxiv.org/abs/2410.14379)[\[code\]](https://github.com/HUST-SLOW/AnomalyNCD) ⭐ 165 | 🐛 11 | 🌐 Python | 📅 2025-09-01
* DualAnoDiff: Dual-Interrelated Diffusion Model for Few-Shot Anomaly Image Generation [\[CVPR 2025\]](https://arxiv.org/abs/2408.13509)[\[code\]](https://github.com/yinyjin/DualAnoDiff) ⭐ 161 | 🐛 11 | 🌐 Python | 📅 2025-06-03
* UniNet: A Contrastive Learning-guided Unified Framework with Feature Selection for Anomaly Detection [\[CVPR 2025\]](https://pangdatangtt.github.io/)[\[code\]](https://github.com/pangdatangtt/UniNet) ⭐ 137 | 🐛 14 | 🌐 Python | 📅 2025-08-12
* Wavelet and Prototype Augmented Query-based Transformer for Pixel-level Surface Defect Detection[\[CVPR 2025\]](https://openaccess.thecvf.com/content/CVPR2025/html/Yan_Wavelet_and_Prototype_Augmented_Query-based_Transformer_for_Pixel-level_Surface_Defect_CVPR_2025_paper.html)[\[code\]](https://github.com/yfhdm/WPFormer) ⭐ 101 | 🐛 5 | 🌐 Python | 📅 2026-04-20
* Towards Training-free Anomaly Detection with Vision and Language Foundation Models [\[CVPR 2025\]](https://arxiv.org/abs/2503.18325)[\[code\]](https://github.com/zhang0jhon/LogSAD) ⭐ 100 | 🐛 12 | 🌐 Python | 📅 2025-05-19
* Multi-Sensor Object Anomaly Detection: Unifying Appearance, Geometry, and Internal Properties [\[CVPR 2025\]](https://zzzbbbzzz.github.io/MulSen_AD/index.html)[\[code\]](https://github.com/ZZZBBBZZZ/MulSen-AD) ⭐ 94 | 🐛 4 | 🌐 Python | 📅 2025-03-20
* Bayesian Prompt Flow Learning for Zero-Shot Anomaly Detection [\[CVPR 2025\]](https://arxiv.org/abs/2503.10080)[\[code coming soon\]](https://github.com/xiaozhen228/Bayes-PFL) ⭐ 71 | 🐛 7 | 🌐 Python | 📅 2025-06-23
* Correcting Deviations from Normality: A Reformulated Diffusion Model for Multi-Class Unsupervised Anomaly Detection [\[CVPR 2025\]](https://arxiv.org/abs/2503.19357)[\[code\]](https://github.com/farzad-bz/DeCo-Diff) ⭐ 67 | 🐛 0 | 🌐 Python | 📅 2026-03-04
* Correcting Deviations from Normality: A Reformulated Diffusion Model for Multi-Class Unsupervised Anomaly Detection[\[CVPR 2025\]](https://openaccess.thecvf.com/content/CVPR2025/html/Beizaee_Correcting_Deviations_from_Normality_A_Reformulated_Diffusion_Model_for_Multi-Class_CVPR_2025_paper.html)[\[code\]](https://github.com/farzad-bz/DeCo-Diff) ⭐ 67 | 🐛 0 | 🌐 Python | 📅 2026-03-04
* One-for-More: Continual Diffusion Model for Anomaly Detection [\[CVPR 2025\]](https://arxiv.org/abs/2502.19848)[\[code\]](https://github.com/FuNz-0/One-for-More) ⭐ 64 | 🐛 11 | 🌐 Python | 📅 2025-05-07
* Odd-One-Out: Anomaly Detection by Comparing with Neighbors [\[CVPR 2025\]](https://arxiv.org/abs/2406.20099)[\[code\]](https://github.com/VICO-UoE/OddOneOutAD) ⭐ 57 | 🐛 0 | 🌐 Python | 📅 2026-01-19
* TailedCore: Few-Shot Sampling for Unsupervised Long-Tail Noisy Anomaly Detection [\[CVPR 2025\]](https://jungyg.github.io/TailedCore_site/)[\[code\]](https://github.com/jungyg/TailedCore) ⭐ 35 | 🐛 1 | 🌐 Python | 📅 2025-06-12
* Distribution Prototype Diffusion Learning for Open-set Supervised Anomaly Detection [\[CVPR 2025\]](https://arxiv.org/abs/2502.20981)[\[code\]](https://github.com/fuyunwang/DPDL) ⭐ 29 | 🐛 0 | 🌐 Python | 📅 2025-02-28
* Towards Visual Discrimination and Reasoning of Real-World Physical Dynamics: Physics-Grounded Anomaly Detection [\[CVPR 2025\]](https://arxiv.org/abs/2503.03562)[\[code\]](https://github.com/Chopper-233/Physics-AD) ⭐ 23 | 🐛 0 | 🌐 Python | 📅 2026-02-24
* RoBiS: Robust Binary Segmentation for High-Resolution Industrial Images [\[CVPR 2025 VAND 3.0 Workshop\]](https://arxiv.org/abs/2505.21152)[\[code\]](https://github.com/xrli-U/RoBiS) ⭐ 23 | 🐛 0 | 🌐 Python | 📅 2025-07-30
* PatchGuard: Adversarially Robust Anomaly Detection and Localization through Vision Transformers and Pseudo Anomalies[\[CVPR 2025\]](https://openaccess.thecvf.com/content/CVPR2025/html/Nafez_PatchGuard_Adversarially_Robust_Anomaly_Detection_and_Localization_through_Vision_Transformers_CVPR_2025_paper.html)[\[code\]](https://github.com/rohban-lab/PatchGuard) ⭐ 18 | 🐛 4 | 🌐 Jupyter Notebook | 📅 2025-11-25
* Beyond Single-Modal Boundary: Cross-Modal Anomaly Detection through Visual Prototype and Harmonization[\[CVPR 2025\]](https://openaccess.thecvf.com/content/CVPR2025/html/Mao_Beyond_Single-Modal_Boundary_Cross-Modal_Anomaly_Detection_through_Visual_Prototype_and_CVPR_2025_paper.html)[\[code\]](https://github.com/Kerio99/CMAD) ⭐ 7 | 🐛 0 | 🌐 Python | 📅 2026-06-04
* When Textures Deceive: Weakly Supervised Industrial Anomaly Detection with Adapted-Loss (AL-CycleGAN) [\[CVPR 2025 VAND Workshop\]](https://openaccess.thecvf.com/content/CVPR2025W/VAND/papers/Nakkina_When_Textures_Deceive_Weakly_Supervised_Industrial_Anomaly_Detection_with_Adapted-Loss_CVPRW_2025_paper.pdf)[\[code\]](https://github.com/ganatma/AL-CycleGAN) ⭐ 2 | 🐛 0 | 🌐 Python | 📅 2026-04-16[\[data / MCBT\]](https://github.com/ganatma/AL-CycleGAN) ⭐ 2 | 🐛 0 | 🌐 Python | 📅 2026-04-16
* A Unified Latent Schrödinger Bridge Diffusion Model for Unsupervised Anomaly Detection and Localization[\[CVPR 2025\]](https://openaccess.thecvf.com/content/CVPR2025/html/Akshay_A_Unified_Latent_Schrodinger_Bridge_Diffusion_Model_for_Unsupervised_Anomaly_CVPR_2025_paper.html)[\[code\]](https://github.com/ShilhoraAkshayPatel/LASB) ⭐ 1 | 🐛 2 | 📅 2025-06-02
* Real-IAD D<sup>3</sup>: A Real-World 2D/Pseudo-3D/3D Dataset for Industrial Anomaly Detection [\[CVPR 2025\]](https://arxiv.org/abs/2504.14221)
* UniVAD: A Training-free Unified Model for Few-shot Visual Anomaly Detection [\[CVPR 2025\]](https://arxiv.org/abs/2412.03342)[\[code\]](https://uni-vad.github.io/#)
* Towards Zero-Shot Anomaly Detection and Reasoning with Multimodal Large Language Models [\[CVPR 2025\]](https://arxiv.org/abs/2502.07601)[\[code\]](https://xujiacong.github.io/Anomaly-OV/)
* MANTA: A Large-Scale Multi-View and Visual-Text Anomaly Detection Dataset for Tiny Objects [\[CVPR 2025\]](https://arxiv.org/abs/2412.04867)[\[data\]](https://grainnet.github.io/MANTA)
* PO3AD: Predicting Point Offsets toward Better 3D Point Cloud Anomaly Detection [\[CVPR 2025\]](https://arxiv.org/abs/2412.12617)
* DefectFill: Realistic Defect Generation with Inpainting Diffusion Model for Visual Inspection [\[CVPR 2025\]](https://arxiv.org/abs/2503.13985)
* DFM: Differentiable Feature Matching for Anomaly Detection[\[CVPR 2025\]](https://openaccess.thecvf.com/content/CVPR2025/html/Wu_DFM_Differentiable_Feature_Matching_for_Anomaly_Detection_CVPR_2025_paper.html)
* VAND 3.0: Visual Anomaly and Novelty Detection - 3rd Edition [\[CVPR 2025W\]](https://sites.google.com/view/vand30cvpr2025)
* Feature Attenuation of Defective Representation Can Resolve Incomplete Masking on Anomaly Detection [\[CVPR 2025 VAND 3.0 Workshop\]](https://arxiv.org/abs/2407.04597)
* AnomalyHybrid: A Domain-agnostic Generative Framework for General Anomaly Detection [\[CVPR 2025 SyntaGen Workshop\]](https://openaccess.thecvf.com/content/CVPR2025W/SyntaGen/papers/Zhao_AnomalyHybrid_A_Domain-agnostic_Generative_Framework_for_General_Anomaly_Detection_CVPRW_2025_paper.pdf)

<!-- ## ICLR 2025
+ MMAD: The Comprehensive Benchmark for Multimodal Large Language Models in Industrial Anomaly Detection [[ICLR 2025]](https://openreview.net/forum?id=JDiER86r8v)[[Code]](https://github.com/jam-cc/MMAD)  [[Data]](https://huggingface.co/datasets/jiang-cc/MMAD)
+ One-for-All Few-Shot Anomaly Detection via Instance-Induced Prompt Learning [[ICLR 2025]](https://openreview.net/forum?id=Zzs3JwknAY) 
+ Language-Assisted Feature Transformation for Anomaly Detection [[ICLR 2025]](https://openreview.net/forum?id=2p03KljxE9)
+ Adversarially Robust Anomaly Detection through Spurious Negative Pair Mitigation [[ICLR 2025]](https://openreview.net/forum?id=t8fu5m8R5m)


## AAAI 2025
+ MVREC: A General Few-shot Defect Classification Model Using Multi-View Region-Context [[AAAI 2025]](https://arxiv.org/abs/2412.16897)
+ Revisiting Multimodal Fusion for 3D Anomaly Detection from an Architectural Perspective [[AAAI 2025]](https://arxiv.org/abs/2412.17297)
+ KAG-prompt: Kernel-Aware Graph Prompt Learning for Few-Shot Anomaly Detection [[AAAI 2025]](https://arxiv.org/abs/2412.17619)[[code]](https://github.com/CVL-hub/KAG-prompt)
+ FiCo: Filter or Compensate: Towards Invariant Representation from Distribution Shift for Anomaly Detection [[AAAI 2025]](https://arxiv.org/abs/2412.10115)[[code]](https://github.com/znchen666/FiCo)
+ CKAAD: Boosting Fine-Grained Visual Anomaly Detection with Coarse-Knowledge-Aware Adversarial Learning [[AAAI 2025]](https://arxiv.org/abs/2412.12850)[[code]](https://github.com/Faustinaqq/CKAAD)
+ CNC: Cross-modal Normality Constraint for Unsupervised Multi-class Anomaly Detection [[AAAI 2025]](https://arxiv.org/abs/2501.00346)[[code]](https://github.com/cvddl/CNC)
+ LogicAD: Explainable Anomaly Detection via VLM-based Text Feature Extraction [[AAAI 2025]](https://arxiv.org/abs/2501.01767)[[code]](https://github.com/jasonjin34/logicAD)
+ Look Inside for More: Internal Spatial Modality Perception for 3D Anomaly Detection [[AAAI 2025]](https://arxiv.org/abs/2412.13461)[[code]](https://github.com/M-3LAB/Look-Inside-for-More)
+ Unlocking the Potential of Reverse Distillation for Anomaly Detection [[AAAI 2025]](https://arxiv.org/abs/2412.07579)[[code]](https://github.com/hito2448/URD)
+ Promptable Anomaly Segmentation with SAM Through Self-Perception Tuning [[AAAI 2025]](https://arxiv.org/abs/2411.17217)[[code]](https://github.com/THU-MIG/SAM-SPT)
+ 3CAD: A Large-Scale Real-World 3C Product Dataset for Unsupervised Anomaly [[AAAI 2025]](https://arxiv.org/abs/2502.05761)[[code]](https://github.com/EnquanYang2022/3CAD) -->

<!--- 

## NeurIPS 2024
+ MambaAD: Exploring State Space Models for Multi-class Unsupervised Anomaly Detection [[NeurIPS 2024]](https://arxiv.org/abs/2404.06564)[[code]](https://lewandofskee.github.io/projects/MambaAD/)
+ PointAD: Comprehending 3D Anomalies from Points and Pixels for Zero-shot 3D Anomaly Detection [[NeurIPS 2024]](https://arxiv.org/abs/2410.00320)[[code]](https://github.com/zqhang/PointAD)
+ CableInspect-AD: An Expert-Annotated Anomaly Detection Dataset [[NeurIPS 2024]](https://arxiv.org/abs/2409.20353)[[data]](https://mila-iqia.github.io/cableinspect-ad/)
+ ResAD: A Simple Framework for Class Generalizable Anomaly Detection [[NeurIPS 2024]](https://arxiv.org/abs/2410.20047)[[code]](https://github.com/xcyao00/ResAD)
+ One-to-Normal: Anomaly Personalization for Few-shot Anomaly Detection [[NeurIPS 2024]](https://openreview.net/pdf?id=tIzW3l2uaN)
+ MetaUAS: Universal Anomaly Segmentation with One-Prompt Meta-Learning [[NeurIPS 2024]](https://arxiv.org/abs/2505.09265)[[code]](https://github.com/gaobb/MetaUAS)

## ECCV 2024
+ R3D-AD: Reconstruction via Diffusion for 3D Anomaly Detection [[ECCV 2024]](https://arxiv.org/abs/2407.10862)[[homepage]](https://zhouzheyuan.github.io/r3d-ad)
+ An Incremental Unified Framework for Small Defect Inspection [[ECCV 2024]](https://arxiv.org/abs/2312.08917v2)[[code]](https://github.com/jqtangust/IUF)
+ Learning Unified Reference Representation for Unsupervised Multi-class Anomaly Detection [[ECCV 2024]](https://arxiv.org/abs/2403.11561)[[code]](https://github.com/hlr7999/RLR)
+ Self-supervised Feature Adaptation for 3D Industrial Anomaly Detection [[ECCV 2024]](https://arxiv.org/abs/2401.03145)
+ Learning to Detect Multi-class Anomalies with Just One Normal Image Prompt [[ECCV 2024]](https://arxiv.org/abs/2505.09264)[[code]](https://github.com/gaobb/OneNIP)
+ Few-Shot Anomaly-Driven Generation for Anomaly Classification and Segmentation [[ECCV 2024]](https://csgaobb.github.io/Pub_files/ECCV2024_AnoGen_CR_0730_Mobile.pdf)[[code]](https://github.com/gaobb/AnoGen)
+ AdaCLIP: Adapting CLIP with Hybrid Learnable Prompts for Zero-Shot Anomaly Detection [[ECCV 2024]](https://arxiv.org/abs/2407.15795)[[code]](https://github.com/caoyunkang/AdaCLIP)
+ GLAD: Towards Better Reconstruction with Global and Local Adaptive Diffusion Models for Unsupervised Anomaly Detection [[ECCV 2024]](https://arxiv.org/abs/2406.07487)[[code]](https://github.com/hyao1/GLAD)
+ GeneralAD: Anomaly Detection Across Domains by Attending to Distorted Features [[ECCV 2024]](https://arxiv.org/abs/2407.12427)[[code]](https://github.com/LucStrater/GeneralAD)
+ VCP-CLIP: A visual context prompting model for zero-shot anomaly segmentation [[ECCV 2024]](https://arxiv.org/abs/2407.12276)[[code]](https://github.com/xiaozhen228/VCP-CLIP)
+ A Unified Anomaly Synthesis Strategy with Gradient Ascent for Industrial Anomaly Detection and Localization [[ECCV 2024]](https://arxiv.org/abs/2407.09359)[[code]](https://github.com/cqylunlun/GLASS)
+ Hierarchical Gaussian Mixture Normalizing Flow Modeling for Unified Anomaly Detection [[ECCV 2024]](https://arxiv.org/abs/2403.13349)[[code]](https://github.com/xcyao00/HGAD)
+ TransFusion -- A Transparency-Based Diffusion Model for Anomaly Detection [[ECCV 2024]](https://arxiv.org/abs/2311.09999)[[code]](https://github.com/MaticFuc/ECCV_TransFusion)
+ Continuous Memory Representation for Anomaly Detection [[ECCV 2024]](https://arxiv.org/abs/2402.18293)[[homepage]](https://tae-mo.github.io/crad/)[[code]](https://github.com/tae-mo/CRAD)
+ Defect Spectrum: A Granular Look of Large-Scale Defect Datasets with Rich Semantics [[ECCV 2024]](https://openreview.net/forum?id=RLhS1TrjK3)[[data]](https://github.com/EnVision-Research/Defect_Spectrum)
+ AD3: Introducing a score for Anomaly Detection Dataset Difficulty assessment using VIADUCT dataset [[ECCV 2024]](https://eccv.ecva.net/virtual/2024/poster/2287)[[data]](https://fordatis.fraunhofer.de/handle/fordatis/363.2)
+ Learning Diffusion Models for Multi-View Anomaly Detection [[ECCV 2024]](https://eccv2024.ecva.net/virtual/2024/poster/1911)
+ MoEAD: A Parameter-efficient Model for Multi-class Anomaly Detection [[ECCV 2024]](https://eccv2024.ecva.net/virtual/2024/poster/2653)[[code]](https://github.com/TheStarOfMSY/MoEAD)
+ Unsupervised, Online and On-The-Fly Anomaly Detection For Non-Stationary Image Distributions [[ECCV 2024]](https://eccv2024.ecva.net/virtual/2024/poster/2289)[[code]](https://github.com/DeclanMcIntosh/Online_InReaCh)
+ Tackling Structural Hallucination in Image Translation with Local Diffusion [[ECCV 2024 oral]](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/10498.pdf)[[code]](https://github.com/edshkim98/LocalDiffusion-Hallucination)


## ACM MM 2024
+ FiLo: Zero-Shot Anomaly Detection by Fine-Grained Description and High-Quality Localization [[ACM MM 2024]](https://arxiv.org/abs/2404.13671)[[code]](https://github.com/CASIA-IVA-Lab/FiLo)
+ Dual-Modeling Decouple Distillation for Unsupervised Anomaly Detection [[ACM MM 2024]](https://arxiv.org/abs/2408.03888)
+ FOCT: Few-shot Industrial Anomaly Detection with Foreground-aware Online Conditional Transport [[ACM MM 2024]](https://dl.acm.org/doi/10.1145/3664647.3680771)
+ Towards High-resolution 3D Anomaly Detection via Group-Level Feature Contrastive Learning [[ACM MM 2024]](https://arxiv.org/abs/2408.04604)[[code]](https://github.com/M-3LAB/Group3AD)

## CVPR 2024
+ Text-Guided Variational Image Generation for Industrial Anomaly Detection and Segmentation [[CVPR 2024]](https://arxiv.org/abs/2403.06247)[[code]](https://github.com/MingyuLee82/TGI_AD_v1)
+ RealNet: A Feature Selection Network with Realistic Synthetic Anomaly for Anomaly Detection [[CVPR 2024]](https://arxiv.org/abs/2403.05897)[[code]](https://github.com/cnulab/RealNet)
+ Toward Generalist Anomaly Detection via In-context Residual Learning with Few-shot Sample Prompts [[CVPR 2024]](https://arxiv.org/abs/2403.06495)[[code]](https://github.com/mala-lab/InCTRL)
+ Multimodal Industrial Anomaly Detection by Crossmodal Feature Mapping [[CVPR 2024]](https://arxiv.org/abs/2312.04521)
+ Towards Scalable 3D Anomaly Detection and Localization: A Benchmark via 3D Anomaly Synthesis and A Self-Supervised Learning Network [[CVPR 2024]](https://arxiv.org/abs/2311.14897)[[code]](https://github.com/Chopper-233/Anomaly-ShapeNet)
+ Real-IAD: A Real-World Multi-view Dataset for Benchmarking Versatile Industrial Anomaly Detection [[CVPR 2024]](https://arxiv.org/abs/2403.12580)[[code]](https://github.com/TencentYoutuResearch/AnomalyDetection_Real-IAD)[[data]](https://realiad4ad.github.io/Real-IAD/)
+ Long-Tailed Anomaly Detection with Learnable Class Names [[CVPR 2024]](https://arxiv.org/abs/2403.20236)[[data split]](https://zenodo.org/records/10854201)
+ PromptAD: Learning Prompts with only Normal Samples for Few-Shot Anomaly Detection [[CVPR 2024]](https://arxiv.org/abs/2404.05231)[[code]](https://github.com/FuNz-0/PromptAD)
+ Supervised Anomaly Detection for Complex Industrial Images [[CVPR 2024]](https://openaccess.thecvf.com/content/CVPR2024/html/Baitieva_Supervised_Anomaly_Detection_for_Complex_Industrial_Images_CVPR_2024_paper.html)[[code]](https://github.com/abc-125/segad)
+ Anomaly Heterogeneity Learning for Open-set Supervised Anomaly Detection [[CVPR 2024]](https://arxiv.org/abs/2310.12790)[[code]](https://github.com/mala-lab/AHL)
+ Prompt-enhanced Multiple Instance Learning for Weakly Supervised Anomaly Detection [[CVPR 2024]](https://openaccess.thecvf.com/content/CVPR2024/html/Chen_Prompt-Enhanced_Multiple_Instance_Learning_for_Weakly_Supervised_Video_Anomaly_Detection_CVPR_2024_paper.html)[[code]](https://github.com/Junxi-Chen/PE-MIL)
+ Looking 3D: Anomaly Detection with 2D-3D Alignment [[CVPR 2024]](https://openaccess.thecvf.com/content/CVPR2024/html/Bhunia_Looking_3D_Anomaly_Detection_with_2D-3D_Alignment_CVPR_2024_paper.html)[[homepage]](https://groups.inf.ed.ac.uk/vico/research/Looking3D)[[code]](https://github.com/VICO-UoE/Looking3D)
+ CVPRW: VAND 2.0: Visual Anomaly and Novelty Detection - 2nd Edition [[Challenge and Call for Papers]](https://sites.google.com/view/vand-2-0-cvpr-2024/home)
+ Divide and Conquer: High-Resolution Industrial Anomaly Detection via Memory Efficient Tiled Ensemble [[CVPR 24 Visual Anomaly Detection Workshop]](https://arxiv.org/abs/2403.04932)[[homepage]](https://summerofcode.withgoogle.com/archive/2023/projects/WUSjdxGl)

## ICASSP 2024
+ Implicit Foreground-Guided Network for Anomaly Detection and Localization [[ICASSP 2024]](https://ieeexplore.ieee.org/abstract/document/10446952)
+ Neural Network Training Strategy To Enhance Anomaly Detection Performance: A Perspective On Reconstruction Loss Amplification [[ICASSP 2024]](https://ieeexplore.ieee.org/document/10446942)
+ Patch-Wise Augmentation for Anomaly Detection and Localization [[ICASSP 2024]](https://ieeexplore.ieee.org/document/10446994)
+ A Reconstruction-Based Feature Adaptation for Anomaly Detection with Self-Supervised Multi-Scale Aggregation [[ICASSP 2024]](https://ieeexplore.ieee.org/document/10446766)
+ Feature-Constrained and Attention-Conditioned Distillation Learning for Visual Anomaly Detection [[ICASSP 2024]](https://ieeexplore.ieee.org/document/10448432)
+ CAGEN: Controllable Anomaly Generator using Diffusion Model [[ICASSP 2024]](https://ieeexplore.ieee.org/document/10447663)
+ Mixed-Attention Auto Encoder for Multi-Class Industrial Anomaly Detection [[ICASSP 2024]](https://ieeexplore.ieee.org/document/10446794)

## ICLR 2024
+ AnomalyCLIP: Object-agnostic Prompt Learning for Zero-shot Anomaly Detection [[ICLR 2024]](https://openreview.net/forum?id=buC4E91xZE)[[code]](https://github.com/zqhang/AnomalyCLIP)
+ MuSc: Zero-Shot Industrial Anomaly Classification and Segmentation with Mutual Scoring of the Unlabeled Images[[ICLR 2024]](https://openreview.net/forum?id=AHgc5SMdtd)[[code]](https://github.com/xrli-U/MuSc)][[2025 v2]](https://arxiv.org/abs/2511.10047)

## AAAI 2024
+ Rethinking Reverse Distillation for Multi-Modal Anomaly Detection [[AAAI 2024]](https://ojs.aaai.org/index.php/AAAI/article/view/28687)
+ Unsupervised Continual Anomaly Detection with Contrastively-learned Prompt [[AAAI 2024]](https://ojs.aaai.org/index.php/AAAI/article/view/28153)[[code]](https://github.com/shirowalker/UCAD)
+ Few Shot Part Segmentation Reveals Compositional Logic for Industrial Anomaly Detection [[AAAI 2024]](https://ojs.aaai.org/index.php/AAAI/article/view/28703)[[code]](https://github.com/oopil/PSAD_logical_anomaly_detection)
+ DiAD: A Diffusion-based Framework for Multi-class Anomaly Detection [[AAAI 2024]](https://ojs.aaai.org/index.php/AAAI/article/view/28690)[[code]](https://lewandofskee.github.io/projects/diad)
+ Generating and Reweighting Dense Contrastive Patterns for Unsupervised Anomaly Detection [[AAAI 2024]](https://ojs.aaai.org/index.php/AAAI/article/view/27910)
+ AnomalyDiffusion: Few-Shot Anomaly Image Generation with Diffusion Model [[AAAI 2024]](https://ojs.aaai.org/index.php/AAAI/article/view/28696)[[code]](https://github.com/sjtuplayer/anomalydiffusion)
+ AnomalyGPT: Detecting Industrial Anomalies using Large Vision-Language Models [[AAAI 2024]](https://ojs.aaai.org/index.php/AAAI/article/view/27963)[[code]](https://github.com/CASIA-IVA-Lab/AnomalyGPT)[[project page]](https://anomalygpt.github.io/)
+ A Comprehensive Augmentation Framework for Anomaly Detection [[AAAI 2024]](https://ojs.aaai.org/index.php/AAAI/article/view/28720)


## WACV 2024
+ ReConPatch: Contrastive Patch Representation Learning for Industrial Anomaly Detection [[WACV 2024]](https://openaccess.thecvf.com/content/WACV2024/papers/Hyun_ReConPatch_Contrastive_Patch_Representation_Learning_for_Industrial_Anomaly_Detection_WACV_2024_paper.pdf)
+ Learning Transferable Representations for Image Anomaly Localization Using Dense Pretraining [[WACV 2024]](https://openaccess.thecvf.com/content/WACV2024/papers/He_Learning_Transferable_Representations_for_Image_Anomaly_Localization_Using_Dense_Pretraining_WACV_2024_paper.pdf)[[code]](https://github.com/terrlo/DS2)
+ EfficientAD: Accurate Visual Anomaly Detection at Millisecond-Level Latencies [[WACV 2024]](https://openaccess.thecvf.com/content/WACV2024/papers/Batzner_EfficientAD_Accurate_Visual_Anomaly_Detection_at_Millisecond-Level_Latencies_WACV_2024_paper.pdf)
+ Contextual Affinity Distillation for Image Anomaly Detection [[WACV 2024]](https://openaccess.thecvf.com/content/WACV2024/papers/Zhang_Contextual_Affinity_Distillation_for_Image_Anomaly_Detection_WACV_2024_paper.pdf)
+ Attention Modules Improve Image-Level Anomaly Detection for Industrial Inspection: A DifferNet Case Study [[WACV 2024]](https://openaccess.thecvf.com/content/WACV2024/papers/Vieira_e_Silva_Attention_Modules_Improve_Image-Level_Anomaly_Detection_for_Industrial_Inspection_A_WACV_2024_paper.pdf)
+ PromptAD: Zero-shot Anomaly Detection using Text Prompts [[WACV 2024]](https://openaccess.thecvf.com/content/WACV2024/papers/Li_PromptAD_Zero-Shot_Anomaly_Detection_Using_Text_Prompts_WACV_2024_paper.pdf)
+ High-Fidelity Zero-Shot Texture Anomaly Localization Using Feature Correspondence Analysis [[WACV 2024]](https://openaccess.thecvf.com/content/WACV2024/html/Ardelean_High-Fidelity_Zero-Shot_Texture_Anomaly_Localization_Using_Feature_Correspondence_Analysis_WACV_2024_paper.html)
+ Cheating Depth: Enhancing 3D Surface Anomaly Detection via Depth Simulation [[WACV 2024]](https://openaccess.thecvf.com/content/WACV2024/papers/Zavrtanik_Cheating_Depth_Enhancing_3D_Surface_Anomaly_Detection_via_Depth_Simulation_WACV_2024_paper.pdf)[[code]](https://github.com/VitjanZ/3DSR)

## NeurIPS 2023
+ Real3D-AD: A Dataset of Point Cloud Anomaly Detection [[NeurIPS 2023]](https://openreview.net/pdf?id=zGthDp4yYe)[[code]](https://github.com/M-3LAB/Real3D-AD)[[中文]](https://blog.csdn.net/m0_63828250/article/details/136667168)
+ PAD: A Dataset and Benchmark for Pose-agnostic Anomaly Detection [[NeurIPS 2023]](https://openreview.net/pdf?id=kxFKgqwFNk)[[code]](https://github.com/EricLee0224/PAD)
+ Zero-Shot Anomaly Detection via Batch Normalization [[NeurIPS 2023]](https://openreview.net/pdf?id=d1wjMBYbP1)[[code]](https://github.com/aodongli/zero-shot-ad-via-batch-norm)
+ SANFlow: Semantic-Aware Normalizing Flow for Anomaly Detection and Localization [[NeurIPS 2023]](https://openreview.net/pdf?id=BqZ70BEtuW)
+ Energy-Based Models for Anomaly Detection: A Manifold Diffusion Recovery Approach [[NeurIPS 2023]](https://openreview.net/pdf?id=4nSDDokpfK)
+ Hierarchical Vector Quantized Transformer for Multi-class Unsupervised Anomaly Detection [[NeurIPS 2023]](https://openreview.net/pdf?id=clJTNssgn6)[[code]](https://github.com/RuiyingLu/HVQ-Trans)
+ ReContrast: Domain-Specific Anomaly Detection via Contrastive Reconstruction [[NeurIPS 2023]](https://openreview.net/pdf?id=KYxD9YCQBH)[[code]](https://github.com/guojiajeremy/ReContrast)

## ICML 2023
+ Shape-Guided Dual-Memory Learning for 3D Anomaly Detection [[ICML 2023]](https://openreview.net/forum?id=IkSGn9fcPz)
+ Fascinating Supervisory Signals and Where to Find Them: Deep Anomaly Detection with Scale Learning [[ICML 2023]](https://openreview.net/forum?id=V6PNBRWRil)

## ACM MM 2023
+ EasyNet: An Easy Network for 3D Industrial Anomaly Detection [[ACM MM 2023]](https://arxiv.org/abs/2307.13925)

## ICCV 2023
+ Remembering Normality: Memory-guided Knowledge Distillation for Unsupervised Anomaly Detection [[ICCV 2023]](https://openaccess.thecvf.com/content/ICCV2023/papers/Gu_Remembering_Normality_Memory-guided_Knowledge_Distillation_for_Unsupervised_Anomaly_Detection_ICCV_2023_paper.pdf)
+ Unsupervised Surface Anomaly Detection with Diffusion Probabilistic Model [[ICCV 2023]](https://openaccess.thecvf.com/content/ICCV2023/papers/Zhang_Unsupervised_Surface_Anomaly_Detection_with_Diffusion_Probabilistic_Model_ICCV_2023_paper.pdf)
+ PNI: Industrial Anomaly Detection using Position and Neighborhood Information [[ICCV 2023]](https://openaccess.thecvf.com/content/ICCV2023/papers/Bae_PNI__Industrial_Anomaly_Detection_using_Position_and_Neighborhood_Information_ICCV_2023_paper.pdf)[[code]](https://github.com/wogur110/PNI_Anomaly_Detection)
+ Anomaly Detection using Score-based Perturbation Resilience [[ICCV 2023]](https://openaccess.thecvf.com/content/ICCV2023/papers/Shin_Anomaly_Detection_using_Score-based_Perturbation_Resilience_ICCV_2023_paper.pdf)
+ Template-guided Hierarchical Feature Restoration for Anomaly Detection [[ICCV 2023]](https://openaccess.thecvf.com/content/ICCV2023/papers/Guo_Template-guided_Hierarchical_Feature_Restoration_for_Anomaly_Detection_ICCV_2023_paper.pdf)
+ Focus the Discrepancy: Intra- and Inter-Correlation Learning for Image Anomaly Detection [[ICCV 2023]](https://openaccess.thecvf.com/content/ICCV2023/papers/Yao_Focus_the_Discrepancy_Intra-_and_Inter-Correlation_Learning_for_Image_Anomaly_ICCV_2023_paper.pdf)[[code]](https://github.com/xcyao00/FOD)
+ Anomaly Detection under Distribution Shift [[ICCV 2023]](https://openaccess.thecvf.com/content/ICCV2023/papers/Cao_Anomaly_Detection_Under_Distribution_Shift_ICCV_2023_paper.pdf)[[code]](https://github.com/mala-lab/ADShift)
+ FastRecon: Few-shot Industrial Anomaly Detection via Fast Feature Reconstruction [[ICCV 2023]](https://openaccess.thecvf.com/content/ICCV2023/papers/Fang_FastRecon_Few-shot_Industrial_Anomaly_Detection_via_Fast_Feature_Reconstruction_ICCV_2023_paper.pdf)[[code]](https://github.com/FzJun26th/FastRecon)
+ Inter-Realization Channels: Unsupervised Anomaly Detection Beyond One-Class Classification [[ICCV 2023]](https://openaccess.thecvf.com/content/ICCV2023/papers/McIntosh_Inter-Realization_Channels_Unsupervised_Anomaly_Detection_Beyond_One-Class_Classification_ICCV_2023_paper.pdf)[[code]](https://github.com/DeclanMcIntosh/InReaCh)
+ Removing Anomalies as Noises for Industrial Defect Localization [[ICCV 2023]](https://openaccess.thecvf.com/content/ICCV2023/papers/Lu_Removing_Anomalies_as_Noises_for_Industrial_Defect_Localization_ICCV_2023_paper.pdf)


## MLLM related
+ Myriad: Large Multimodal Model by Applying Vision Experts for Industrial Anomaly Detection [[2023]](https://arxiv.org/abs/2310.19070)[[code]](https://github.com/tzjtatata/Myriad)
+ AnomalyGPT: Detecting Industrial Anomalies using Large Vision-Language Models [[AAAI 2024]](https://arxiv.org/abs/2308.15366)[[code]](https://github.com/CASIA-IVA-Lab/AnomalyGPT)[[project page]](https://anomalygpt.github.io/)
+ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision) [[2023 Section 9.2]](https://arxiv.org/abs/2309.17421)
+ Towards Generic Anomaly Detection and Understanding: Large-scale Visual-linguistic Model (GPT-4V) Takes the Lead [[2023]](https://arxiv.org/abs/2311.02782)[[code]](https://github.com/caoyunkang/GPT4V-for-Generic-Anomaly-Detection)
+ Exploring Grounding Potential of VQA-oriented GPT-4V for Zero-shot Anomaly Detection [[2023]](https://arxiv.org/abs/2311.02612)[[code]](https://github.com/zhangzjn/GPT-4V-AD)
+ Customizing Visual-Language Foundation Models for Multi-modal Anomaly Detection and Reasoning [[2024]](https://arxiv.org/abs/2403.11083)
+ Do LLMs Understand Visual Anomalies? Uncovering LLM Capabilities in Zero-shot Anomaly Detection [[2024]](https://arxiv.org/abs/2404.09654)
+ LogiCode: an LLM-Driven Framework for Logical Anomaly Detection [[2024]](https://arxiv.org/pdf/2406.04687)
+ FabGPT: An Efficient Large Multimodal Model for Complex Wafer Defect Knowledge Queries [[ICCAD 2024]](https://arxiv.org/abs/2407.10810)
+ VMAD: Visual-enhanced Multimodal Large Language Model for Zero-Shot Anomaly Detection [[2024]](https://arxiv.org/abs/2409.20146)
+ Are Anomaly Scores Telling the Whole Story? A Benchmark for Multilevel Anomaly Detection [[2024]](https://arxiv.org/abs/2411.14515)
+ MMAD: The Comprehensive Benchmark for Multimodal Large Language Models in Industrial Anomaly Detection [[ICLR 2025]](https://openreview.net/forum?id=JDiER86r8v)[[Code]](https://github.com/jam-cc/MMAD)  [[Data]](https://huggingface.co/datasets/jiang-cc/MMAD)
+ Can Multimodal Large Language Models be Guided to Improve Industrial Anomaly Detection? [[2025]](https://arxiv.org/abs/2501.15795)
+ EIAD: Explainable Industrial Anomaly Detection Via Multi-Modal Large Language Models [[ICME 2025]](https://arxiv.org/abs/2503.14162v1)
+ Towards Zero-Shot Anomaly Detection and Reasoning with Multimodal Large Language Models [[CVPR 2025]](https://arxiv.org/abs/2502.07601)[[code]](https://xujiacong.github.io/Anomaly-OV/)
+ AnomalyR1: A GRPO-based End-to-end MLLM for Industrial Anomaly Detection [[2025]](https://arxiv.org/abs/2504.11914)
+ Detect, Classify, Act: Categorizing Industrial Anomalies with Multi-Modal Large Language Models [[2025]](https://arxiv.org/abs/2505.02626)
+ Triad: Empowering LMM-based Anomaly Detection with Vision Expert-guided Visual Tokenizer and Manufacturing Process [[ICCV 2025]](https://openaccess.thecvf.com/content/ICCV2025/html/Li_Triad_Empowering_LMM-based_Anomaly_Detection_with_Expert-guided_Region-of-Interest_Tokenizer_and_ICCV_2025_paper.html)
+ LR-IAD:Mask-Free Industrial Anomaly Detection with Logical Reasoning [[2025]](https://arxiv.org/abs/2504.19524)
+ OmniAD: Detect and Understand Industrial Anomaly via Multimodal Reasoning[[2025]](https://arxiv.org/abs/2505.22039)
+ EMIT: Enhancing MLLMs for Industrial Anomaly Detection via Difficulty-Aware [[2025]](https://arxiv.org/abs/2507.21619)[[code]](https://github.com/guanwei49/EMIT)
+ IAD-R1: Reinforcing Consistent Reasoning in Industrial Anomaly Detection [[ICCV 2025]](https://arxiv.org/abs/2508.09178)[[code]](https://github.com/Yanhui-Lee/IAD-R1)
+ AD-FM: Multimodal LLMs for Anomaly Detection via Multi-Stage Reasoning and Fine-Grained Reward Optimization [[AAAI 2026]](https://arxiv.org/abs/2508.04175)
+ AgentIAD: Tool-Augmented Single-Agent for Industrial Anomaly Detection [[2025]](https://arxiv.org/abs/2512.13671)

## CVPR 2023
+ CVPR 2023 Tutorial on "Recent Advances in Anomaly Detection" [[CVPR Workshop 2023(mainly on video anomaly detection)]](https://sites.google.com/view/cvpr2023-tutorial-on-ad/)[[video]](https://www.youtube.com/watch?v=dXxrzWeybBo&feature=youtu.be)
+ Workshop on Vision-Based Industrial Inspection [[CVPR Workshop paper list 2023]](https://openaccess.thecvf.com/CVPR2023_workshops/VISION)
+ Visual Anomaly and Novelty Detection [[CVPR Workshop paper list 2023]](https://openaccess.thecvf.com/CVPR2023_workshops/VAND)
+ Revisiting Reverse Distillation for Anomaly Detection [[CVPR 2023]](https://openaccess.thecvf.com/content/CVPR2023/papers/Tien_Revisiting_Reverse_Distillation_for_Anomaly_Detection_CVPR_2023_paper.pdf) [[code]](https://github.com/tientrandinh/Revisiting-Reverse-Distillation)
+ OmniAL A unifiled CNN framework for unsupervised anomaly localization [[CVPR 2023]](https://openaccess.thecvf.com/content/CVPR2023/papers/Zhao_OmniAL_A_Unified_CNN_Framework_for_Unsupervised_Anomaly_Localization_CVPR_2023_paper.pdf)
+ Explicit Boundary Guided Semi-Push-Pull Contrastive Learning for Supervised Anomaly Detection [[CVPR 2023]](https://arxiv.org/abs/2207.01463)[[code]](https://github.com/xcyao00/BGAD)
+ DeSTSeg: Segmentation Guided Denoising Student-Teacher for Anomaly Detection [[CVPR 2023]](https://arxiv.org/abs/2211.11317)[[code]](https://github.com/apple/ml-destseg)
+ Diversity-Measurable Anomaly Detection [[CVPR 2023]](https://arxiv.org/abs/2303.05047)
+ WinCLIP: Zero-/Few-Shot Anomaly Classification and Segmentation [[CVPR 2023]](https://arxiv.org/abs/2303.14814)
+ SimpleNet: A Simple Network for Image Anomaly Detection and Localization [[CVPR 2023]](https://arxiv.org/abs/2303.15140)[[code]](https://github.com/DonaldRR/SimpleNet)
+ PyramidFlow: High-Resolution Defect Contrastive Localization using Pyramid Normalizing Flow [[CVPR 2023]](https://arxiv.org/abs/2303.02595)[[code]](https://github.com/gasharper/PyramidFlow)
+ Multimodal Industrial Anomaly Detection via Hybrid Fusion [[CVPR 2023]](https://arxiv.org/abs/2303.00601)[[code]](https://github.com/nomewang/M3DM)
+ Prototypical Residual Networks for Anomaly Detection and Localization [[CVPR 2023]](https://arxiv.org/abs/2212.02031)[[code]](https://github.com/xcyao00/PRNet)
+ SQUID: Deep Feature In-Painting for Unsupervised Anomaly Detection [[CVPR 2023]](https://arxiv.org/abs/2111.13495)
+ APRIL-GAN: A Zero-/Few-Shot Anomaly Classification and Segmentation Method for CVPR 2023 VAND Workshop Challenge Tracks 1&2: 1st Place on Zero-shot AD and 4th Place on Few-shot AD [[CVPR 2023 VAND Workshop Challenge]](https://arxiv.org/abs/2305.17382)

## SAM segment anything
+ Segment Anything Is Not Always Perfect: An Investigation of SAM on Different Real-world Applications [[2023 SAM tech report]](https://arxiv.org/abs/2304.05750)
+ SAM Struggles in Concealed Scenes -- Empirical Study on "Segment Anything" [[2023 SAM tech report]](https://arxiv.org/abs/2304.06022)
+ Segment Any Anomaly without Training via Hybrid Prompt Regularization [[2023]](https://arxiv.org/abs/2305.10724) [[code]](https://github.com/caoyunkang/GroundedSAM-zero-shot-anomaly-detection)
+ Application of Segment Anything Model for Civil Infrastructure Defect Assessment [[2023 SAM tech report]](https://arxiv.org/abs/2304.12600)
+ Segment Anything in Defect Detection [[2023]](https://arxiv.org/abs/2311.10245)
+ Unsupervised Continual Anomaly Detection with Contrastively-learned Prompt [[AAAI 2024]](https://ojs.aaai.org/index.php/AAAI/article/view/28153)[[code]](https://github.com/shirowalker/UCAD)
+ ClipSAM: CLIP and SAM Collaboration for Zero-Shot Anomaly Segmentation [[2023]](https://arxiv.org/pdf/2401.12665)
+ A SAM-guided Two-stream Lightweight Model for Anomaly Detection [[2024]](https://arxiv.org/abs/2402.19145)[[code]](https://github.com/StitchKoala/STLM)
+ Inspiring the Next Generation of Segment Anything Models: Comprehensively Evaluate SAM and SAM 2 with Diverse Prompts Towards Context-Dependent Concepts under Different Scenes [[2024]](https://arxiv.org/abs/2412.01240)[[code]](https://github.com/lartpang/SAMs-CDConcepts-Eval)


## ICLR 2023
+ Pushing the Limits of Fewshot Anomaly Detection in Industry Vision: Graphcore [[ICLR 2023]](https://openreview.net/pdf?id=xzmqxHdZAwO)
+ RGI: robust GAN-inversion for mask-free image inpainting and unsupervised pixel-wise anomaly detection [[ICLR 2023]](https://openreview.net/pdf?id=1UbNwQC89a)

## Others
+ Self-Tuning Self-Supervised Anomaly Detection [[2023]](https://openreview.net/forum?id=saj54kqrBj)
+ Model Selection of Anomaly Detectors in the Absence of Labeled Validation Data [[2023]](https://arxiv.org/abs/2310.10461)
+ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision) [[2023 Section 9.2]](https://arxiv.org/abs/2309.17421)
+ End-to-End Augmentation Hyperparameter Tuning for Self-Supervised Anomaly Detection [[2023]](https://arxiv.org/abs/2306.12033)
+ CVPR 1st workshop on Vision-based InduStrial InspectiON [[CVPR 2023 Workshop]](https://vision-based-industrial-inspection.github.io/cvpr-2023/) [[data link]](https://drive.google.com/drive/folders/1TVp_UXJuXudqhC2L3ZKyIDcmQ_2O3JVi)
+ How Low Can You Go? Surfacing Prototypical In-Distribution Samples for Unsupervised Anomaly Detection [Dataset Distillation][[2023]](http://arxiv.org/pdf/2312.03804v1)
+ RAD: A Comprehensive Dataset for Benchmarking the Robustness of Image Anomaly Detection [[CASE 2024]](https://arxiv.org/abs/2406.07176)[[github page]](https://github.com/hustCYQ/RAD-dataset)

-->

<!-- ## Medical (related)
+ Towards Universal Unsupervised Anomaly Detection in Medical Imaging [[2024]](http://arxiv.org/pdf/2401.10637v1)
+ MAEDiff: Masked Autoencoder-enhanced Diffusion Models for Unsupervised Anomaly Detection in Brain Images [[2024]](http://arxiv.org/pdf/2401.10561v1)
+ BMAD: Benchmarks for Medical Anomaly Detection [[2023]](https://arxiv.org/abs/2306.11876)
+ Unsupervised Pathology Detection: A Deep Dive Into the State of the Art [[2023]](https://arxiv.org/abs/2303.00609)
+ Adapting Visual-Language Models for Generalizable Anomaly Detection in Medical Images [[CVPR 2024]](https://arxiv.org/abs/2403.12570)
+ Multi-Image Visual Question Answering for Unsupervised Anomaly Detection [[2024]](http://arxiv.org/abs/2404.07622v1)
+ Contrastive Language Prompting to Ease False Positives in Medical Anomaly Detection [[ISBI 2025]](https://arxiv.org/abs/2411.07546v2)   -->

# Paper Tree (Classification of representative methods)

![PaperTree](https://github.com/M-3LAB/awesome-industrial-anomaly-detection/blob/main/paper_tree.png)

# Timeline

![Timeline](https://github.com/M-3LAB/awesome-industrial-anomaly-detection/blob/main/timeline.png)

# Paper list for industrial image anomaly detection

# Related Survey, Benchmark, and Framework

* Anomalib: A Deep Learning Library for Anomaly Detection [\[ICIP 2022\]](https://ieeexplore.ieee.org/abstract/document/9897283/)[\[code\]](https://github.com/open-edge-platform/anomalib) ⭐ 6,106 | 🐛 75 | 🌐 Python | 📅 2026-09-01
* OpenOOD: Benchmarking Generalized Out-of-Distribution Detection [\[NeurIPS2022v1\]](https://openreview.net/pdf?id=gT6j4_tskUt)[\[2024v1.5\]](https://arxiv.org/abs/2306.09301)[\[github page\]](https://github.com/Jingkang50/OpenOOD) ⭐ 1,071 | 🐛 30 | 🌐 Python | 📅 2025-12-01
* A Deep Learning-based Software for Manufacturing Defect Inspection [\[TII 2017\]](https://ieeexplore.ieee.org/document/9795891)[\[code\]](https://github.com/sundyCoder/DEye) ⭐ 884 | 🐛 1 | 🌐 C++ | 📅 2023-03-14
* IM-IAD: Industrial Image Anomaly Detection Benchmark in Manufacturing [\[TCYB 2024\]](https://arxiv.org/abs/2301.13359)[\[code\]](https://github.com/M-3LAB/open-iad) ⭐ 171 | 🐛 3 | 🌐 Python | 📅 2025-02-20[\[中文\]](https://blog.csdn.net/m0_63828250/article/details/136891730)
* Towards High-Resolution Industrial Image Anomaly Detection [\[2025\]](https://arxiv.org/abs/2508.12931)[\[code\]](https://github.com/cnulab/HiAD) ⭐ 135 | 🐛 5 | 🌐 Python | 📅 2026-02-05
* Large Language Models for Anomaly and Out-of-Distribution Detection: A Survey [\[2024\]](https://arxiv.org/abs/2409.01980)[\[github page\]](https://github.com/rux001/Awesome-LLM-Anomaly-OOD-Detection) ⭐ 124 | 🐛 0 | 📅 2024-09-20
* A Survey on RGB, 3D, and Multimodal Approaches for Unsupervised Industrial Anomaly Detection [\[2024\]](https://arxiv.org/abs/2410.21982)[\[github page\]](https://github.com/Sunny5250/Awesome-Multi-Setting-UIAD) ⭐ 124 | 🐛 1 | 📅 2025-11-14
* Generalized Out-of-Distribution Detection and Beyond in Vision Language Model Era: A Survey [\[2024\]](https://arxiv.org/abs/2407.21794)[\[github page\]](https://github.com/AtsuMiyai/Awesome-OOD-VLM) ⭐ 102 | 🐛 1 | 📅 2025-06-16
* Explainable Anomaly Detection in Images and Videos: A Survey [\[2024\]](https://arxiv.org/pdf/2302.06670)[\[repo\]](https://github.com/wyzjack/Awesome-XAD) ⭐ 36 | 🐛 0 | 📅 2025-04-27
* RAD: A Comprehensive Dataset for Benchmarking the Robustness of Image Anomaly Detection [\[CASE 2024\]](https://arxiv.org/abs/2406.07176)[\[github page\]](https://github.com/hustCYQ/RAD-dataset) ⭐ 12 | 🐛 1 | 🌐 Python | 📅 2024-09-19
* Beyond Academic Benchmarks: Critical Analysis and Best Practices for Visual Industrial Anomaly Detection [\[2025\]](https://arxiv.org/abs/2503.23451)[\[code\]](https://github.com/abc-125/viad-benchmark) ⭐ 7 | 🐛 0 | 🌐 Jupyter Notebook | 📅 2025-08-16
* A Survey of Methods for Automated Quality Control Based on Images [\[IJCV 2023\]](https://link.springer.com/article/10.1007/s11263-023-01822-w)[\[github page\]](https://github.com/jandiers/mvtec-results) ⭐ 5 | 🐛 0 | 📅 2023-06-13
* ASBench: Image Anomalies Synthesis Benchmark for Anomaly Detection [\[TAI 2026\]](https://arxiv.org/abs/2510.07927)[\[code\]](https://github.com/M-3LAB/ASBench) ⭐ 2 | 🐛 0 | 🌐 Jupyter Notebook | 📅 2026-05-11
* A review on computer vision based defect detection and condition assessment of concrete and asphalt civil infrastructure [\[2015\]](https://www.sciencedirect.com/science/article/abs/pii/S1474034615000208)
* Visual-based defect detection and classification approaches for industrial applications: a survey [\[2020\]](https://pdfs.semanticscholar.org/1dfc/080a5f26b5ce78f9ce3e9f106bf7e8124f74.pdf)
* A Unified Survey on Anomaly, Novelty, Open-Set, and Out-of-Distribution Detection: Solutions and Future Challenges [\[TMLR 2022\]](https://arxiv.org/abs/2110.14051)
* Deep Learning for Unsupervised Anomaly Localization in Industrial Images: A Survey [\[TIM 2022\]](http://arxiv.org/pdf/2207.10298)
* A Survey on Unsupervised Industrial Anomaly Detection Algorithms [\[2022\]](https://arxiv.org/abs/2204.11161)
* Benchmarking Unsupervised Anomaly Detection and Localization [\[2022\]](https://arxiv.org/abs/2205.14852)
* Ph.D. thesis of Paul Bergmann(The first author of MVTec AD series) [\[2022\]](https://mediatum.ub.tum.de/1662158)
* CVPR 2023 Tutorial on "Recent Advances in Anomaly Detection" [\[CVPR Workshop 2023\]](https://sites.google.com/view/cvpr2023-tutorial-on-ad/)[\[video\]](https://www.youtube.com/watch?v=dXxrzWeybBo\&feature=youtu.be)
* A Survey on Visual Anomaly Detection: Challenge, Approach, and Prospect [\[2024\]](https://arxiv.org/pdf/2401.16402.pdf)
* AUPIMO: Redefining Visual Anomaly Detection Benchmarks with High Speed and Low Tolerance [\[2024\]](https://arxiv.org/abs/2401.01984)
* Exploring Plain ViT Reconstruction for Multi-class Unsupervised Anomaly Detection [\[CVIU 2025\]](https://www.sciencedirect.com/science/article/abs/pii/S1077314225000311?via%3Dihub)[\[code\]](https://zhangzjn.github.io/projects/ViTAD/)
* A Survey on Foundation-Model-Based Industrial Defect Detection [\[2025\]](https://arxiv.org/abs/2502.19106)
* Foundation Models for Anomaly Detection: Vision and Challenges [\[2025\]](https://arxiv.org/abs/2502.06911)
* A Comprehensive Survey for Real-World Industrial Defect Detection: Challenges, Approaches, and Prospects [\[2025\]](https://www.arxiv.org/abs/2507.13378)

# 2 Unsupervised AD

## 2.1 Feature-Embedding-based Methods

### 2.1.1 Teacher-Student

* Anomaly Detection via Reverse Distillation from One-Class Embedding [\[CVPR 2022\]](http://arxiv.org/pdf/2201.10703)[\[code\]](https://github.com/hq-deng/RD4AD) ⭐ 252 | 🐛 21 | 🌐 Python | 📅 2023-06-23
* Revisiting Reverse Distillation for Anomaly Detection [\[CVPR 2023\]](https://openaccess.thecvf.com/content/CVPR2023/papers/Tien_Revisiting_Reverse_Distillation_for_Anomaly_Detection_CVPR_2023_paper.pdf) [\[code\]](https://github.com/tientrandinh/Revisiting-Reverse-Distillation) ⭐ 170 | 🐛 9 | 🌐 Python | 📅 2023-12-28
* Asymmetric Student-Teacher Networks for Industrial Anomaly Detection [\[WACV 2022\]](https://arxiv.org/pdf/2210.07829.pdf)[\[code\]](https://github.com/marco-rudolph/AST) ⭐ 83 | 🐛 4 | 🌐 Python | 📅 2023-02-28
* Student-Teacher Feature Pyramid Matching for Anomaly Detection [\[2021\]](https://arxiv.org/pdf/2103.04257.pdf)[\[code\]](https://github.com/smiler96/PFM-and-PEFM-for-Image-Anomaly-Detection-and-Segmentation) ⭐ 36 | 🐛 0 | 🌐 Python | 📅 2023-05-24
* PFM and PEFM for Image Anomaly Detection and Segmentation [\[CASE 2022\]](https://ieeexplore.ieee.org/abstract/document/9926547/) [\[TII 2022\]](https://ieeexplore.ieee.org/document/9795121)[\[code\]](https://github.com/smiler96/PFM-and-PEFM-for-Image-Anomaly-Detection-and-Segmentation) ⭐ 36 | 🐛 0 | 🌐 Python | 📅 2023-05-24
* A Discrepancy Aware Framework for Robust Anomaly Detection [\[2023\]](https://arxiv.org/abs/2310.07585)[\[code\]](https://github.com/caiyuxuan1120/DAF) ⭐ 32 | 🐛 3 | 🌐 Python | 📅 2023-11-01
* Informative knowledge distillation for image anomaly segmentation [\[2022\]](https://www.sciencedirect.com/science/article/pii/S0950705122004038/pdfft?md5=758c327dd4d1d052b61a19882f957123\&pid=1-s2.0-S0950705122004038-main.pdf)[\[code\]](https://github.com/caoyunkang/IKD) ⭐ 23 | 🐛 1 | 🌐 Python | 📅 2022-06-11
* Learning deep feature correspondence for unsupervised anomaly detection and segmentation[\[PR 2022\]](https://www.sciencedirect.com/science/article/abs/pii/S0031320322003557)[\[code\]](https://github.com/YoungGod/DFC) ⭐ 21 | 🐛 1 | 🌐 Python | 📅 2022-11-30
* Memory-Distilled Selection for Noise-Robust Anomaly Detection [\[ICML 2026\]](https://arxiv.org/abs/2605.26676)[\[code\]](https://github.com/SirojbekSafarov/MeDS) ⭐ 18 | 🐛 0 | 🌐 Python | 📅 2026-06-17
* Contextual Affinity Distillation for Image Anomaly Detection [\[WACV 2024\]](https://openaccess.thecvf.com/content/WACV2024/papers/Zhang_Contextual_Affinity_Distillation_for_Image_Anomaly_Detection_WACV_2024_paper.pdf)
* Uninformed students: Student-teacher anomaly detection with discriminative latent embeddings [\[CVPR 2020\]](http://arxiv.org/pdf/1911.02357)
* Multiresolution knowledge distillation for anomaly detection [\[CVPR 2021\]](https://arxiv.org/pdf/2011.11108)
* Glancing at the Patch: Anomaly Localization With Global and Local Feature Comparison [\[CVPR 2021\]](https://openaccess.thecvf.com/content/CVPR2021/html/Wang_Glancing_at_the_Patch_Anomaly_Localization_With_Global_and_Local_CVPR_2021_paper.html)
* Reconstruction Student with Attention for Student-Teacher Pyramid Matching [\[2021\]](https://arxiv.org/pdf/2111.15376.pdf)
* Reconstructed Student-Teacher and Discriminative Networks for Anomaly Detection [\[2022\]](https://arxiv.org/pdf/2210.07548.pdf)
* Remembering Normality: Memory-guided Knowledge Distillation for Unsupervised Anomaly Detection [\[ICCV 2023\]](https://openaccess.thecvf.com/content/ICCV2023/papers/Gu_Remembering_Normality_Memory-guided_Knowledge_Distillation_for_Unsupervised_Anomaly_Detection_ICCV_2023_paper.pdf)
* Enhanced multi-scale features mutual mapping fusion based on reverse knowledge distillation for industrial anomaly detection and localization [\[TBD 2024\]](https://ieeexplore.ieee.org/abstract/document/10382612)
* AEKD: Unsupervised auto-encoder knowledge distillation for industrial anomaly detection [\[JMS 2024\]](https://www.sciencedirect.com/science/article/pii/S0278612524000244)
* Masked feature regeneration based asymmetric student–teacher network for anomaly detection [\[Multimedia Tools and Applications 2024\]](https://link.springer.com/article/10.1007/s11042-024-18512-5)
* Feature-Constrained and Attention-Conditioned Distillation Learning for Visual Anomaly Detection [\[ICASSP 2024\]](https://ieeexplore.ieee.org/document/10448432)
* MiniMaxAD: A Lightweight Autoencoder for Feature-Rich Anomaly Detection [\[2024\]](https://arxiv.org/abs/2405.09933)
* Enhanced Fabric Defect Detection with Feature Contrast Interference Suppression [\[TIM 2025\]](https://ieeexplore.ieee.org/abstract/document/10937904)
* Anomaly Detection and Localization via Reverse Distillation with Latent Anomaly Suppression [\[TCSVT 2025\]](https://ieeexplore.ieee.org/abstract/document/10969994)

### 2.1.2 One-Class Classification (OCC)

* SimpleNet: A Simple Network for Image Anomaly Detection and Localization [\[CVPR 2023\]](https://github.com/DonaldRR/SimpleNet) ⭐ 603 | 🐛 30 | 🌐 Python | 📅 2024-08-12[\[code\]](https://github.com/DonaldRR/SimpleNet) ⭐ 603 | 🐛 30 | 🌐 Python | 📅 2024-08-12
* A Unified Anomaly Synthesis Strategy with Gradient Ascent for Industrial Anomaly Detection and Localization [\[ECCV 2024\]](https://arxiv.org/abs/2407.09359)[\[code\]](https://github.com/cqylunlun/GLASS) ⭐ 392 | 🐛 5 | 🌐 Python | 📅 2026-03-30
* Cutpaste: Self-supervised learning for anomaly detection and localization [\[ICCV 2021\]](http://arxiv.org/pdf/2104.04015)[\[unofficial code\]](https://github.com/Runinho/pytorch-cutpaste) ⭐ 261 | 🐛 21 | 🌐 Python | 📅 2024-05-07
* MemSeg: A semi-supervised method for image surface defect detection using differences and commonalities [\[2022\]](https://arxiv.org/pdf/2205.00908.pdf)[\[unofficial code\]](https://github.com/TooTouch/MemSeg) ⭐ 206 | 🐛 8 | 🌐 Jupyter Notebook | 📅 2024-06-24
* SuperSimpleNet: Unifying Unsupervised and Supervised Learning for Fast and Reliable Surface Defect Detection [\[ICPR 2024\]](https://arxiv.org/abs/2408.03143)[\[JIMS 2025\]](https://link.springer.com/article/10.1007/s10845-025-02680-8)[\[code\]](https://github.com/blaz-r/SuperSimpleNet/tree/main) ⭐ 176 | 🐛 1 | 🌐 Python | 📅 2025-10-16
* GeneralAD: Anomaly Detection Across Domains by Attending to Distorted Features [\[ECCV 2024\]](https://arxiv.org/abs/2407.12427)[\[code\]](https://github.com/LucStrater/GeneralAD) ⭐ 58 | 🐛 2 | 🌐 Python | 📅 2024-12-17
* Anomaly Detection under Distribution Shift [\[ICCV 2023\]](https://openaccess.thecvf.com/content/ICCV2023/papers/Cao_Anomaly_Detection_Under_Distribution_Shift_ICCV_2023_paper.pdf)[\[code\]](https://github.com/mala-lab/ADShift) ⭐ 40 | 🐛 6 | 🌐 Python | 📅 2023-10-25
* Progressive Boundary Guided Anomaly Synthesis for Industrial Anomaly Detection [\[TCSVT 2024\]](https://ieeexplore.ieee.org/document/10716437)[\[code\]](https://github.com/cqylunlun/PBAS) ⭐ 36 | 🐛 1 | 🌐 Python | 📅 2026-03-30
* Learning Transferable Representations for Image Anomaly Localization Using Dense Pretraining [\[WACV 2024\]](https://openaccess.thecvf.com/content/WACV2024/papers/He_Learning_Transferable_Representations_for_Image_Anomaly_Localization_Using_Dense_Pretraining_WACV_2024_paper.pdf)[\[code\]](https://github.com/terrlo/DS2) ⭐ 8 | 🐛 1 | 🌐 Python | 📅 2023-10-29
* Patch svdd: Patch-level svdd for anomaly detection and segmentation [\[ACCV 2020\]](https://arxiv.org/pdf/2006.16067.pdf)
* Anomaly detection using improved deep SVDD model with data structure preservation [\[2021\]](https://www.sciencedirect.com/science/article/am/pii/S0167865521001598)
* A Semantic-Enhanced Method Based On Deep SVDD for Pixel-Wise Anomaly Detection [\[2021\]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=\&arnumber=9428370)
* MOCCA: Multilayer One-Class Classification for Anomaly Detection [\[2021\]](http://arxiv.org/pdf/2012.12111)
* Defect Detection of Metal Nuts Applying Convolutional Neural Networks [\[2021\]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=\&arnumber=9529439)
* Panda: Adapting pretrained features for anomaly detection and segmentation [\[2021\]](http://arxiv.org/pdf/2010.05903)
* Mean-shifted contrastive loss for anomaly detection [\[2021\]](https://arxiv.org/pdf/2106.03844.pdf)
* Learning and Evaluating Representations for Deep One-Class Classification [\[2020\]](https://arxiv.org/pdf/2011.02578.pdf)
* Self-supervised learning for anomaly detection with dynamic local augmentation [\[2021\]](https://ieeexplore.ieee.org/ielx7/6287639/6514899/09597511.pdf)
* Contrastive Predictive Coding for Anomaly Detection [\[2021\]](https://arxiv.org/pdf/2107.07820.pdf)
* Consistent estimation of the max-flow problem: Towards unsupervised image segmentation [\[2020\]](http://arxiv.org/pdf/1811.00220)
* End-to-End Augmentation Hyperparameter Tuning for Self-Supervised Anomaly Detection [\[2023\]](https://arxiv.org/abs/2306.12033)
* Dual-Modeling Decouple Distillation for Unsupervised Anomaly Detection [\[ACM MM 2024\]](https://arxiv.org/abs/2408.03888)

### 2.1.3 Distribution-Map

* Cflow-ad: Real-time unsupervised anomaly detection with localization via conditional normalizing flows [\[WACV 2022\]](http://arxiv.org/pdf/2107.12571)[\[code\]](https://github.com/gudovskiy/cflow-ad) ⭐ 270 | 🐛 25 | 🌐 Python | 📅 2023-08-18
* Same same but differnet: Semi-supervised defect detection with normalizing flows [\[WACV 2021\]](http://arxiv.org/pdf/2008.12577)[\[code\]](https://github.com/marco-rudolph/differnet) ⭐ 229 | 🐛 4 | 🌐 Python | 📅 2023-03-21
* Fastflow: Unsupervised anomaly detection and localization via 2d normalizing flows [\[2021\]](https://arxiv.org/pdf/2111.07677.pdf)[\[unofficial code\]](https://github.com/gathierry/FastFlow) ⭐ 169 | 🐛 9 | 🌐 Python | 📅 2023-01-26
* Fully convolutional cross-scale-flows for image-based defect detection [\[WACV 2022\]](http://arxiv.org/pdf/2110.02855)[\[code\]](https://github.com/marco-rudolph/cs-flow) ⭐ 120 | 🐛 1 | 🌐 Python | 📅 2023-03-16
* MSFlow: Multi-Scale Flow-based Framework for Unsupervised Anomaly Detection [\[2024\]](https://arxiv.org/abs/2308.15300)[\[code\]](https://github.com/cool-xuan/msflow) ⭐ 87 | 🐛 8 | 🌐 Python | 📅 2024-03-08
* Collaborative Discrepancy Optimization for Reliable Image Anomaly Localization [\[TII 2023\]](https://ieeexplore.ieee.org/document/10034849)[\[code\]](https://github.com/caoyunkang/CDO) ⭐ 84 | 🐛 11 | 🌐 Python | 📅 2025-03-05
* Position Encoding Enhanced Feature Mapping for Image Anomaly Detection [\[2022\]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=\&arnumber=9926547)[\[code\]](https://github.com/smiler96/PFM-and-PEFM-for-Image-Anomaly-Detection-and-Segmentation) ⭐ 36 | 🐛 0 | 🌐 Python | 📅 2023-05-24
* Distribution Prototype Diffusion Learning for Open-set Supervised Anomaly Detection [\[CVPR 2025\]](https://arxiv.org/abs/2502.20981)[\[code\]](https://github.com/fuyunwang/DPDL) ⭐ 29 | 🐛 0 | 🌐 Python | 📅 2025-02-28
* Anomaly Detection of Defect using Energy of Point Pattern Features within Random Finite Set Framework [\[2021\]](https://arxiv.org/abs/2108.12159)[\[code\]](https://github.com/AmmarKamoona/RFS-Energy-Anomaly-Detection-of-Defect) ⭐ 13 | 🐛 0 | 🌐 Python | 📅 2021-09-23
* Anomaly Detection in Nanofibrous Materials by CNN-Based Self-Similarity [\[Sensors 2018\]](https://www.mdpi.com/1424-8220/18/1/209)
* A Multi-Scale A Contrario method for Unsupervised Image Anomaly Detection [\[2021\]](http://arxiv.org/pdf/2110.02407)
* Modeling the distribution of normal data in pre-trained deep features for anomaly detection [\[2021\]](http://arxiv.org/pdf/2005.14140)
* Transfer Learning Gaussian Anomaly Detection by Fine-Tuning Representations [\[2021\]](https://arxiv.org/pdf/2108.04116.pdf)
* PEDENet: Image anomaly localization via patch embedding and density estimation [\[2022\]](https://arxiv.org/pdf/2110.15525.pdf)
* Unsupervised image anomaly detection and segmentation based on pre-trained feature mapping [\[2022\]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=\&arnumber=9795121)
* Focus your distribution: Coarse-to-fine non-contrastive learning for anomaly detection and localization [\[ICME 2022\]](http://arxiv.org/pdf/2110.04538)
* CAINNFlow: Convolutional block Attention modules and Invertible Neural Networks Flow for anomaly detection and localization tasks [\[2022\]](https://arxiv.org/pdf/2206.01992.pdf)
* AltUB: Alternating Training Method to Update Base Distribution of Normalizing Flow for Anomaly Detection [\[2022\]](https://arxiv.org/pdf/2210.14913.pdf)
* PyramidFlow: High-Resolution Defect Contrastive Localization using Pyramid Normalizing Flow [\[CVPR 2023\]](https://arxiv.org/abs/2303.02595)[\[code\]](https://github.com/gasharper/PyramidFlow)
* Attention Modules Improve Image-Level Anomaly Detection for Industrial Inspection: A DifferNet Case Study [\[WACV 2024\]](https://openaccess.thecvf.com/content/WACV2024/papers/Vieira_e_Silva_Attention_Modules_Improve_Image-Level_Anomaly_Detection_for_Industrial_Inspection_A_WACV_2024_paper.pdf)
* Fascinating Supervisory Signals and Where to Find Them: Deep Anomaly Detection with Scale Learning [\[ICML 2023\]](https://openreview.net/forum?id=V6PNBRWRil)
* FRAnomaly: flow-based rapid anomaly detection from images [\[Applied Intelligence 2024\]](https://link.springer.com/article/10.1007/s10489-024-05332-1)
* Image alignment-based patch distribution framework for anomaly detection [\[ICCVDM 2024\]](https://www.spiedigitallibrary.org/conference-proceedings-of-spie/13063/130630O/Image-alignment-based-patch-distribution-framework-for-anomaly-detection/10.1117/12.3021499.full)
* Multi-Prototype Compactness and Boundary-Aware Synthesis for Unsupervised Anomaly Detection [\[CVPR 2026\]](https://cvpr.thecvf.com/virtual/2026/poster/37868)

### 2.1.4 Memory Bank

* Towards total recall in industrial anomaly detection[\[CVPR 2022\]](http://arxiv.org/pdf/2106.08265)[\[code\]](https://github.com/amazon-science/patchcore-inspection) ⭐ 1,378 | 🐛 83 | 🌐 Python | 📅 2024-07-10
* PaDiM: A Patch Distribution Modeling Framework for Anomaly Detection and Localization [\[ICPR 2021\]](https://link.springer.com/chapter/10.1007/978-3-030-68799-1_35)[\[unofficial code\]](https://github.com/xiahaifeng1995/PaDiM-Anomaly-Detection-Localization-master) ⭐ 488 | 🐛 26 | 🌐 Python | 📅 2023-11-29
* CFA: Coupled-Hypersphere-Based Feature Adaptation for Target-Oriented Anomaly Localization[\[2022\]](https://arxiv.org/pdf/2206.04325.pdf)[\[code\]](https://github.com/sungwool/CFA_for_anomaly_localization) ⭐ 121 | 🐛 6 | 🌐 Python | 📅 2022-07-14
* Target before Shooting: Accurate Anomaly Detection and Localization under One Millisecond via Cascade Patch Retrieval [\[TIP 2024\]](https://arxiv.org/abs/2308.06748)[\[code\]](https://github.com/flyinghu123/CPR) ⭐ 119 | 🐛 0 | 🌐 Python | 📅 2024-10-28
* PNI : Industrial Anomaly Detection using Position and Neighborhood Information [\[ICCV 2023\]](https://openaccess.thecvf.com/content/ICCV2023/papers/Bae_PNI__Industrial_Anomaly_Detection_using_Position_and_Neighborhood_Information_ICCV_2023_paper.pdf)[\[code\]](https://github.com/wogur110/PNI_Anomaly_Detection) ⭐ 68 | 🐛 0 | 🌐 Python | 📅 2023-09-28
* RAID: Retrieval-Augmented Anomaly Detection [\[CVPR 2026\]](https://arxiv.org/abs/2602.19611)[\[code\]](https://github.com/Mingxiu-Cai/RAID) ⭐ 46 | 🐛 6 | 🌐 Python | 📅 2026-04-20
* Grid-Based Continuous Normal Representation for Anomaly Detection [\[2024\]](https://arxiv.org/abs/2402.18293)[\[code\]](https://github.com/tae-mo/GRAD) ⭐ 34 | 🐛 1 | 🌐 Python | 📅 2024-11-27
* REB: Reducing Biases in Representation for Industrial Anomaly Detection [\[2023\]](https://arxiv.org/abs/2308.12577)[\[code\]](https://github.com/ShuaiLYU/REB) ⭐ 26 | 🐛 9 | 🌐 Jupyter Notebook | 📅 2024-01-22
* EAGLE: Expert-Augmented Attention Guidance for Tuning-Free Industrial Anomaly Detection in Multimodal Large Language Models [\[2026\]](https://arxiv.org/abs/2602.17419)[\[code\]](https://github.com/shengtun/Eagle-Anomaly-Detection) ⭐ 20 | 🐛 1 | 🌐 Python | 📅 2026-02-24
* Self-supervised Context Learning for Visual Inspection of Industrial Defects [\[2023\]](https://arxiv.org/abs/2311.06504)[\[code\]](https://github.com/wangpeng000/VisualInspection) ⭐ 16 | 🐛 0 | 🌐 Python | 📅 2024-02-21
* VQ-Flow: Taming Normalizing Flows for Multi-Class Anomaly Detection via Hierarchical Vector Quantization [\[2024\]](https://arxiv.org/abs/2409.00942)[\[code\]](https://github.com/cool-xuan/vqflow) ⭐ 16 | 🐛 1 | 🌐 Python | 📅 2024-03-14
* Inter-Realization Channels: Unsupervised Anomaly Detection Beyond One-Class Classification [\[ICCV 2023\]](https://openaccess.thecvf.com/content/ICCV2023/papers/McIntosh_Inter-Realization_Channels_Unsupervised_Anomaly_Detection_Beyond_One-Class_Classification_ICCV_2023_paper.pdf)[\[code\]](https://github.com/DeclanMcIntosh/InReaCh) ⭐ 15 | 🐛 0 | 🌐 Python | 📅 2026-01-12
* Tailored Transformation Invariance for Industrial Anomaly Detection [\[2025\]](https://arxiv.org/abs/2509.17670)[\[code\]](https://github.com/marietteschonfeld/LWinNN) ⭐ 6 | 🐛 0 | 🌐 Python | 📅 2026-07-03
* ReConPatch: Contrastive Patch Representation Learning for Industrial Anomaly Detection [\[WACV 2024\]](https://openaccess.thecvf.com/content/WACV2024/papers/Hyun_ReConPatch_Contrastive_Patch_Representation_Learning_for_Industrial_Anomaly_Detection_WACV_2024_paper.pdf)
* Sub-image anomaly detection with deep pyramid correspondences [\[2020\]](https://arxiv.org/pdf/2005.02357.pdf)
* Semi-orthogonal embedding for efficient unsupervised anomaly segmentation [\[2021\]](https://arxiv.org/pdf/2105.14737.pdf)
* Anomaly Detection Via Self-Organizing Map [\[2021\]](http://arxiv.org/pdf/2107.09903)
* Industrial Image Anomaly Localization Based on Gaussian Clustering of Pretrained Feature [\[2021\]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=\&arnumber=9479740)
* FAPM: Fast Adaptive Patch Memory for Real-time Industrial Anomaly Detection[\[2022\]](https://arxiv.org/pdf/2211.07381.pdf)
* N-pad: Neighboring Pixel-based Industrial Anomaly Detection [\[2022\]](https://arxiv.org/pdf/2210.08768.pdf)
* Multi-scale patch-based representation learning for image anomaly detection and segmentation [\[2022\]](https://openaccess.thecvf.com/content/WACV2022/papers/Tsai_Multi-Scale_Patch-Based_Representation_Learning_for_Image_Anomaly_Detection_and_Segmentation_WACV_2022_paper.pdf)
* SPot-the-Difference Self-supervised Pre-training for Anomaly Detection and Segmentation [\[ECCV 2022\]](https://arxiv.org/pdf/2207.14315.pdf)
* Diversity-Measurable Anomaly Detection [\[CVPR 2023\]](https://arxiv.org/abs/2303.05047)
* SelFormaly: Towards Task-Agnostic Unified Anomaly Detection[\[2023\]](https://arxiv.org/abs/2307.12540)
* PointCore: Efficient Unsupervised Point Cloud Anomaly Detector Using Local-Global Features [\[2024\]](https://arxiv.org/abs/2403.01804)
* DMAD: Dual Memory Bank for Real-World Anomaly Detection [\[2024\]](https://arxiv.org/abs/2403.12362)
* A Reconstruction-Based Feature Adaptation for Anomaly Detection with Self-Supervised Multi-Scale Aggregation [\[ICASSP 2024\]](https://ieeexplore.ieee.org/document/10446766)
* AnomalousPatchCore: Exploring the Use of Anomalous Samples in Industrial Anomaly Detection [\[ECCVW 2024\]](https://arxiv.org/abs/2408.15113)
* FOCT: Few-shot Industrial Anomaly Detection with Foreground-aware Online Conditional Transport [\[ACM MM 2024\]](https://dl.acm.org/doi/10.1145/3664647.3680771)
* Unsupervised, Online and On-The-Fly Anomaly Detection For Non-Stationary Image Distributions [\[ECCV 2024\]](https://eccv2024.ecva.net/virtual/2024/poster/2289)\[\[code]]
* Anomaly as Non-Conformity via Training-Free Graph Laplacian Energy Minimization [\[CVPR 2026\]](https://cvpr.thecvf.com/virtual/2026/poster/38295)

### 2.1.5 Vison Language AD

* AnomalyGPT: Detecting Industrial Anomalies using Large Vision-Language Models [\[AAAI 2024\]](https://arxiv.org/abs/2308.15366)[\[code\]](https://github.com/CASIA-IVA-Lab/AnomalyGPT) ⭐ 1,132 | 🐛 61 | 🌐 Python | 📅 2023-12-20[\[project page\]](https://anomalygpt.github.io/)
* AnomalyCLIP: Object-agnostic Prompt Learning for Zero-shot Anomaly Detection [\[ICLR 2024\]](https://openreview.net/forum?id=buC4E91xZE)[\[code\]](https://github.com/zqhang/AnomalyCLIP) ⭐ 663 | 🐛 74 | 🌐 Python | 📅 2025-07-08
* PromptAD: Learning Prompts with only Normal Samples for Few-Shot Anomaly Detection [\[CVPR 2024\]](https://arxiv.org/abs/2404.05231)[\[code\]](https://github.com/FuNz-0/PromptAD) ⭐ 212 | 🐛 33 | 🌐 Python | 📅 2024-09-01
* MultiADS: Defect-aware Supervision for Multi-type Anomaly Detection and Segmentation in Zero-Shot Learning [\[ICCV 2025\]](https://arxiv.org/abs/2504.06740)[\[code\]](https://github.com/boschresearch/MultiADS) ⭐ 68 | 🐛 3 | 🌐 Python | 📅 2026-08-28
* AnoVL: Adapting Vision-Language Models for Unified Zero-shot Anomaly Localization [\[2023\]](https://arxiv.org/abs/2308.15939)[\[code\]](https://github.com/hq-deng/AnoVL) ⭐ 57 | 🐛 9 | 🌐 Python | 📅 2023-09-07
* CoPS: Conditional Prompt Synthesis for Zero-Shot Anomaly Detection [\[2025\]](https://arxiv.org/abs/2508.03447)[\[code\]](https://github.com/cqylunlun/CoPS) ⭐ 52 | 🐛 1 | 🌐 Python | 📅 2026-06-02
* Triad: Empowering LMM-based Anomaly Detection with Vision Expert-guided Visual Tokenizer and Manufacturing Process [\[ICCV 2025\]](https://arxiv.org/abs/2503.13184)[\[code\]](https://github.com/tzjtatata/Triad) ⭐ 42 | 🐛 0 | 🌐 Python | 📅 2025-08-15
* Customizing Visual-Language Foundation Models for Multi-modal Anomaly Detection and Reasoning [\[2024\]](https://arxiv.org/abs/2403.11083)[\[code\]](https://github.com/Xiaohao-Xu/Customizable-VLM) ⭐ 30 | 🐛 0 | 🌐 Python | 📅 2025-05-24
* EMIT: Enhancing MLLMs for Industrial Anomaly Detection via Difficulty-Aware [\[2025\]](https://arxiv.org/abs/2507.21619)[\[code\]](https://github.com/guanwei49/EMIT) ⭐ 29 | 🐛 1 | 🌐 Python | 📅 2026-01-24
* EAGLE: Expert-Augmented Attention Guidance for Tuning-Free Industrial Anomaly Detection in Multimodal Large Language Models [\[2026\]](https://arxiv.org/abs/2602.17419)[\[code\]](https://github.com/shengtun/Eagle-Anomaly-Detection) ⭐ 20 | 🐛 1 | 🌐 Python | 📅 2026-02-24
* VLMDiff: Leveraging Vision-Language Models for Multi-Class Anomaly Detection with Diffusion [\[2025\]](https://arxiv.org/abs/2511.08173)[\[code\]](https://github.com/giddyyupp/VLMDiff) ⭐ 7 | 🐛 1 | 🌐 Python | 📅 2026-02-02
* AnoPLe: Few-Shot Anomaly Detection via Bi-directional Prompt Learning with Only Normal Samples [\[2024\]](https://arxiv.org/abs/2408.13516)[\[code\]](https://github.com/YoojLee/AnoPLe) ⭐ 1 | 🐛 2 | 📅 2026-03-31
* IAD-GPT: Advancing Visual Knowledge in Multimodal Large Language Model for Industrial Anomaly Detection [\[2025\]](https://arxiv.org/abs/2510.16036)[\[code\]](https://github.com/LiZeWen1225/IAD-GPT) ⭐ 1 | 🐛 1 | 📅 2025-01-17
* Random Word Data Augmentation with CLIP for Zero-Shot Anomaly Detection [\[BMVC 2023\]](https://arxiv.org/abs/2308.11119)
* WinCLIP: Zero-/Few-Shot Anomaly Classification and Segmentation [\[CVPR 2023\]](https://arxiv.org/abs/2303.14814)
* ClipSAM: CLIP and SAM Collaboration for Zero-Shot Anomaly Segmentation [\[2023\]](https://arxiv.org/pdf/2401.12665)
* CLIP-AD: A Language-Guided Staged Dual-Path Model for Zero-shot Anomaly Detection [\[2023\]](https://arxiv.org/abs/2311.00453)
* Anomaly Detection by Adapting a pre-trained Vision Language Model [\[2024\]](https://arxiv.org/abs/2403.09493)
* Do LLMs Understand Visual Anomalies? Uncovering LLM Capabilities in Zero-shot Anomaly Detection [\[2024\]](https://arxiv.org/abs/2404.09654)
* FiLo: Zero-Shot Anomaly Detection by Fine-Grained Description and High-Quality Localization [\[2024\]](https://arxiv.org/abs/2404.13671)
* Dual-Image Enhanced CLIP for Zero-Shot Anomaly Detection [\[2024\]](https://arxiv.org/abs/2405.04782)
* GlocalCLIP: Object-agnostic Global-Local Prompt Learning for Zero-shot Anomaly Detection [\[2024\]](https://arxiv.org/abs/2411.06071)
* UniVAD: A Training-free Unified Model for Few-shot Visual Anomaly Detection [\[2024\]](https://arxiv.org/abs/2412.03342)[\[code\]](https://uni-vad.github.io/#)
* One-to-Normal: Anomaly Personalization for Few-shot Anomaly Detection [\[NeurIPS 2024\]](https://openreview.net/pdf?id=tIzW3l2uaN)
* SEM-CLIP: Precise Few-Shot Learning for Nanoscale Defect Detection in Scanning Electron Microscope Image [\[2025\]](https://arxiv.org/abs/2502.14884)
* PA-CLIP: Enhancing Zero-Shot Anomaly Detection through Pseudo-Anomaly Awareness [\[2025\]](https://arxiv.org/abs/2503.01292)
* Language-Assisted Feature Transformation for Anomaly Detection [\[ICLR 2025\]](https://openreview.net/forum?id=2p03KljxE9)
* Detect, Classify, Act: Categorizing Industrial Anomalies with Multi-Modal Large Language Models [\[2025\]](https://arxiv.org/abs/2505.02626)
* AnomalyR1: A GRPO-based End-to-end MLLM for Industrial Anomaly Detection [\[2025\]](https://arxiv.org/abs/2504.11914)
* OmniAD: Detect and Understand Industrial Anomaly via Multimodal Reasoning[\[2025\]](https://arxiv.org/abs/2505.22039)

## 2.2 Reconstruction-Based Methods

### 2.2.1 Autoencoder (AE)

* RealNet: A Feature Selection Network with Realistic Synthetic Anomaly for Anomaly Detection [\[CVPR 2024\]](https://arxiv.org/abs/2403.05897)[\[code\]](https://github.com/cnulab/RealNet) ⭐ 426 | 🐛 79 | 🌐 Python | 📅 2025-02-12
* Draem-a discriminatively trained reconstruction embedding for surface anomaly detection [\[ICCV 2021\]](http://arxiv.org/pdf/2108.07610)[\[code\]](https://github.com/vitjanz/draem) ⭐ 291 | 🐛 18 | 🌐 Python | 📅 2023-01-02
* Self-supervised predictive convolutional attentive block for anomaly detection [\[CVPR 2022 oral\]](http://arxiv.org/pdf/2111.09099)[\[code\]](https://github.com/ristea/sspcab) ⭐ 163 | 🐛 11 | 🌐 Python | 📅 2023-04-20
* Natural Synthetic Anomalies for Self-supervised Anomaly Detection and Localization [\[ECCV 2022\]](https://arxiv.org/pdf/2109.15222.pdf)[\[code\]](https://github.com/hmsch/natural-synthetic-anomalies) ⭐ 73 | 🐛 1 | 🌐 Jupyter Notebook | 📅 2023-04-20
* Self-Supervised Masked Convolutional Transformer Block for Anomaly Detection [\[TPAMI 2022\]](https://arxiv.org/pdf/2209.12148.pdf)[\[code\]](https://github.com/ristea/ssmctb) ⭐ 62 | 🐛 4 | 🌐 Python | 📅 2023-12-08
* DSR: A dual subspace re-projection network for surface anomaly detection [\[ECCV 2022\]](https://arxiv.org/pdf/2208.01521.pdf)[\[code\]](https://github.com/VitjanZ/DSR_anomaly_detection) ⭐ 56 | 🐛 4 | 🌐 Python | 📅 2022-07-18
* FastRecon: Few-shot Industrial Anomaly Detection via Fast Feature Reconstruction [\[ICCV 2023\]](https://openaccess.thecvf.com/content/ICCV2023/papers/Fang_FastRecon_Few-shot_Industrial_Anomaly_Detection_via_Fast_Feature_Reconstruction_ICCV_2023_paper.pdf)[\[code\]](https://github.com/FzJun26th/FastRecon) ⭐ 48 | 🐛 6 | 🌐 Python | 📅 2023-10-25
* FAIR: Frequency-aware Image Restoration for Industrial Visual Anomaly Detection [\[2023\]](https://arxiv.org/abs/2309.07068)[\[code\]](https://github.com/liutongkun/FAIR) ⭐ 47 | 🐛 1 | 🌐 Python | 📅 2025-08-23
* Reconstruction from edge image combined with color and gradient difference for industrial surface anomaly detection [\[2022\]](http://arxiv.org/pdf/2210.14485)[\[code\]](https://github.com/liutongkun/edgrec) ⭐ 29 | 🐛 1 | 🌐 Python | 📅 2024-08-15
* Variational Autoencoder for Anomaly Detection: A Comparative Study [\[2024\]](https://arxiv.org/abs/2408.13561)[\[code\]](https://github.com/endtheme123/VAE-compare) ⭐ 13 | 🐛 0 | 🌐 Python | 📅 2026-06-17
* Revitalizing Reconstruction Models for Multi-class Anomaly Detection via Class-Aware Contrastive Learning [\[2024\]](https://arxiv.org/abs/2412.04769)[\[code\]](https://github.com/LGC-AD/AD-LGC) ⭐ 11 | 🐛 2 | 🌐 Python | 📅 2025-11-26
* Improving unsupervised defect segmentation by applying structural similarity to autoencoders [\[2018\]](https://arxiv.org/pdf/1807.02011.pdf)
* Automatic Fabric Defect Detection with a Multi-Scale Convolutional Denoising Autoencoder Network Model [\[Sensors 2018\]](https://www.mdpi.com/1424-8220/18/4/1064)
* An Unsupervised-Learning-Based Approach for Automated Defect Inspection on Textured Surfaces [\[TIM 2018\]](https://ieeexplore.ieee.org/abstract/document/8281622)
* Unsupervised anomaly detection using style distillation [\[2020\]](https://ieeexplore.ieee.org/ielx7/6287639/6514899/09288772.pdf)
* Unsupervised two-stage anomaly detection [\[2021\]](https://arxiv.org/pdf/2103.11671.pdf)
* Dfr: Deep feature reconstruction for unsupervised anomaly segmentation [\[Neurocomputing 2020\]](https://arxiv.org/pdf/2012.07122.pdf)
* Unsupervised anomaly segmentation via multilevel image reconstruction and adaptive attention-level transition [\[2021\]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=\&arnumber=9521893)
* Encoding structure-texture relation with p-net for anomaly detection in retinal images [\[2020\]](http://arxiv.org/pdf/2008.03632)
* Improved anomaly detection by training an autoencoder with skip connections on images corrupted with stain-shaped noise [\[2021\]](http://arxiv.org/pdf/2008.12977)
* Unsupervised anomaly detection for surface defects with dual-siamese network [\[2022\]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=\&arnumber=9681338)
* Divide-and-assemble: Learning block-wise memory for unsupervised anomaly detection [\[ICCV 2021\]](https://openaccess.thecvf.com/content/ICCV2021/papers/Hou_Divide-and-Assemble_Learning_Block-Wise_Memory_for_Unsupervised_Anomaly_Detection_ICCV_2021_paper.pdf)
* Spatial Contrastive Learning for Anomaly Detection and Localization [\[2022\]](https://ieeexplore.ieee.org/ielx7/6287639/9668973/09709224.pdf)
* Superpixel masking and inpainting for self-supervised anomaly detection [\[BMVC 2020\]](https://www.bmvc2020-conference.com/assets/papers/0275.pdf)
* Iterative image inpainting with structural similarity mask for anomaly detection [\[2020\]](https://openreview.net/pdf?id=b4ach0lGuYO)
* Self-Supervised Masking for Unsupervised Anomaly Detection and Localization [\[2022\]](https://arxiv.org/pdf/2205.06568.pdf)
* Reconstruction by inpainting for visual anomaly detection [\[PR 2021\]](https://www.sciencedirect.com/science/article/pii/S0031320320305094/pdfft?md5=9bbe942017de1acd3a97034bc2d4a8fb\&pid=1-s2.0-S0031320320305094-main.pdf)
* Self-Supervised Training with Autoencoders for Visual Anomaly Detection [\[2022\]](https://arxiv.org/pdf/2206.11723.pdf)
* Iterative energy-based projection on a normal data manifold for anomaly localization [\[2019\]](https://arxiv.org/pdf/2002.03734.pdf)
* Towards visually explaining variational autoencoders [\[2020\]](http://arxiv.org/pdf/1911.07389)
* Deep generative model using unregularized score for anomaly detection with heterogeneous complexity [\[2020\]](http://arxiv.org/pdf/1807.05800)
* Anomaly localization by modeling perceptual features [\[2020\]](https://arxiv.org/pdf/2008.05369.pdf)
* Image anomaly detection using normal data only by latent space resampling [\[2020\]](https://pdfs.semanticscholar.org/cb59/dab0a725c0b511f3140ea47ea0967f3643bf.pdf)
* Noise-to-Norm Reconstruction for Industrial Anomaly Detection and Localization [\[2023\]](https://arxiv.org/abs/2307.02836)
* Patch-wise Auto-Encoder for Visual Anomaly Detection [\[2023\]](https://arxiv.org/abs/2308.00429)
* Template-guided Hierarchical Feature Restoration for Anomaly Detection [\[ICCV 2023\]](https://openaccess.thecvf.com/content/ICCV2023/papers/Guo_Template-guided_Hierarchical_Feature_Restoration_for_Anomaly_Detection_ICCV_2023_paper.pdf)
* Produce Once, Utilize Twice for Anomaly Detection [\[2023\]](https://arxiv.org/abs/2312.12913)
* Implicit Foreground-Guided Network for Anomaly Detection and Localization [\[ICASSP 2024\]](https://ieeexplore.ieee.org/abstract/document/10446952)
* Neural Network Training Strategy To Enhance Anomaly Detection Performance: A Perspective On Reconstruction Loss Amplification [\[ICASSP 2024\]](https://ieeexplore.ieee.org/document/10446942)
* Patch-Wise Augmentation for Anomaly Detection and Localization [\[ICASSP 2024\]](https://ieeexplore.ieee.org/document/10446994)
* A Reconstruction-Based Feature Adaptation for Anomaly Detection with Self-Supervised Multi-Scale Aggregation [\[ICASSP 2024\]](https://ieeexplore.ieee.org/document/10446766)
* Neural Network Training Strategy To Enhance Anomaly Detection Performance: A Perspective On Reconstruction Loss Amplification [\[ICASSP 2024\]](https://ieeexplore.ieee.org/abstract/document/10446942)
* Mixed-Attention Auto Encoder for Multi-Class Industrial Anomaly Detection [\[ICASSP 2024\]](https://ieeexplore.ieee.org/document/10446794)
* Dual-Constraint Autoencoder and Adaptive Weighted Similarity Spatial Attention for Unsupervised Anomaly Detection [\[TII 2024\]](https://ieeexplore.ieee.org/abstract/document/10504620)
* Multi-feature Reconstruction Network using Crossed-mask Restoration for Unsupervised Anomaly Detection [\[2024\]](https://arxiv.org/abs/2404.13273)
* R3D-AD: Reconstruction via Diffusion for 3D Anomaly Detection [\[ECCV 2024\]](https://arxiv.org/abs/2407.10862)[\[homepage\]](https://zhouzheyuan.github.io/r3d-ad)
* Visual defect obfuscation based self-supervised anomaly detection [\[2024\]](https://www.nature.com/articles/s41598-024-69698-5)
* RcAE: Recursive Reconstruction Framework for Unsupervised Industrial Anomaly Detection [\[AAAI 2026\]](https://arxiv.org/abs/2512.11284)

### 2.2.2 Generative Adversarial Networks (GANs)

* Few-shot defect image generation via defect-aware feature manipulation [\[AAAI 2023\]](https://arxiv.org/abs/2303.02389)[\[code\]](https://github.com/Ldhlwh/DFMGAN) ⭐ 158 | 🐛 5 | 🌐 Python | 📅 2024-12-19
* Omni-frequency Channel-selection Representations for Unsupervised Anomaly Detection [\[TIP 2023\]](https://ieeexplore.ieee.org/abstract/document/10192551/)[\[code\]](https://github.com/zhangzjn/ocr-gan) ⭐ 52 | 🐛 4 | 🌐 Python | 📅 2022-07-14
* CKAAD: Boosting Fine-Grained Visual Anomaly Detection with Coarse-Knowledge-Aware Adversarial Learning [\[AAAI 2025\]](https://arxiv.org/abs/2412.12850)[\[code\]](https://github.com/Faustinaqq/CKAAD) ⭐ 17 | 🐛 1 | 🌐 Python | 📅 2025-05-29
* Learning semantic context from normal samples for unsupervised anomaly detection [\[AAAI 2021\]](https://ojs.aaai.org/index.php/AAAI/article/download/16420/16227)
* Anoseg: Anomaly segmentation network using self-supervised learning [\[2021\]](https://arxiv.org/pdf/2110.03396.pdf)
* A Surface Defect Detection Method Based on Positive Samples [\[PRICAI 2018\]](https://link.springer.com/chapter/10.1007/978-3-319-97310-4_54)

### 2.2.3 Transformer

* Focus the Discrepancy: Intra- and Inter-Correlation Learning for Image Anomaly Detection [\[ICCV 2023\]](https://openaccess.thecvf.com/content/ICCV2023/papers/Yao_Focus_the_Discrepancy_Intra-_and_Inter-Correlation_Learning_for_Image_Anomaly_ICCV_2023_paper.pdf)[\[code\]](https://github.com/xcyao00/FOD) ⭐ 47 | 🐛 12 | 🌐 Python | 📅 2026-06-02
* Multi-scale feature reconstruction network for industrial anomaly detection [\[KBS 2024\]](https://www.sciencedirect.com/science/article/pii/S095070512401284X)[\[code\]](https://github.com/Ehteshamciitwah/MSFR) ⭐ 19 | 🐛 0 | 🌐 Python | 📅 2025-07-01
* Context Enhancement with Reconstruction as Sequence for Unified Unsupervised Anomaly Detection[\[2024\]](https://arxiv.org/abs/2409.06285)[\[code\]](https://github.com/Nothingtolose9979/RAS) ⭐ 6 | 🐛 0 | 🌐 Python | 📅 2025-02-20
* VT-ADL: A vision transformer network for image anomaly detection and localization [\[ISIE 2021\]](http://arxiv.org/pdf/2104.10036)
* ADTR: Anomaly Detection Transformer with Feature Reconstruction [\[2022\]](https://arxiv.org/pdf/2209.01816.pdf)
* AnoViT: Unsupervised Anomaly Detection and Localization With Vision Transformer-Based Encoder-Decoder [\[2022\]](https://ieeexplore.ieee.org/ielx7/6287639/6514899/09765986.pdf)
* HaloAE: An HaloNet based Local Transformer Auto-Encoder for Anomaly Detection and Localization [\[2022\]](https://arxiv.org/pdf/2208.03486.pdf)
* Inpainting transformer for anomaly detection [\[ICIAP 2022\]](https://arxiv.org/pdf/2104.13897.pdf)
* Masked Swin Transformer Unet for Industrial Anomaly Detection [\[2022\]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=\&arnumber=9858596)
* Masked Transformer for image Anomaly Localization [\[TII 2022\]](http://arxiv.org/pdf/2210.15540)
* AMI-Net: Adaptive Mask Inpainting Network for Industrial Anomaly Detection and Localization [\[TASE 2024\]](https://ieeexplore.ieee.org/abstract/document/10445116)
* Prior Normality Prompt Transformer for Multi-class Industrial Image Anomaly Detection [\[TII 2024\]](https://arxiv.org/abs/2406.11507)
* Masked Autoencoder Self Pre-Training for Defect Detection in Microelectronics [\[2025\]](https://arxiv.org/html/2504.10021v1)
* Vague Prototype-Oriented Diffusion Model for Multi-Class Anomaly Detection [\[ICML 2024\]](https://openreview.net/forum?id=FvLd8Gr7xq)
* MC3D-AD: A Unified Geometry-aware Reconstruction Model for Multi-category 3D Anomaly Detection [\[IJCAI 2025\]](https://arxiv.org/abs/2505.01969)

### 2.2.4 Diffusion Model

* DiffusionAD: Denoising Diffusion for Anomaly Detection [\[2023\]](https://arxiv.org/abs/2303.08730)[\[code\]](https://github.com/HuiZhang0812/DiffusionAD) ⭐ 222 | 🐛 8 | 🌐 Python | 📅 2024-01-31
* Anomaly Detection with Conditioned Denoising Diffusion Models [\[2023\]](https://arxiv.org/abs/2305.15956)[\[code\]](https://github.com/arimousa/DDAD) ⭐ 202 | 🐛 11 | 🌐 Python | 📅 2024-06-30
* GLAD: Towards Better Reconstruction with Global and Local Adaptive Diffusion Models for Unsupervised Anomaly Detection [\[ECCV 2024\]](https://arxiv.org/abs/2406.07487)[\[code\]](https://github.com/hyao1/GLAD) ⭐ 117 | 🐛 33 | 🌐 Python | 📅 2025-05-13
* InvAD: Inversion-based Reconstruction-Free Anomaly Detection with Diffusion Models [\[CVPR 2026\]](https://arxiv.org/abs/2504.05662)[\[code\]](https://github.com/SkyShunsuke/InversionAD) ⭐ 62 | 🐛 2 | 🌐 Python | 📅 2026-07-27
* TransFusion -- A Transparency-Based Diffusion Model for Anomaly Detection [\[ECCV 2024\]](https://arxiv.org/abs/2311.09999)[\[code\]](https://github.com/MaticFuc/ECCV_TransFusion) ⭐ 52 | 🐛 6 | 🌐 Python | 📅 2025-04-07
* Tackling Structural Hallucination in Image Translation with Local Diffusion [\[ECCV 2024 oral\]](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/10498.pdf)[\[code\]](https://github.com/edshkim98/LocalDiffusion-Hallucination) ⭐ 28 | 🐛 3 | 🌐 Python | 📅 2026-07-03
* Mixture Prototype Flow Matching for Open-Set Supervised Anomaly Detection [\[ICML 2026\]](https://arxiv.org/abs/2605.02438)[\[code\]](https://github.com/fuyunwang/MPFM-OSAD) ⭐ 5 | 🐛 1 | 🌐 Python | 📅 2026-05-19
* FDP: A Frequency-Decomposition Preprocessing Pipeline for Unsupervised Anomaly Detection in Brain MRI [\[AAAI 2026\]](https://arxiv.org/abs/2511.12899)[\[code\]](https://github.com/ls1rius/MRI_FDP) ⭐ 4 | 🐛 0 | 🌐 Python | 📅 2026-05-23
* How and Why: Taming Flow Matching for Unsupervised Anomaly Detection and Localization [\[2025\]](https://arxiv.org/abs/2508.05461) [\[code\]](https://github.com/lil-wayne-0319/fmad) ⭐ 3 | 🐛 1 | 🌐 Python | 📅 2026-04-13
* AnoDDPM: Anomaly Detection With Denoising Diffusion Probabilistic Models Using Simplex Noise [\[CVPR Workshop 2022\]](http://dro.dur.ac.uk/36134/1/36134.pdf)
* Unsupervised Visual Defect Detection with Score-Based Generative Model[\[2022\]](https://arxiv.org/pdf/2211.16092.pdf)
* Unsupervised Surface Anomaly Detection with Diffusion Probabilistic Model [\[ICCV 2023\]](https://openaccess.thecvf.com/content/ICCV2023/papers/Zhang_Unsupervised_Surface_Anomaly_Detection_with_Diffusion_Probabilistic_Model_ICCV_2023_paper.pdf)
* Removing Anomalies as Noises for Industrial Defect Localization [\[ICCV 2023\]](https://openaccess.thecvf.com/content/ICCV2023/papers/Lu_Removing_Anomalies_as_Noises_for_Industrial_Defect_Localization_ICCV_2023_paper.pdf)
* LafitE: Latent Diffusion Model with Feature Editing for Unsupervised Multi-class Anomaly Detection [\[2023\]](https://arxiv.org/abs/2307.08059)
* DiAD: A Diffusion-based Framework for Multi-class Anomaly Detection [\[AAAI 2024\]](https://ojs.aaai.org/index.php/AAAI/article/view/28690)[\[code\]](https://lewandofskee.github.io/projects/diad)
* D3AD: Dynamic Denoising Diffusion Probabilistic Model for Anomaly Detection [\[2024\]](https://arxiv.org/abs/2401.04463)
* HDM: Hybrid Diffusion Model for Unified Image Anomaly Detection [\[2025\]](https://arxiv.org/abs/2502.19200)
* One-for-More: Continual Diffusion Model for Anomaly Detection [\[CVPR 2025\]](https://arxiv.org/abs/2502.19848)

### 2.2.5 Others

* Anomaly Detection using Score-based Perturbation Resilience [\[ICCV 2023\]](https://openaccess.thecvf.com/content/ICCV2023/papers/Shin_Anomaly_Detection_using_Score-based_Perturbation_Resilience_ICCV_2023_paper.pdf)

## 2.3 Supervised AD

### More Normal Samples With (Less Abnormal Samples or Weak Labels)

* AnomalyDiffusion: Few-Shot Anomaly Image Generation with Diffusion Model [\[AAAI 2024\]](https://ojs.aaai.org/index.php/AAAI/article/view/28696)[\[code\]](https://github.com/sjtuplayer/anomalydiffusion) ⭐ 328 | 🐛 52 | 🌐 Jupyter Notebook | 📅 2024-08-11
* SuperSimpleNet: Unifying Unsupervised and Supervised Learning for Fast and Reliable Surface Defect Detection [\[ICPR 2024\]](https://arxiv.org/abs/2408.03143)[\[JIMS 2025\]](https://link.springer.com/article/10.1007/s10845-025-02680-8)[\[code\]](https://github.com/blaz-r/SuperSimpleNet/tree/main) ⭐ 176 | 🐛 1 | 🌐 Python | 📅 2025-10-16
* Few-shot defect image generation via defect-aware feature manipulation [\[AAAI 2023\]](https://arxiv.org/abs/2303.02389)[\[code\]](https://github.com/Ldhlwh/DFMGAN) ⭐ 158 | 🐛 5 | 🌐 Python | 📅 2024-12-19
* Catching Both Gray and Black Swans: Open-set Supervised Anomaly Detection [\[CVPR 2022\]](http://arxiv.org/pdf/2203.14506)[\[code\]](https://github.com/Choubo/DRA) ⭐ 95 | 🐛 7 | 🌐 Python | 📅 2022-04-11
* Explainable deep few-shot anomaly detection with deviation networks [\[2021\]](https://arxiv.org/pdf/2108.00462.pdf)[\[code\]](https://github.com/Choubo/deviation-network-image) ⭐ 89 | 🐛 3 | 🌐 Python | 📅 2022-10-29
* Anomaly Heterogeneity Learning for Open-set Supervised Anomaly Detection [\[CVPR 2024\]](https://arxiv.org/abs/2310.12790)[\[code\]](https://github.com/mala-lab/AHL) ⭐ 58 | 🐛 3 | 🌐 Python | 📅 2024-04-20
* Prototypical Residual Networks for Anomaly Detection and Localization [\[CVPR 2023\]](https://arxiv.org/abs/2212.02031)[\[code\]](https://github.com/xcyao00/PRNet) ⭐ 39 | 🐛 10 | 🌐 Python | 📅 2023-07-07
* VarAD: Lightweight High-Resolution Image Anomaly Detection via Visual Autoregressive Modeling [\[TII 2025\]](https://arxiv.org/abs/2412.17263)[\[code\]](https://github.com/caoyunkang/VarAD) ⭐ 29 | 🐛 6 | 📅 2025-04-14
* Distribution Prototype Diffusion Learning for Open-set Supervised Anomaly Detection [\[CVPR 2025\]](https://arxiv.org/abs/2502.20981)[\[code\]](https://github.com/fuyunwang/DPDL) ⭐ 29 | 🐛 0 | 🌐 Python | 📅 2025-02-28
* ArcAD: Anomaly-Rectified Calibration for Cold-Start Supervised Anomaly Detection [\[ECCV 2026\]](https://arxiv.org/abs/2607.02252)[\[code\]](https://github.com/LGC-AD/ArcAD) ⭐ 14 | 🐛 1 | 🌐 Python | 📅 2026-07-04
* Mixture Prototype Flow Matching for Open-Set Supervised Anomaly Detection [\[ICML 2026\]](https://arxiv.org/abs/2605.02438)[\[code\]](https://github.com/fuyunwang/MPFM-OSAD) ⭐ 5 | 🐛 1 | 🌐 Python | 📅 2026-05-19
* Self-Tuning Self-Supervised Image Anomaly Detection [\[KDD 2025\]](https://arxiv.org/abs/2306.12033) [\[code\]](https://github.com/jaeminyoo/ST-SSAD) ⭐ 0 | 🐛 0 | 🌐 Python | 📅 2025-07-21
* Neural batch sampling with reinforcement learning for semi-supervised anomaly detection [\[ECCV 2020\]](https://www.ri.cmu.edu/wp-content/uploads/2020/05/WenHsuan_MSR_Thesis-1.pdf)
* Explainable Deep One-Class Classification [\[ICLR 2020\]](https://arxiv.org/pdf/2007.01760.pdf)
* Attention guided anomaly localization in images [\[ECCV 2020\]](http://arxiv.org/pdf/1911.08616)
* Mixed supervision for surface-defect detection: From weakly to fully supervised learning [\[2021\]](https://arxiv.org/pdf/2104.06064.pdf)
* Anomaly Clustering: Grouping Images into Coherent Clusters of Anomaly Types[\[WACV 2023\]](https://openaccess.thecvf.com/content/WACV2023/html/Sohn_Anomaly_Clustering_Grouping_Images_Into_Coherent_Clusters_of_Anomaly_Types_WACV_2023_paper.html)
* Efficient Anomaly Detection with Budget Annotation Using Semi-Supervised Residual Transformer [\[2023\]](https://arxiv.org/abs/2306.03492)
* BiaS: Incorporating Biased Knowledge to Boost Unsupervised Image Anomaly Localization [\[TSMC 2024\]](https://ieeexplore.ieee.org/abstract/document/10402554)
* DMAD: Dual Memory Bank for Real-World Anomaly Detection [\[2024\]](https://arxiv.org/abs/2403.12362)
* AnomalousPatchCore: Exploring the Use of Anomalous Samples in Industrial Anomaly Detection [\[ECCVW 2024\]](https://arxiv.org/abs/2408.15113)

### More Abnormal Samples

* Small Object Few-shot Segmentation for Vision-based Industrial Inspection [\[2024\]](https://arxiv.org/abs/2407.21351)[\[code\]](https://github.com/zhangzilongc/SOFS) ⭐ 43 | 🐛 4 | 🌐 Python | 📅 2026-08-21
* Supervised Anomaly Detection for Complex Industrial Images [\[2024\]](https://arxiv.org/abs/2405.04953)[\[code\]](https://github.com/abc-125/segad) ⭐ 35 | 🐛 1 | 🌐 Python | 📅 2025-09-07
* Dual Attention U-Net with Feature Infusion: Pushing the Boundaries of Multiclass Defect Segmentation [\[2023\]](https://arxiv.org/abs/2312.14053)[\[code\]](https://github.com/RashaAlshawi/Dual-Attention-U-Net-with-Feature-Infusion-Pushing-the-Boundaries-of-Multiclass-Defect-Segmentation) ⭐ 18 | 🐛 1 | 🌐 Jupyter Notebook | 📅 2024-11-01
* ISP-AD: a large-scale real-world dataset for advancing industrial anomaly detection with synthetic and real defects [\[JIMS 2026\]](https://link.springer.com/article/10.1007/s10845-025-02778-z)[\[code\]](https://github.com/p4ulk/isp-ad) ⭐ 0 | 🐛 0 | 🌐 Python | 📅 2026-07-10[\[data\]](https://zenodo.org/records/14911042)
* Logit Inducing With Abnormality Capturing for Semi-Supervised Image Anomaly Detection [\[2022\]](https://ieeexplore.ieee.org/document/9885240)
* An effective framework of automated visual surface defect detection for metal parts [\[2021\]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=\&arnumber=9475966)
* Interleaved Deep Artifacts-Aware Attention Mechanism for Concrete Structural Defect Classification [\[TIP 2021\]](https://eprints.keele.ac.uk/10031/1/TIP24Jul2021.pdf)
* Reference-based defect detection network [\[TIP 2021\]](http://arxiv.org/pdf/2108.04456)
* Fabric defect detection using tactile information [\[ICRA 2021\]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=\&arnumber=9561092)
* A lightweight spatial and temporal multi-feature fusion network for defect detection [\[TIP 2020\]](http://nrl.northumbria.ac.uk/id/eprint/48908/1/ALightweightSpatialandTemporalMulti-featureFusionNetworkforDefectDetection.pdf)
* SDD-CNN: Small Data-Driven Convolution Neural Networks for Subtle Roller Defect Inspection [\[Robotics and Computer-Integrated Manufacturing 2020\]](https://www.sciencedirect.com/science/article/abs/pii/S0736584518304770)
* A High-Efficiency Fully Convolutional Networks for Pixel-Wise Surface Defect Detection [\[IEEE Access 2019\]](https://ieeexplore.ieee.org/abstract/document/8624360)
* SDD-CNN: Small Data-Driven Convolution Neural Networks for Subtle Roller Defect Inspection [\[Applied Sciences 2019\]](https://www.mdpi.com/2076-3417/9/7/1364)
* Autonomous Structural Visual Inspection Using Region-Based Deep Learning for Detecting Multiple Damage Types [\[CACIE 2018\]](https://dl.acm.org/doi/abs/10.1111/mice.12334)
* Detection and segmentation of manufacturing defects with convolutional neural networks and transfer learning [\[2018\]](https://europepmc.org/articles/pmc6512995?pdf=render)
* Automatic Metallic Surface Defect Detection and Recognition with Convolutional Neural Networks [\[Applied Sciences 2018\]](https://www.mdpi.com/2076-3417/8/9/1575)
* Real-time Detection of Steel Strip Surface Defects Based on Improved YOLO Detection Network [\[IFAC-PapersOnLine 2018\]](https://www.sciencedirect.com/science/article/pii/S2405896318321001)
* Domain adaptation for automatic OLED panel defect detection using adaptive support vector data description [\[IJCV 2017\]](https://link.springer.com/article/10.1007/s11263-016-0953-y)
* Automatic Defect Detection of Fasteners on the Catenary Support Device Using Deep Convolutional Neural Network [\[TIM 2017\]](https://ieeexplore.ieee.org/abstract/document/8126877)
* Deep Active Learning for Civil Infrastructure Defect Detection and Classification [\[Computing in civil engineering 2017\]](https://ascelibrary.org/doi/abs/10.1061/9780784480823.036)
* A fast and robust convolutional neural network-based defect detection model in product quality control [\[IJAMT 2017\]](https://link.springer.com/article/10.1007/s00170-017-0882-0)
* Defects Detection Based on Deep Learning and Transfer Learning [\[Metallurgical & Mining Industry 2015\]](https://web.s.ebscohost.com/abstract?direct=true\&profile=ehost\&scope=site\&authtype=crawler\&jrnl=20760507\&AN=115932631\&h=Xxf%2binGAfPaFG1E3Net%2fQQIu5U%2fD2pFkichv9fJ63Bx%2bjW2wr5y1UZWYaHbOQCE%2bZc%2bYJQz117Xd06J3IxAbSg%3d%3d\&crl=c\&resultNs=AdminWebAuth\&resultLocal=ErrCrlNotAuth\&crlhashurl=login.aspx%3fdirect%3dtrue%26profile%3dehost%26scope%3dsite%26authtype%3dcrawler%26jrnl%3d20760507%26AN%3d115932631)
* Design of deep convolutional neural network architectures for automated feature extraction in industrial inspection [\[CIRP annals 2016\]](https://www.sciencedirect.com/science/article/abs/pii/S0007850616300725)
* Decision Fusion Network with Perception Fine-tuning for Defect Classification [\[2023\]](https://arxiv.org/abs/2309.12630)
* Global Context Aggregation Network for Lightweight Saliency Detection of Surface Defects [\[2023\]](https://arxiv.org/abs/2309.12641)
* MemoryMamba: Memory-Augmented State Space Model for Defect Recognition [\[2024\]](https://arxiv.org/abs/2405.03673)
* SEM-CLIP: Precise Few-Shot Learning for Nanoscale Defect Detection in Scanning Electron Microscope Image [\[2025\]](https://arxiv.org/abs/2502.14884)
* SynSur: An end-to-end generative pipeline for synthetic industrial surface defect generation and detection [\[2025\]](https://arxiv.org/abs/2604.26633)

# 3 Other Research Direction

## 3.1 Zero/Few-Shot AD

### Zero-Shot AD

* Segment Any Anomaly without Training via Hybrid Prompt Regularization [\[2023\]](https://arxiv.org/abs/2305.10724) [\[code\]](https://github.com/caoyunkang/GroundedSAM-zero-shot-anomaly-detection) ⭐ 844 | 🐛 12 | 🌐 Jupyter Notebook | 📅 2025-02-22
* AnomalyCLIP: Object-agnostic Prompt Learning for Zero-shot Anomaly Detection [\[ICLR 2024\]](https://openreview.net/forum?id=buC4E91xZE)[\[code\]](https://github.com/zqhang/AnomalyCLIP) ⭐ 663 | 🐛 74 | 🌐 Python | 📅 2025-07-08
* MuSc: Zero-Shot Industrial Anomaly Classification and Segmentation with Mutual Scoring of the Unlabeled Images[\[ICLR 2024\]](https://openreview.net/forum?id=AHgc5SMdtd)[\[code\]](https://github.com/xrli-U/MuSc) ⭐ 438 | 🐛 33 | 🌐 Python | 📅 2026-04-25[\[2025 v2\]](https://arxiv.org/abs/2511.10047)
* MuSc-V2: Zero-Shot Multimodal Industrial Anomaly Classification and Segmentation with Mutual Scoring of Unlabeled Samples [\[TPAMI 2026\]](https://arxiv.org/abs/2511.10047)[\[code\]](https://github.com/HUST-SLOW/MuSc-V2) ⭐ 339 | 🐛 2 | 🌐 Python | 📅 2026-05-14
* AdaCLIP: Adapting CLIP with Hybrid Learnable Prompts for Zero-Shot Anomaly Detection [\[ECCV 2024\]](https://arxiv.org/abs/2407.15795)[\[code\]](https://github.com/caoyunkang/AdaCLIP) ⭐ 310 | 🐛 37 | 🌐 Python | 📅 2025-07-07
* APRIL-GAN: A Zero-/Few-Shot Anomaly Classification and Segmentation Method for CVPR 2023 VAND Workshop Challenge Tracks 1&2: 1st Place on Zero-shot AD and 4th Place on Few-shot AD [\[CVPRW 2023\]](https://arxiv.org/abs/2305.17382)[\[code\]](https://github.com/ByChelsea/VAND-APRIL-GAN) ⭐ 270 | 🐛 22 | 🌐 Python | 📅 2025-08-01
* AdaptCLIP: Adapting CLIP for Universal Visual Anomaly Detection [\[AAAI 2026\]](https://arxiv.org/abs/2505.09926)[\[code\]](https://github.com/gaobb/AdaptCLIP) ⭐ 182 | 🐛 3 | 🌐 Python | 📅 2026-02-05
* WinCLIP: Zero-/Few-Shot Anomaly Classification and Segmentation [\[CVPR 2023\]](https://arxiv.org/abs/2303.14814) [\[unofficial code in AnomalyCLIP\]](https://github.com/zqhang/Accurate-WinCLIP-pytorch) ⭐ 139 | 🐛 7 | 🌐 Python | 📅 2024-12-08 [\[unofficial code in SAA\]](https://github.com/caoyunkang/WinClip) ⭐ 429 | 🐛 11 | 🌐 Python | 📅 2024-03-16 [\[unofficial code in mala-lab\]](https://github.com/mala-lab/WinCLIP) ⭐ 73 | 🐛 2 | 🌐 Python | 📅 2025-07-17
* VCP-CLIP: A visual context prompting model for zero-shot anomaly segmentation [\[ECCV 2024\]](https://arxiv.org/abs/2407.12276)[\[code\]](https://github.com/xiaozhen228/VCP-CLIP) ⭐ 106 | 🐛 4 | 🌐 Python | 📅 2025-06-04
* IAD-R1: Reinforcing Consistent Reasoning in Industrial Anomaly Detection [\[AAAI 2026\]](https://arxiv.org/abs/2508.09178)[\[code\]](https://github.com/Yanhui-Lee/IAD-R1) ⭐ 96 | 🐛 17 | 🌐 Python | 📅 2025-12-09
* PointAD: Comprehending 3D Anomalies from Points and Pixels for Zero-shot 3D Anomaly Detection [\[NeurIPS 2024\]](https://arxiv.org/abs/2410.00320)[\[code\]](https://github.com/zqhang/PointAD) ⭐ 89 | 🐛 23 | 🌐 Python | 📅 2025-05-29
* MultiADS: Defect-aware Supervision for Multi-type Anomaly Detection and Segmentation in Zero-Shot Learning [\[ICCV 2025\]](https://arxiv.org/abs/2504.06740)[\[code\]](https://github.com/boschresearch/MultiADS) ⭐ 68 | 🐛 3 | 🌐 Python | 📅 2026-08-28
* AnomalyVFM -- Transforming Vision Foundation Models into Zero-Shot Anomaly Detectors [\[CVPR 2026\]](https://arxiv.org/abs/2601.20524)[\[code\]](https://github.com/MaticFuc/AnomalyVFM) ⭐ 66 | 🐛 2 | 🌐 Python | 📅 2026-04-21
* AnoVL: Adapting Vision-Language Models for Unified Zero-shot Anomaly Localization [\[2023\]](https://arxiv.org/abs/2308.15939)[\[code\]](https://github.com/hq-deng/AnoVL) ⭐ 57 | 🐛 9 | 🌐 Python | 📅 2023-09-07
* CoPS: Conditional Prompt Synthesis for Zero-Shot Anomaly Detection [\[2025\]](https://arxiv.org/abs/2508.03447)[\[code\]](https://github.com/cqylunlun/CoPS) ⭐ 52 | 🐛 1 | 🌐 Python | 📅 2026-06-02
* MRAD: Zero-Shot Anomaly Detection with Memory-Driven Retrieval [\[ICLR 2026\]](https://openreview.net/forum?id=TQkFiW3AEX)[\[code\]](https://github.com/CROVO1026/MRAD) ⭐ 52 | 🐛 4 | 🌐 Python | 📅 2026-03-13
* Fine-grained Abnormality Prompt Learning for Zero-shot Anomaly Detection [\[ICCV 2025\]](https://arxiv.org/abs/2410.10289)[\[code\]](https://github.com/mala-lab/FAPrompt) ⭐ 47 | 🐛 3 | 🌐 Python | 📅 2026-03-07
* AF-CLIP: Zero-Shot Anomaly Detection via Anomaly-Focused CLIP Adaptation [\[ACM MM 2025\]](https://arxiv.org/abs/2507.19949)[\[code\]](https://github.com/Faustinaqq/AF-CLIP) ⭐ 37 | 🐛 3 | 🌐 Python | 📅 2025-07-12
* MoECLIP: Patch-Specialized Experts for Zero-shot Anomaly Detection [\[CVPR 2026\]](https://arxiv.org/abs/2603.03101)[\[code\]](https://github.com/CoCoRessa/MoECLIP) ⭐ 37 | 🐛 1 | 🌐 Python | 📅 2026-05-28
* Crane: Context-Guided Prompt Learning and Attention Refinement for Zero-Shot Anomaly Detections [\[2025\]](https://arxiv.org/abs/2504.11055)[\[code\]](https://github.com/AlirezaSalehy/Crane) ⭐ 35 | 🐛 0 | 🌐 Python | 📅 2026-08-28
* FB-CLIP: Fine-Grained Zero-Shot Anomaly Detection with Foreground-Background Disentanglement [\[CVPR 2026\]](https://arxiv.org/abs/2603.19608)[\[code\]](https://github.com/Xi-Mu-Yu/FB-CLIP) ⭐ 19 | 🐛 0 | 🌐 Python | 📅 2026-07-03
* AG-VAS: Anchor-Guided Zero-Shot Visual Anomaly Segmentation with Large Multimodal Models [\[CVPR 2026\]](https://arxiv.org/abs/2603.01305)[\[code\]](https://github.com/xiaozhen228/AG-VAS) ⭐ 11 | 🐛 2 | 📅 2026-06-30
* Random Word Data Augmentation with CLIP for Zero-Shot Anomaly Detection [\[BMVC 2023\]](https://arxiv.org/abs/2308.11119)
* Zero-Shot Batch-Level Anomaly Detection [\[2023\]](https://arxiv.org/abs/2302.07849)
* Zero-shot versus Many-shot: Unsupervised Texture Anomaly Detection [\[WACV 2023\]](https://ieeexplore.ieee.org/document/10030870)
* MAEDAY: MAE for few and zero shot AnomalY-Detection [\[2022\]](https://arxiv.org/pdf/2211.14307.pdf)
* Anomaly Detection in an Open World by a Neuro-symbolic Program on Zero-shot Symbols [\[IROS 2022 Workshop\]](https://openreview.net/pdf?id=Bg3ZO3nXJuA)
* CLIP-AD: A Language-Guided Staged Dual-Path Model for Zero-shot Anomaly Detection [\[2023\]](https://arxiv.org/abs/2311.00453)
* PromptAD: Zero-shot Anomaly Detection using Text Prompts [\[WACV 2024\]](https://openaccess.thecvf.com/content/WACV2024/papers/Li_PromptAD_Zero-Shot_Anomaly_Detection_Using_Text_Prompts_WACV_2024_paper.pdf)
* High-Fidelity Zero-Shot Texture Anomaly Localization Using Feature Correspondence Analysis [\[WACV 2024\]](https://openaccess.thecvf.com/content/WACV2024/html/Ardelean_High-Fidelity_Zero-Shot_Texture_Anomaly_Localization_Using_Feature_Correspondence_Analysis_WACV_2024_paper.html)
* ClipSAM: CLIP and SAM Collaboration for Zero-Shot Anomaly Segmentation [\[2023\]](https://arxiv.org/pdf/2401.12665)
* Model Selection of Zero-shot Anomaly Detectors in the Absence of Labeled Validation Data [\[2024\]](https://arxiv.org/abs/2310.10461)
* Do LLMs Understand Visual Anomalies? Uncovering LLM Capabilities in Zero-shot Anomaly Detection [\[2024\]](https://arxiv.org/abs/2404.09654)
* FiLo: Zero-Shot Anomaly Detection by Fine-Grained Description and High-Quality Localization [\[2024\]](https://arxiv.org/abs/2404.13671)
* Dual-Image Enhanced CLIP for Zero-Shot Anomaly Detection [\[2024\]](https://arxiv.org/abs/2405.04782)
* Investigating the Semantic Robustness of CLIP-based Zero-Shot Anomaly Segmentation [\[2024\]](https://arxiv.org/abs/2405.07969)
* SAM-LAD: Segment Anything Model Meets Zero-Shot Logic Anomaly Detection [\[2024\]](https://arxiv.org/abs/2406.00625)
* Towards Zero-shot Point Cloud Anomaly Detection: A Multi-View Projection Framework [\[2024\]](https://arxiv.org/abs/2409.13162)
* VMAD: Visual-enhanced Multimodal Large Language Model for Zero-Shot Anomaly Detection [\[2024\]](https://arxiv.org/abs/2409.20146)
* GlocalCLIP: Object-agnostic Global-Local Prompt Learning for Zero-shot Anomaly Detection [\[2024\]](https://arxiv.org/abs/2411.06071)
* Towards Zero-shot 3D Anomaly Localization [\[WACV 2025\]](https://arxiv.org/abs/2412.04304)
* Towards Zero-Shot Anomaly Detection and Reasoning with Multimodal Large Language Models [\[2025\]](https://arxiv.org/abs/2502.07601)[\[code\]](https://xujiacong.github.io/Anomaly-OV/)
* PA-CLIP: Enhancing Zero-Shot Anomaly Detection through Pseudo-Anomaly Awareness [\[2025\]](https://arxiv.org/abs/2503.01292)
* MFP-CLIP: Exploring the Efficacy of Multi-Form Prompts for Zero-Shot Industrial Anomaly Detection [\[2025\]](https://arxiv.org/abs/2503.12910)
* EIAD: Explainable Industrial Anomaly Detection Via Multi-Modal Large Language Models [\[2025\]](https://arxiv.org/abs/2503.14162v1)
* On the Problem of Consistent Anomalies in Zero-Shot Industrial Anomaly Detection [\[TMLR 2025\]](https://openreview.net/forum?id=o2MRb5QZ34)
* FE-CLIP: Frequency Enhanced CLIP Model for Zero-Shot Anomaly Detection and Segmentation [\[ICCV 2025\]](https://openaccess.thecvf.com/content/ICCV2025/papers/Gong_FE-CLIP_Frequency_Enhanced_CLIP_Model_for_Zero-Shot_Anomaly_Detection_and_ICCV_2025_paper.pdf)
* MRAD: Zero-Shot Anomaly Detection with Memory-Driven Retrieval [\[ICLR 2026\]](https://arxiv.org/abs/2602.00522)
* PromptMoE: Generalizable Zero-Shot Anomaly Detection via Visually-Guided Prompt Mixtures [\[AAAI 2026\]](https://arxiv.org/abs/2511.18116)[\[code\]](https://github.com/yourusername/PromptMoE)
* Exploring High-order-aware Prompt Learning for Zero-shot Anomaly Detection [\[AAAI 2026\]](https://ojs.aaai.org/index.php/AAAI/article/view/38029)
* DLVP-CLIP: Enhancing Fine-Grained Zero-Shot Anomaly Detection via Dynamic Local Visual Prompting [\[CVPR 2026\]](https://cvpr.thecvf.com/virtual/2026/poster/39303)
* From Attraction to Equilibrium: Physics-Inspired Semantic Gravitons for Zero-Shot Anomaly Detection [\[CVPR 2026\]](https://cvpr.thecvf.com/virtual/2026/poster/39782)
* Defect-aware Hybrid Prompt Optimization via Progressive Tuning for Zero-Shot Multi-type Anomaly Detection and Segmentation [\[ECCV 2026\]](https://arxiv.org/abs/2512.09446)
* Towards Active Real-to-Twin Inspection: A New Paradigm for Zero-Shot Anomaly Detection[\[IEEE-CYBER 2026\]](https://arxiv.org/abs/2605.25407)
* Robust Zero-shot Anomaly Detection under Limited Auxiliary Anomaly Priors [\[ECCV 2026\]](https://arxiv.org/abs/2606.29428)

### Few-Shot AD

* AnomalyGPT: Detecting Industrial Anomalies using Large Vision-Language Models [\[AAAI 2024\]](https://arxiv.org/abs/2308.15366)[\[code\]](https://github.com/CASIA-IVA-Lab/AnomalyGPT) ⭐ 1,132 | 🐛 61 | 🌐 Python | 📅 2023-12-20[\[project page\]](https://anomalygpt.github.io/)
* Registration based few-shot anomaly detection [\[ECCV 2022 oral\]](https://arxiv.org/pdf/2207.07361.pdf)[\[code\]](https://github.com/MediaBrain-SJTU/RegAD) ⭐ 323 | 🐛 14 | 🌐 Python | 📅 2022-09-07
* PromptAD: Learning Prompts with only Normal Samples for Few-Shot Anomaly Detection [\[CVPR 2024\]](https://arxiv.org/abs/2404.05231)[\[code\]](https://github.com/FuNz-0/PromptAD) ⭐ 212 | 🐛 33 | 🌐 Python | 📅 2024-09-01
* InCTRL: Toward Generalist Anomaly Detection via In-context Residual Learning with Few-shot Sample Prompts [\[CVPR 2024\]](https://arxiv.org/abs/2403.06495)[\[code\]](https://github.com/mala-lab/InCTRL) ⭐ 202 | 🐛 7 | 🌐 Python | 📅 2025-07-17
* SeaS: Few-shot Industrial Anomaly Image Generation with Separation and Sharing Fine-tuning [\[ICCV 2025\]](https://arxiv.org/pdf/2410.14987)[\[code\]](https://github.com/HUST-SLOW/SeaS) ⭐ 160 | 🐛 21 | 🌐 Python | 📅 2025-08-04
* MetaUAS: Universal Anomaly Segmentation with One-Prompt Meta-Learning [\[NeurIPS 2024\]](https://arxiv.org/abs/2505.09265)[\[code\]](https://github.com/gaobb/MetaUAS) ⭐ 94 | 🐛 5 | 🌐 Python | 📅 2026-02-07
* Foundation Visual Encoders Are Secretly Few-Shot Anomaly Detectors [\[2025\]](https://arxiv.org/abs/2510.01934)[\[code\]](https://github.com/ymxlzgy/FoundAD) ⭐ 81 | 🐛 4 | 🌐 Python | 📅 2026-03-16
* Foundation Visual Encoders Are Secretly Few-Shot Anomaly Detectors [\[ICLR 2026\]](https://arxiv.org/abs/2510.01934)[\[code\]](https://github.com/ymxlzgy/FoundAD) ⭐ 81 | 🐛 4 | 🌐 Python | 📅 2026-03-16
* Learning to Detect Multi-class Anomalies with Just One Normal Image Prompt [\[ECCV 2024\]](https://arxiv.org/abs/2505.09264)[\[code\]](https://github.com/gaobb/OneNIP) ⭐ 78 | 🐛 2 | 🌐 Python | 📅 2026-01-18
* Optimizing PatchCore for Few/many-shot Anomaly Detection [\[2023\]](https://arxiv.org/abs/2307.10792)[\[code\]](https://github.com/scortexio/patchcore-few-shot/) ⭐ 68 | 🐛 2 | 🌐 Python | 📅 2023-07-25
* CoPS: Conditional Prompt Synthesis for Zero-Shot Anomaly Detection [\[CVPR 2026\]](https://arxiv.org/abs/2508.03447)[\[code\]](https://github.com/cqylunlun/CoPS) ⭐ 52 | 🐛 1 | 🌐 Python | 📅 2026-06-02
* FastRecon: Few-shot Industrial Anomaly Detection via Fast Feature Reconstruction [\[ICCV 2023\]](https://openaccess.thecvf.com/content/ICCV2023/papers/Fang_FastRecon_Few-shot_Industrial_Anomaly_Detection_via_Fast_Feature_Reconstruction_ICCV_2023_paper.pdf)[\[code\]](https://github.com/FzJun26th/FastRecon) ⭐ 48 | 🐛 6 | 🌐 Python | 📅 2023-10-25
* Small Object Few-shot Segmentation for Vision-based Industrial Inspection [\[2024\]](https://arxiv.org/abs/2407.21351)[\[code\]](https://github.com/zhangzilongc/SOFS) ⭐ 43 | 🐛 4 | 🌐 Python | 📅 2026-08-21
* Few-Shot Anomaly Detection via Category-Agnostic Registration Learning [\[2024\]](https://arxiv.org/abs/2406.08810)[\[code\]](https://github.com/Haoyan-Guan/CAReg) ⭐ 39 | 🐛 5 | 🌐 Python | 📅 2024-06-17
* FADE: Few-shot/zero-shot Anomaly Detection Engine using Large Vision-Language Model[\[BMVC 2024\]](https://arxiv.org/abs/2409.00556#)[\[code\]](https://github.com/BMVC-FADE/BMVC-FADE) ⭐ 35 | 🐛 4 | 🌐 Python | 📅 2024-11-22
* SOWA: Adapting Hierarchical Frozen Window Self-Attention to Visual-Language Models for Better Anomaly Detection [\[2024\]](https://arxiv.org/abs/2407.03634)[\[code\]](https://github.com/huzongxiang/sowa) ⭐ 30 | 🐛 1 | 🌐 Python | 📅 2024-12-12
* KAG-prompt: Kernel-Aware Graph Prompt Learning for Few-Shot Anomaly Detection [\[AAAI 2025\]](https://arxiv.org/abs/2412.17619)[\[code\]](https://github.com/CVL-hub/KAG-prompt) ⭐ 28 | 🐛 14 | 🌐 Python | 📅 2025-01-09
* Commonality in Few: Few-Shot Multimodal Anomaly Detection via Hypergraph-Enhanced Memory [\[AAAI 2026\]](https://arxiv.org/abs/2511.05966)[\[code\]](https://github.com/Sunny5250/CIF) ⭐ 26 | 🐛 2 | 🌐 Python | 📅 2025-11-13
* CLIP-FSAC++: Few-Shot Anomaly Classification with Anomaly Descriptor Based on CLIP [\[2024\]](https://arxiv.org/abs/2412.03829)[\[code\]](https://github.com/Jay-zzcoder/clip-fsac-pp) ⭐ 21 | 🐛 3 | 📅 2024-12-08
* FewSOME: One-Class Few Shot Anomaly Detection with Siamese Networks [\[CVPRW 2023\]](https://openaccess.thecvf.com/content/CVPR2023W/VAND/papers/Belton_FewSOME_One-Class_Few_Shot_Anomaly_Detection_With_Siamese_Networks_CVPRW_2023_paper.pdf)[\[code\]](https://github.com/niamhbelton/FewSOME) ⭐ 20 | 🐛 3 | 🌐 Python | 📅 2023-11-21
* Text-Guided Variational Image Generation for Industrial Anomaly Detection and Segmentation [\[CVPR 2024\]](https://arxiv.org/abs/2403.06247)[\[code\]](https://github.com/MingyuLee82/TGI_AD_v1) ⭐ 14 | 🐛 4 | 📅 2024-06-24
* CMDS-AD: Cross-Modal Dual-Stream Decoupling for Few-Shot Anomaly Detection [\[ECCV 2026\]](https://arxiv.org/abs/2606.20300)[\[code\]](https://github.com/Junhaocai27/CMDS-AD) ⭐ 10 | 🐛 0 | 🌐 Python | 📅 2026-09-01
* AnoPLe: Few-Shot Anomaly Detection via Bi-directional Prompt Learning with Only Normal Samples [\[2024\]](https://arxiv.org/abs/2408.13516)[\[code\]](https://github.com/YoojLee/AnoPLe) ⭐ 1 | 🐛 2 | 📅 2026-03-31
* Learning unsupervised metaformer for anomaly detection [\[ICCV 2021\]](https://openaccess.thecvf.com/content/ICCV2021/papers/Wu_Learning_Unsupervised_Metaformer_for_Anomaly_Detection_ICCV_2021_paper.pdf)
* Same same but differnet: Semi-supervised defect detection with normalizing flows [\[(Distribution)WACV 2021\]](http://arxiv.org/pdf/2008.12577)
* Towards total recall in industrial anomaly detection [\[(Memory bank)CVPR 2022\]](http://arxiv.org/pdf/2106.08265)
* A hierarchical transformation-discriminating generative model for few shot anomaly detection [\[ICCV 2021\]](http://arxiv.org/pdf/2104.14535)
* Anomaly detection of defect using energy of point pattern features within random finite set framework [\[2021\]](https://arxiv.org/pdf/2108.12159.pdf)
* Pushing the limits of fewshot anomaly detection in industry vision: Graphcore [\[ICLR 2023\]](https://openreview.net/pdf?id=xzmqxHdZAwO)
* Produce Once, Utilize Twice for Anomaly Detection [\[2023\]](https://arxiv.org/abs/2312.12913)
* COFT-AD: COntrastive Fine-Tuning for Few-Shot Anomaly Detection [\[TIP2024\]](http://arxiv.org/abs/2402.18998)
* Multimodal Industrial Anomaly Detection by Crossmodal Feature Mapping [\[CVPR 2024\]](https://arxiv.org/abs/2312.04521)
* Dual-path Frequency Discriminators for Few-shot Anomaly Detection [\[2024\]](https://arxiv.org/abs/2403.04151)
* Few-shot Online Anomaly Detection and Segmentation [\[2024\]](https://arxiv.org/abs/2403.18201)
* AnomalyDINO: Boosting Patch-based Few-shot Anomaly Detection with DINOv2 [\[2024\]](https://arxiv.org/abs/2405.14529)
* FOCT: Few-shot Industrial Anomaly Detection with Foreground-aware Online Conditional Transport [\[ACM MM 2024\]](https://dl.acm.org/doi/10.1145/3664647.3680771)
* UniVAD: A Training-free Unified Model for Few-shot Visual Anomaly Detection [\[2024\]](https://arxiv.org/abs/2412.03342)[\[code\]](https://uni-vad.github.io/#)
* One-for-All Few-Shot Anomaly Detection via Instance-Induced Prompt Learning [\[ICLR 2025\]](https://openreview.net/forum?id=Zzs3JwknAY)
* One-to-Normal: Anomaly Personalization for Few-shot Anomaly Detection [\[NeurIPS 2024\]](https://openreview.net/pdf?id=tIzW3l2uaN)
* Search is All You Need for Few-shot Anomaly Detection [\[2025\]](https://arxiv.org/abs/2504.11895)
* MetaCAN: Improving Generalizability of Few-shot Anomaly Detection with Meta-learning[\[CIKM 2025\]](https://dl.acm.org/doi/abs/10.1145/3746252.3761253?casa_token=qG2F2WLmNhEAAAAA:9hbnovS_MTgyUyBH7GL8H3cRy68xr9HIkKxno6njzVLKtmG84gfDSFDmTgVrtClsBAQtn6xMa27tZsQ)
* UniADC: A Unified Framework for Anomaly Detection and Classification [\[2025\]](https://arxiv.org/abs/2511.06644)
* Towards Fine-Grained Vision-Language Alignment for Few-Shot Anomaly Detection [\[2025\]](https://arxiv.org/abs/2510.26464)
* Dual Distillation for Few-Shot Anomaly Detection [\[ICLR 2026\]](https://openreview.net/forum?id=tRO6G20Qba)
* FastRef:Fast Prototype Refinement for Few-Shot Industrial Anomaly Detection [\[CVPR 2026\]](https://arxiv.org/abs/2506.21398)
* ADSeeker: A Knowledge-Grounded Reasoning Framework for Industry Anomaly Detection and Reasoning [\[CVPR 2026\]](https://arxiv.org/abs/2508.03088)

## 3.2 Noisy AD

* SoftPatch: Unsupervised Anomaly Detection with Noisy Data [\[NeurIPS 2022\]](https://openreview.net/pdf?id=pIYYJflkhZ)[\[code\]](https://github.com/TencentYoutuResearch/AnomalyDetection-SoftPatch) ⭐ 96 | 🐛 5 | 🌐 Python | 📅 2024-07-13
* SoftPatch+: Fully Unsupervised Anomaly Classification and Segmentation [\[PR 2025\]](https://arxiv.org/abs/2412.20870)[\[code\]](https://github.com/TencentYoutuResearch/AnomalyDetection-SoftPatch) ⭐ 96 | 🐛 5 | 🌐 Python | 📅 2024-07-13
* Deep one-class classification via interpolated gaussian descriptor [\[AAAI 2022 oral\]](https://arxiv.org/pdf/2101.10043.pdf)[\[code\]](https://github.com/tianyu0207/IGD) ⭐ 69 | 🐛 5 | 🌐 Python | 📅 2025-10-29
* FUN-AD: Fully Unsupervised Learning for Anomaly Detection with Noisy Training Data [\[WACV 2025\]](https://arxiv.org/abs/2411.16110)[\[code\]](https://github.com/HY-Vision-Lab/FUNAD) ⭐ 33 | 🐛 1 | 🌐 Python | 📅 2025-03-11
* Memory-Distilled Selection for Noise-Robust Anomaly Detection [\[ICML 2026\]](https://arxiv.org/abs/2605.26676)[\[code\]](https://github.com/SirojbekSafarov/MeDS) ⭐ 18 | 🐛 0 | 🌐 Python | 📅 2026-06-17
* Inter-Realization Channels: Unsupervised Anomaly Detection Beyond One-Class Classification [\[ICCV 2023\]](https://openaccess.thecvf.com/content/ICCV2023/papers/McIntosh_Inter-Realization_Channels_Unsupervised_Anomaly_Detection_Beyond_One-Class_Classification_ICCV_2023_paper.pdf)[\[code\]](https://github.com/DeclanMcIntosh/InReaCh) ⭐ 15 | 🐛 0 | 🌐 Python | 📅 2026-01-12
* Trustmae: A noise-resilient defect classification framework using memory-augmented auto-encoders with trust regions [\[WACV 2021\]](http://arxiv.org/pdf/2012.14629)
* Self-Supervise, Refine, Repeat: Improving Unsupervised Anomaly Detection [\[TMLR 2021\]](https://arxiv.org/pdf/2106.06115.pdf)
* Data refinement for fully unsupervised visual inspection using pre-trained networks [\[2022\]](https://arxiv.org/pdf/2202.12759.pdf)
* Latent Outlier Exposure for Anomaly Detection with Contaminated Data [\[ICML 2022\]](https://arxiv.org/pdf/2202.08088.pdf)
* M3DM-NR: RGB-3D Noisy-Resistant Industrial Anomaly Detection via Multimodal Denoising [\[2024\]](https://arxiv.org/abs/2406.02263)
* Meta Learning-Driven Iterative Refinement for Robust Anomaly Detection in Industrial Inspection [\[ECCVW 2024\]](https://arxiv.org/abs/2503.01569)
* Towards Real Unsupervised Anomaly Detection Via Confident Meta-Learning [\[ICCV 2025\]](https://arxiv.org/abs/2508.02293)

## 3.3 Anomaly Synthesis [\[awesome-anomaly-synthesis\]](https://github.com/M-3LAB/awesome-anomaly-synthesis) ⭐ 80 | 🐛 0 | 📅 2025-11-18

* RealNet: A Feature Selection Network with Realistic Synthetic Anomaly for Anomaly Detection [\[CVPR 2024\]](https://arxiv.org/abs/2403.05897)[\[code\]](https://github.com/cnulab/RealNet) ⭐ 426 | 🐛 79 | 🌐 Python | 📅 2025-02-12
* A Unified Anomaly Synthesis Strategy with Gradient Ascent for Industrial Anomaly Detection and Localization [\[ECCV 2024\]](https://arxiv.org/abs/2407.09359)[\[code\]](https://github.com/cqylunlun/GLASS) ⭐ 392 | 🐛 5 | 🌐 Python | 📅 2026-03-30
* AnomalyDiffusion: Few-Shot Anomaly Image Generation with Diffusion Model [\[AAAI 2024\]](https://ojs.aaai.org/index.php/AAAI/article/view/28696)[\[code\]](https://github.com/sjtuplayer/anomalydiffusion) ⭐ 328 | 🐛 52 | 🌐 Jupyter Notebook | 📅 2024-08-11
* Anomaly Anything: Promptable Unseen Visual Anomaly Generation [\[CVPR 2025\]](https://arxiv.org/abs/2406.01078)[\[code\]](https://github.com/EPFL-IMOS/AnomalyAny) ⭐ 301 | 🐛 14 | 🌐 Jupyter Notebook | 📅 2025-06-30
* Draem-a discriminatively trained reconstruction embedding for surface anomaly detection [\[(Reconstruction AE)ICCV 2021\]](http://arxiv.org/pdf/2108.07610)[\[code\]](https://github.com/vitjanz/draem) ⭐ 291 | 🐛 18 | 🌐 Python | 📅 2023-01-02
* Cutpaste: Self-supervised learning for anomaly detection and localization [\[(OCC)ICCV 2021\]](http://arxiv.org/pdf/2104.04015)[\[unofficial code\]](https://github.com/Runinho/pytorch-cutpaste) ⭐ 261 | 🐛 21 | 🌐 Python | 📅 2024-05-07
* MemSeg: A semi-supervised method for image surface defect detection using differences and commonalities [\[(OCC)2022\]](https://arxiv.org/pdf/2205.00908.pdf)[\[unofficial code\]](https://github.com/TooTouch/MemSeg) ⭐ 206 | 🐛 8 | 🌐 Jupyter Notebook | 📅 2024-06-24
* SuperSimpleNet: Unifying Unsupervised and Supervised Learning for Fast and Reliable Surface Defect Detection [\[ICPR 2024\]](https://arxiv.org/abs/2408.03143)[\[JIMS 2025\]](https://link.springer.com/article/10.1007/s10845-025-02680-8)[\[code\]](https://github.com/blaz-r/SuperSimpleNet/tree/main) ⭐ 176 | 🐛 1 | 🌐 Python | 📅 2025-10-16
* Few-shot defect image generation via defect-aware feature manipulation [\[AAAI 2023\]](https://arxiv.org/abs/2303.02389)[\[code\]](https://github.com/Ldhlwh/DFMGAN) ⭐ 158 | 🐛 5 | 🌐 Python | 📅 2024-12-19
* Anomagic: Crossmodal Prompt-driven Zero-shot Anomaly Generation [\[AAAI 2026\]](https://arxiv.org/abs/2511.10020)[\[code\]](https://github.com/yuxin-jiang/Anomagic) ⭐ 158 | 🐛 0 | 🌐 Python | 📅 2026-04-25
* A Novel Approach to Industrial Defect Generation through Blended Latent Diffusion Model with Online Adaptation [\[2024\]](https://arxiv.org/abs/2402.19330)[\[code\]](https://github.com/GrandpaXun242/AdaBLDM) ⭐ 92 | 🐛 5 | 🌐 Python | 📅 2024-04-25
* DeSTSeg: Segmentation Guided Denoising Student-Teacher for Anomaly Detection [\[CVPR 2023\]](https://arxiv.org/abs/2211.11317)[\[code\]](https://github.com/apple/ml-destseg) ⭐ 85 | 🐛 0 | 🌐 Python | 📅 2024-05-16
* Few-Shot Anomaly-Driven Generation for Anomaly Classification and Segmentation [\[ECCV 2024\]](https://csgaobb.github.io/Pub_files/ECCV2024_AnoGen_CR_0730_Mobile.pdf)[\[code\]](https://github.com/gaobb/AnoGen) ⭐ 82 | 🐛 7 | 🌐 Jupyter Notebook | 📅 2025-09-08
* Natural Synthetic Anomalies for Self-supervised Anomaly Detection and Localization [\[ECCV 2022\]](https://arxiv.org/pdf/2109.15222.pdf)[\[code\]](https://github.com/hmsch/natural-synthetic-anomalies) ⭐ 73 | 🐛 1 | 🌐 Jupyter Notebook | 📅 2023-04-20
* One-to-More: High-Fidelity Training-Free Anomaly Generation with Attention Control [\[CVPR 2026\]](https://arxiv.org/abs/2603.18093)[\[code\]](https://github.com/echrao/O2MAG) ⭐ 67 | 🐛 3 | 🌐 Jupyter Notebook | 📅 2026-03-26
* DictAS: A Framework for Class-Generalizable Few-Shot Anomaly Segmentation via Dictionary Lookup [\[ICCV 2025\]](https://www.arxiv.org/abs/2508.13560)[\[code\]](https://github.com/xiaozhen228/DictAS) ⭐ 58 | 🐛 1 | 🌐 Python | 📅 2025-12-13
* DSR: A dual subspace re-projection network for surface anomaly detection [\[ECCV 2022\]](https://arxiv.org/pdf/2208.01521.pdf)[\[code\]](https://github.com/VitjanZ/DSR_anomaly_detection) ⭐ 56 | 🐛 4 | 🌐 Python | 📅 2022-07-18
* Progressive Boundary Guided Anomaly Synthesis for Industrial Anomaly Detection [\[TCSVT 2024\]](https://ieeexplore.ieee.org/document/10716437)[\[code\]](https://github.com/cqylunlun/PBAS) ⭐ 36 | 🐛 1 | 🌐 Python | 📅 2026-03-30
* UniADC: A Unified Framework for Anomaly Detection and Classification [\[2025\]](https://arxiv.org/abs/2511.06644)[\[code is comming\]](https://github.com/cnulab/UniADC) ⭐ 33 | 🐛 2 | 🌐 Python | 📅 2026-06-30
* AnomalyXFusion: Multi-modal Anomaly Synthesis with Diffusion [\[2024\]](https://arxiv.org/abs/2404.19444)[\[data\]](https://github.com/hujiecpp/MVTec-Caption) ⭐ 22 | 🐛 3 | 🌐 Python | 📅 2024-10-30
* AnoStyler: Text-Driven Localized Anomaly Generation via Lightweight Style Transfer [\[AAAI 2026\]](https://arxiv.org/abs/2511.06687)[\[code\]](https://github.com/yulimso/AnoStyler) ⭐ 22 | 🐛 7 | 🌐 Python | 📅 2025-11-14
* Bounding Box-Guided Diffusion for Synthesizing Industrial Images and Segmentation Map [\[CVPRW 2025\]](https://arxiv.org/abs/2505.03623)[\[code\]](https://github.com/covisionlab/diffusion_labeling) ⭐ 13 | 🐛 0 | 🌐 Python | 📅 2025-08-28
* CHIMERA: Controllable High-quality Image-Mask Extraction for Reliable Diffusion-Based Anomaly Synthesis [\[AAAI 2026\]](https://ojs.aaai.org/index.php/AAAI/article/view/37511)[\[code\]](https://github.com/cvlab-kaist/CHIMERA) ⭐ 11 | 🐛 2 | 📅 2025-11-16
* Quality-Aware Language-Conditioned Local Auto-Regressive Anomaly Synthesis and Detection [\[AAAI 2026\]](https://arxiv.org/abs/2508.03539)[\[code\]](https://github.com/neymarql/QARAD) ⭐ 8 | 🐛 0 | 🌐 Python | 📅 2025-11-12
* DeCo: Zero-Shot Industrial Anomaly Generation through Decoupling and Recoupling [\[ECCV 2026\]](https://arxiv.org/abs/2608.07904)[\[code\]](https://github.com/HUST-SLOW/DeCo) ⭐ 5 | 🐛 2 | 📅 2026-06-25
* SynSpill: Improved Industrial Spill Detection With Synthetic Data [\[ICCVW 2025 oral\]](https://arxiv.org/abs/2508.10171)[\[code\]](https://github.com/eternal-f1ame/SynSpill) ⭐ 4 | 🐛 0 | 🌐 TypeScript | 📅 2025-09-10[\[homepage\]](https://synspill.vercel.app/)
* UniScale: Arbitrary-Scale Industrial Anomaly Generation [\[ECCV 2026\]](https://arxiv.org/abs/2608.07864)[\[code\]](https://github.com/HUST-SLOW/UniScale) ⭐ 4 | 🐛 2 | 📅 2026-06-25
* ASBench: Image Anomalies Synthesis Benchmark for Anomaly Detection [\[TAI 2026\]](https://arxiv.org/abs/2510.07927)[\[code\]](https://github.com/M-3LAB/ASBench) ⭐ 2 | 🐛 0 | 🌐 Jupyter Notebook | 📅 2026-05-11
* ISP-AD: a large-scale real-world dataset for advancing industrial anomaly detection with synthetic and real defects [\[JIMS 2026\]](https://link.springer.com/article/10.1007/s10845-025-02778-z)[\[code\]](https://github.com/p4ulk/isp-ad) ⭐ 0 | 🐛 0 | 🌐 Python | 📅 2026-07-10[\[data\]](https://zenodo.org/records/14911042)
* A High-Efficiency Fully Convolutional Networks for Pixel-Wise Surface Defect Detection [\[IEEE Access 2019\]](https://ieeexplore.ieee.org/abstract/document/8624360)
* Multistage GAN for fabric defect detection [\[2019\]](https://pubmed.ncbi.nlm.nih.gov/31870985/)
* Gan-based defect synthesis for anomaly detection in fabrics [\[2020\]](https://www.lfb.rwth-aachen.de/bibtexupload/pdf/RIP20c.pdf)
* Defect image sample generation with GAN for improving defect recognition [\[2020\]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=\&arnumber=9000806)
* Defective samples simulation through neural style transfer for automatic surface defect segment [\[2020\]](http://arxiv.org/pdf/1910.03334)
* A simulation-based few samples learning method for surface defect segmentation [\[2020\]](https://www.sciencedirect.com/science/article/pii/S0925231220310791/pdfft?md5=f3f72bc8687c8f9968d4a2a1bd3ea17e\&pid=1-s2.0-S0925231220310791-main.pdf)
* Synthetic data augmentation for surface defect detection and classification using deep learning [\[2020\]](https://link.springer.com/article/10.1007/s10845-020-01710-x)
* Defect Transfer GAN: Diverse Defect Synthesis for Data Augmentation [\[BMVC 2022\]](https://openreview.net/pdf?id=2hMEdc35xZ6)
* Defect-GAN: High-fidelity defect synthesis for automated defect inspection [\[2021\]](https://dr.ntu.edu.sg/bitstream/10356/146285/2/WACV_2021_Defect_GAN__Camera_Ready_.pdf)
* EID-GAN: Generative Adversarial Nets for Extremely Imbalanced Data Augmentation[\[TII 2022\]](https://ieeexplore.ieee.org/document/9795891)
* Multilevel Saliency-Guided Self-Supervised Learning for Image Anomaly Detection [\[2023\]](http://arxiv.org/pdf/2311.18332v1)
* Dual-path Frequency Discriminators for Few-shot Anomaly Detection [\[2024\]](https://arxiv.org/abs/2403.04151)
* A Comprehensive Augmentation Framework for Anomaly Detection [\[AAAI 2024\]](https://ojs.aaai.org/index.php/AAAI/article/view/28720)
* CAGEN: Controllable Anomaly Generator using Diffusion Model [\[ICASSP 2024\]](https://ieeexplore.ieee.org/document/10447663)
* SLSG: Industrial Image Anomaly Detection with Improved Feature Embeddings and One-Class Classification [\[PR 2024\]](https://www.sciencedirect.com/science/article/pii/S0031320324006137)
* Dual-Modeling Decouple Distillation for Unsupervised Anomaly Detection [\[ACM MM 2024\]](https://arxiv.org/abs/2408.03888)
* AnomalyControl: Learning Cross-modal Semantic Features for Controllable Anomaly Synthesis [\[2024\]](https://arxiv.org/abs/2412.06510)
* Component-aware Unsupervised Logical Anomaly Generation for Industrial Anomaly Detection [\[2025\]](https://arxiv.org/abs/2502.11712)
* "Stones from Other Hills can Polish Jade": Zero-shot Anomaly Image Synthesis via Cross-domain Anomaly Injection [\[2025\]](https://arxiv.org/abs/2501.15211)
* Fully-Synthetic Training for Visual Quality Inspection in Automotive Production [\[CIRP 2025\]](https://arxiv.org/abs/2503.09354)
* Enhanced Fabric Defect Detection with Feature Contrast Interference Suppression [\[TIM 2025\]](https://ieeexplore.ieee.org/abstract/document/10937904)
* Open-Set Fabric Defect Detection With Defect Generation and Transfer [\[TIM 2025\]](https://ieeexplore.ieee.org/abstract/document/10909357)
* Enhancing Glass Defect Detection with Diffusion Models: Addressing Imbalanced Datasets in Manufacturing Quality Control [\[2025\]](https://arxiv.org/abs/2505.03134)
* Photovoltaic Defect Image Generator with Boundary Alignment Smoothing Constraint for Domain Shift Mitigation [\[2025\]](https://arxiv.org/abs/2505.06117)
* Anodapter: A Unified Framework for Generating Aligned Anomaly Images and Masks Using Diffusion Models[\[2025\]](https://ieeexplore.ieee.org/document/11000123)
* AnomalyHybrid: A Domain-agnostic Generative Framework for General Anomaly Detection [\[CVPR 2025 SyntaGen Workshop\]](https://openaccess.thecvf.com/content/CVPR2025W/SyntaGen/papers/Zhao_AnomalyHybrid_A_Domain-agnostic_Generative_Framework_for_General_Anomaly_Detection_CVPRW_2025_paper.pdf)
* AnomalyControl: Highly-Aligned Anomalous Image Generation with Controlled Diffusion Model [\[ACM MM 2025\]](https://dl.acm.org/doi/abs/10.1145/3746027.3755274)
* AnomalyPainter: Vision-Language-Diffusion Synergy for Zero-Shot Realistic and Diverse Industrial Anomaly Synthesis [\[AAAI 2026\]](https://arxiv.org/abs/2503.07253)
* CADiff: Context-Aware Diffusion for Controllable Anomaly Generation in Anomaly Detection [\[AAAI 2026\]](https://ojs.aaai.org/index.php/AAAI/article/view/37917)
* Anomaly-Preference Image Generation [\[ICML 2026\]](https://arxiv.org/abs/2605.02439)

## 3.4 RGBD AD

* Multimodal Industrial Anomaly Detection via Hybrid Fusion [\[CVPR 2023\]](https://arxiv.org/abs/2303.00601)[\[code\]](https://github.com/nomewang/M3DM) ⭐ 215 | 🐛 22 | 🌐 Python | 📅 2023-09-08
* Back to the feature: classical 3d features are (almost) all you need for 3D anomaly detection [\[2022\]](https://arxiv.org/pdf/2203.05550.pdf)[\[code\]](https://github.com/eliahuhorwitz/3D-ADS) ⭐ 143 | 🐛 9 | 🌐 Python | 📅 2022-11-28
* Complementary Pseudo Multimodal Feature for Point Cloud Anomaly Detection [\[2023\]](https://arxiv.org/abs/2303.13194)[\[code\]](https://github.com/caoyunkang/CPMF) ⭐ 85 | 🐛 2 | 🌐 Python | 📅 2025-03-04
* Towards Scalable 3D Anomaly Detection and Localization: A Benchmark via 3D Anomaly Synthesis and A Self-Supervised Learning Network [\[CVPR 2024\]](https://arxiv.org/abs/2311.14897)[\[code\]](https://github.com/Chopper-233/Anomaly-ShapeNet) ⭐ 57 | 🐛 8 | 📅 2025-10-17
* Cheating Depth: Enhancing 3D Surface Anomaly Detection via Depth Simulation [\[WACV 2024\]](https://openaccess.thecvf.com/content/WACV2024/papers/Zavrtanik_Cheating_Depth_Enhancing_3D_Surface_Anomaly_Detection_via_Depth_Simulation_WACV_2024_paper.pdf)[\[code\]](https://github.com/VitjanZ/3DSR) ⭐ 43 | 🐛 6 | 🌐 Python | 📅 2024-02-13
* Cross-Modal Distillation in Industrial Anomaly Detection: Exploring Efficient Multi-Modal IAD [\[2024\]](https://arxiv.org/abs/2405.13571)[\[code\]](https://github.com/evenrose/CMDIAD) ⭐ 26 | 🐛 0 | 🌐 Python | 📅 2026-03-17
* Commonality in Few: Few-Shot Multimodal Anomaly Detection via Hypergraph-Enhanced Memory [\[AAAI 2026\]](https://arxiv.org/abs/2511.05966)[\[code\]](https://github.com/Sunny5250/CIF) ⭐ 26 | 🐛 2 | 🌐 Python | 📅 2025-11-13
* G2SF: Geometry-Guided Score Fusion for Multimodal Industrial Anomaly Detection[\[ICCV 2025\]](https://openaccess.thecvf.com/content/ICCV2025/papers/Tao_G2SF_Geometry-Guided_Score_Fusion_for_Multimodal_Industrial_Anomaly_Detection_ICCV_2025_paper.pdf)[\[code\]](https://github.com/ctaoaa/G2SF) ⭐ 17 | 🐛 2 | 🌐 Python | 📅 2025-12-14
* Image-Pointcloud Fusion based Anomaly Detection using PD-REAL Dataset [\[2023\]](https://arxiv.org/abs/2311.04095)[\[data\]](https://github.com/Andy-cs008/PD-REAL) ⭐ 9 | 🐛 1 | 📅 2026-03-09
* Anomaly detection in 3d point clouds using deep geometric descriptors [\[WACV 2022\]](https://arxiv.org/pdf/2202.11660.pdf)
* Anomaly Detection Requires Better Representations [\[2022\]](https://arxiv.org/pdf/2210.10773.pdf)
* Asymmetric Student-Teacher Networks for Industrial Anomaly Detection [\[WACV 2022\]](https://arxiv.org/pdf/2210.07829.pdf)
* Shape-Guided Dual-Memory Learning for 3D Anomaly Detection [\[ICML 2023\]](https://openreview.net/forum?id=IkSGn9fcPz)
* EasyNet: An Easy Network for 3D Industrial Anomaly Detection [\[ACM MM 2023\]](https://arxiv.org/abs/2307.13925)
* Self-supervised Feature Adaptation for 3D Industrial Anomaly Detection [\[2024\]](https://arxiv.org/abs/2401.03145)
* Incremental Template Neighborhood Matching for 3D anomaly detection [\[Neurocomputing 2024\]](https://www.sciencedirect.com/science/article/abs/pii/S0925231224002546)
* Keep DRÆMing: Discriminative 3D anomaly detection through anomaly simulation [\[PRL 2024\]](https://www.sciencedirect.com/science/article/pii/S0167865524000862)
* Rethinking Reverse Distillation for Multi-Modal Anomaly Detection [\[AAAI 2024\]](https://ojs.aaai.org/index.php/AAAI/article/view/28687)
* Multimodal Industrial Anomaly Detection by Crossmodal Feature Mapping [\[CVPR 2024\]](https://arxiv.org/abs/2312.04521)
* M3DM-NR: RGB-3D Noisy-Resistant Industrial Anomaly Detection via Multimodal Denoising [\[2024\]](https://arxiv.org/abs/2406.02263)
* Learning Diffusion Models for Multi-View Anomaly Detection [\[ECCV 2024\]](https://eccv2024.ecva.net/virtual/2024/poster/1911)
* Towards Zero-shot 3D Anomaly Localization [\[WACV 2025\]](https://arxiv.org/abs/2412.04304)
* Revisiting Multimodal Fusion for 3D Anomaly Detection from an Architectural Perspective [\[AAAI 2025\]](https://arxiv.org/abs/2412.17297)
* Mentor3AD: Feature Reconstruction-based 3D Anomaly Detection via Multi-modality Mentor Learning [\[2025\]](https://arxiv.org/abs/2505.21420)
* AnomalyHybrid: A Domain-agnostic Generative Framework for General Anomaly Detection [\[CVPR 2025 SyntaGen Workshop\]](https://openaccess.thecvf.com/content/CVPR2025W/SyntaGen/papers/Zhao_AnomalyHybrid_A_Domain-agnostic_Generative_Framework_for_General_Anomaly_Detection_CVPRW_2025_paper.pdf)
* Unsupervised Multi-View Visual Anomaly Detection via Progressive Homography-Guided Alignment [\[AAAI 2026\]](https://arxiv.org/abs/2511.18766)
* RPE-PAD: Relative Pose Estimation for Pose-agnostic Anomaly Detection [\[AAAI 2026\]](https://ojs.aaai.org/index.php/AAAI/article/view/38304)

## 3.5 3D AD

* MuSc-V2: Zero-Shot Multimodal Industrial Anomaly Classification and Segmentation with Mutual Scoring of Unlabeled Samples [\[TPAMI 2026\]](https://arxiv.org/abs/2511.10047)[\[code\]](https://github.com/HUST-SLOW/MuSc-V2) ⭐ 339 | 🐛 2 | 🌐 Python | 📅 2026-05-14
* Real3D-AD: A Dataset of Point Cloud Anomaly Detection [\[NeurIPS 2023\]](https://arxiv.org/abs/2309.13226)[\[code\]](https://github.com/M-3LAB/Real3D-AD) ⭐ 162 | 🐛 0 | 🌐 Python | 📅 2024-03-13
* Towards Zero-shot Point Cloud Anomaly Detection: A Multi-View Projection Framework [\[2024\]](https://arxiv.org/abs/2409.13162)[\[code\]](https://github.com/hustCYQ/MVP-PCLIP) ⭐ 117 | 🐛 11 | 🌐 Python | 📅 2025-12-17
* Multi-Sensor Object Anomaly Detection: Unifying Appearance, Geometry, and Internal Properties [\[CVPR 2025\]](https://zzzbbbzzz.github.io/MulSen_AD/index.html)[\[code\]](https://github.com/ZZZBBBZZZ/MulSen-AD) ⭐ 94 | 🐛 4 | 🌐 Python | 📅 2025-03-20
* PointAD: Comprehending 3D Anomalies from Points and Pixels for Zero-shot 3D Anomaly Detection [\[NeurIPS 2024\]](https://arxiv.org/abs/2410.00320)[\[code\]](https://github.com/zqhang/PointAD) ⭐ 89 | 🐛 23 | 🌐 Python | 📅 2025-05-29
* Complementary Pseudo Multimodal Feature for Point Cloud Anomaly Detection [\[PR 2024\]](https://www.sciencedirect.com/science/article/abs/pii/S0031320324005120) [\[code\]](https://github.com/caoyunkang/CPMF) ⭐ 85 | 🐛 2 | 🌐 Python | 📅 2025-03-04
* Towards Scalable 3D Anomaly Detection and Localization: A Benchmark via 3D Anomaly Synthesis and A Self-Supervised Learning Network [\[CVPR 2024\]](https://arxiv.org/abs/2311.14897)[\[code\]](https://github.com/Chopper-233/Anomaly-ShapeNet) ⭐ 57 | 🐛 8 | 📅 2025-10-17
* Multi-View Reconstruction with Global Context for 3D Anomaly Detection [\[IEEE SMC 2025\]](https://arxiv.org/abs/2507.21555)[\[code\]](https://github.com/hustSYH/MVR) ⭐ 26 | 🐛 2 | 🌐 Python | 📅 2025-08-01
* Back to Point: Exploring Point-Language Models for Zero-Shot 3D Anomaly Detection [\[CVPR 2026\]](https://arxiv.org/abs/2603.21511)[\[code\]](https://github.com/wistful-8029/BTP-3DAD) ⭐ 23 | 🐛 4 | 🌐 Python | 📅 2026-06-22
* Bridging 3D Anomaly Localization and Repair via High-Quality Continuous Geometric Representation [\[ICCV 2025\]](https://arxiv.org/abs/2505.24431)[\[code\]](https://github.com/ZZZBBBZZZ/PASDF) ⭐ 18 | 🐛 2 | 🌐 Python | 📅 2025-12-17
* Look Inside for More: Internal Spatial Modality Perception for 3D Anomaly Detection [\[AAAI 2025\]](https://arxiv.org/abs/2412.13461)[\[code\]](https://github.com/M-3LAB/Look-Inside-for-More) ⭐ 16 | 🐛 0 | 🌐 Python | 📅 2025-10-05
* PIAD: Pose and Illumination agnostic Anomaly Detection [\[CVPR 2025\]](https://openaccess.thecvf.com/content/CVPR2025/papers/Yang_PIAD_Pose_and_Illumination_agnostic_Anomaly_Detection_CVPR_2025_paper.pdf) [\[code\]](https://github.com/Kaichen-Yang/piad_baseline) ⭐ 14 | 🐛 1 | 🌐 Python | 📅 2025-11-11[\[data\]](kaichen-yang.github.io/piad/)
* CASL: Curvature-Augmented Self-supervised Learning for 3D Anomaly Detection [\[AAAI 2026\]](https://arxiv.org/abs/2511.12909)[\[code\]](https://github.com/zyh16143998882/CASL) ⭐ 12 | 🐛 2 | 🌐 Python | 📅 2025-12-15
* Towards High-resolution 3D Anomaly Detection via Group-Level Feature Contrastive Learning [\[ACM MM 2024\]](https://arxiv.org/abs/2408.04604)[\[code\]](https://github.com/M-3LAB/Group3AD) ⭐ 10 | 🐛 2 | 📅 2024-08-08
* Point Cloud Segmentation of Integrated Circuits Package Substrates Surface Defects Using Causal Inference: Dataset Construction and Methodology [\[2025\]](https://arxiv.org/abs/2511.05853)[\[code is comming\]](https://github.com/Bingyang0410/CPS3D-Seg) ⭐ 7 | 🐛 1 | 📅 2025-11-08
* Examining the Source of Defects from a Mechanical Perspective for 3D Anomaly Detection [\[2025\]](https://arxiv.org/abs/2505.05901v2)[\[code\]](https://github.com/hzzzzzhappy/MC4AD) ⭐ 5 | 🐛 1 | 📅 2025-07-31
* 3D-ADAM: A Dataset for 3D Anomaly Detection in Advanced Manufacturing [\[2025\]](https://arxiv.org/abs/2507.07838)[\[data\]](https://huggingface.co/datasets/pmchard/3D-ADAM)[\[code\]](https://github.com/PaulMcHard/3D-ADAMBench) ⭐ 5 | 🐛 1 | 🌐 Python | 📅 2025-05-25
* PointCore: Efficient Unsupervised Point Cloud Anomaly Detector Using Local-Global Features [\[2024\]](https://arxiv.org/abs/2403.01804)
* R3D-AD: Reconstruction via Diffusion for 3D Anomaly Detection [\[ECCV 2024\]](https://arxiv.org/abs/2407.10862)[\[homepage\]](https://zhouzheyuan.github.io/r3d-ad)
* Exploiting Point-Language Models with Dual-Prompts for 3D Anomaly Detection [\[2025\]](https://arxiv.org/abs/2502.11307)
* Fence Theorem: Preprocessing is Dual-Objective Semantic Structure Isolator in 3D Anomaly Detection [\[2025\]](https://arxiv.org/abs/2503.01100)
* Odd-One-Out: Anomaly Detection by Comparing with Neighbors [\[CVPR 2025\]](https://arxiv.org/abs/2406.20099)
* PO3AD: Predicting Point Offsets toward Better 3D Point Cloud Anomaly Detection [\[CVPR 2025\]](https://arxiv.org/abs/2412.12617)
* MC3D-AD: A Unified Geometry-aware Reconstruction Model for Multi-category 3D Anomaly Detection [\[IJCAI 2025\]](https://arxiv.org/abs/2505.01969)
* Bridging 3D Anomaly Localization and Repair via High-Quality Continuous Geometric Representation [\[2025\]](https://arxiv.org/pdf/2505.24431v1)
* Taming Anomalies with Down-Up Sampling Networks: Group Center Preserving Reconstruction for 3D Anomaly Detection [\[ACM MM 2025\]](https://arxiv.org/abs/2507.03903v1)
* 3DKeyAD: High-Resolution 3D Point Cloud Anomaly Detection via Keypoint-Guided Point Clustering [\[2025\]](https://arxiv.org/abs/2507.13110)
* Registration is a Powerful Rotation-Invariance Learner for 3D Anomaly Detection [\[NeurIPS 2025\]](https://arxiv.org/abs/2510.16865)
* IEC3D-AD: A 3D Dataset of Industrial Equipment Components for Unsupervised Point Cloud Anomaly Detection [\[2025\]](https://arxiv.org/abs/2511.03267)
* Towards High-Resolution 3D Anomaly Detection: A Scalable Dataset and Real-Time Framework for Subtle Industrial Defects [\[AAAI 2026 oral\]](https://arxiv.org/abs/2507.07435)[\[code\]](https://hustcyq.github.io/MiniShift-Simple3D/)
* Robust Modality-Incomplete Anomaly Detection: A Modality-Instructive Framework with Benchmark [\[ACM MM 2025\]](https://dl.acm.org/doi/abs/10.1145/3746027.3754766)
* PIRN: Prototypical-based Intra-modal Reconstruction with Normality Communication for Multi-modal Anomaly Detection [\[ICLR 2026\]](https://openreview.net/forum?id=7L7kmHHfgf)
* SCoNE: Spherical Consistent Neighborhoods Ensemble for Effective and Efficient Multi-View Anomaly Detection [\[AAAI 2026\]](https://arxiv.org/abs/2512.05540)
* Wavelet-Driven 3D Anomaly Detection under Pose-Agnostic and Sparse-View [\[CVPR 2026\]](https://cvpr.thecvf.com/virtual/2026/poster/38024)

<!-- + DropGaussian: Structural Regularization for Sparse-view Gaussian Splatting [[CVPR 2025]](https://openaccess.thecvf.com/content/CVPR2025/papers/Park_DropGaussian_Structural_Regularization_for_Sparse-view_Gaussian_Splatting_CVPR_2025_paper.pdf)[[code]](https://github.com/DCVL-3D/DropGaussian_release) -->

* Towards Active Real-to-Twin Inspection: A New Paradigm for Zero-Shot Anomaly Detection[\[IEEE-CYBER 2026\]](https://arxiv.org/abs/2605.25407)
* PADFormer: Pose-agnostic Anomaly Detection from Sparse View Images [\[ECCV 2026 Oral\]](https://arxiv.org/abs/2608.04210)

## 3.6 Continual AD

* Unsupervised Continual Anomaly Detection with Contrastively-learned Prompt [\[AAAI 2024\]](https://ojs.aaai.org/index.php/AAAI/article/view/28153)[\[code\]](https://github.com/shirowalker/UCAD) ⭐ 115 | 🐛 0 | 🌐 Python | 📅 2024-08-06
* ReplayCAD: Generative Diffusion Replay for Continual Anomaly Detection [\[IJCAI 2025\]](https://arxiv.org/abs/2505.06603)[\[code\]](https://github.com/HULEI7/ReplayCAD) ⭐ 59 | 🐛 0 | 🌐 Jupyter Notebook | 📅 2026-08-16
* An Incremental Unified Framework for Small Defect Inspection [\[ECCV2024\]](https://arxiv.org/abs/2312.08917v2)[\[code\]](https://github.com/jqtangust/IUF) ⭐ 49 | 🐛 2 | 🌐 Python | 📅 2025-02-17
* GS-CLIP: Zero-shot 3D Anomaly Detection by Geometry-Aware Prompt and Synergistic View Representation Learning [\[CVPR 2026\]](https://arxiv.org/abs/2602.19206)[\[code\]](https://github.com/zhushengxinyue/GS-CLIP) ⭐ 25 | 🐛 5 | 🌐 Python | 📅 2026-03-07
* C3D-AD: Toward Continual 3D Anomaly Detection via Kernel Attention with Learnable Advisor [\[2025\]](https://arxiv.org/abs/2508.01311)[\[code\]](https://github.com/hzzzzzhappy/CL3AD) ⭐ 8 | 🐛 0 | 🌐 Python | 📅 2026-02-24
* Hierarchical Point-Patch Fusion with Adaptive Patch Codebook for 3D Shape Anomaly Detection [\[CVPR 2026\]](https://arxiv.org/abs/2604.03972)[\[code\]](https://github.com/alexandor91/Shape-Anomaly-Codebook) ⭐ 4 | 🐛 2 | 🌐 Python | 📅 2026-05-19
* Rethinking Continual Anomaly Detection on the Edge: Benchmarking Under Realistic Industrial Conditions [\[ECCV 2026\]](https://arxiv.org/abs/2605.24251)[\[code\]](https://github.com/Continue-Edge-AI-Lab/Rethinking-Continual-AD) ⭐ 3 | 🐛 0 | 🌐 Python | 📅 2026-06-29
* Towards Total Online Unsupervised Anomaly Detection and Localization in Industrial Vision [\[2023\]](https://arxiv.org/abs/2305.15652)
* Towards Continual Adaptation in Industrial Anomaly Detection [\[ACM MM 2022\]](https://dl.acm.org/doi/abs/10.1145/3503161.3548232)
* One-for-More: Continual Diffusion Model for Anomaly Detection [\[CVPR 2025\]](https://arxiv.org/abs/2502.19848)
* Memory Efficient Continual Learning for Edge-Based Visual Anomaly Detection [\[2025\]](https://arxiv.org/abs/2503.02691)[\[code\]](https://bitbucket.org/papers_vad_group/clad_paste/src/master/)
* CADIC: Continual Anomaly Detection Based on Incremental Coreset [\[2025\]](https://arxiv.org/abs/2511.08634)
* Exploring Multimodal Prompts For Unsupervised Continuous Anomaly Detection [\[ACM MM 2025\]](https://dl.acm.org/doi/abs/10.1145/3746027.3755219)
* Complementary Prototype Mapping for Efficient Multimodal Anomaly Detection [\[CVPR 2026\]](https://cvpr.thecvf.com/virtual/2026/poster/37601)
* GPFlow: Gaussian Prototype Probability Flow for Unsupervised Multi-Modal Anomaly Detection [\[CVPR 2026\]](https://cvpr.thecvf.com/virtual/2026/poster/37603)
* Geometry-Aligned and Anomaly-Aware Reconstruction for 3D Anomaly Detection [\[CVPR 2026\]](https://cvpr.thecvf.com/virtual/2026/poster/36699)
* A Semantically Disentangled Unified Model for Multi-category 3D Anomaly Detection [\[CVPR 2026\]](https://arxiv.org/abs/2603.25159)[\[code\]](https://visualsciencelab-khu.github.io/SeDiR_project)
* DeCoFlow: Structural Decomposition of Normalizing Flows for Continual Anomaly Detection [\[ECCV 2026\]](https://arxiv.org/abs/2606.26687)

## 3.7 Uniform/Multi-Class AD

* Dinomaly: The Less Is More Philosophy in Multi-Class Unsupervised Anomaly Detection [\[CVPR 2025\]](https://arxiv.org/abs/2405.14325)[\[code\]](https://github.com/guojiajeremy/Dinomaly) ⭐ 516 | 🐛 30 | 🌐 Python | 📅 2026-05-21
* Learning Feature Inversion for Multi-class Anomaly Detection under General-purpose COCO-AD Benchmark [\[2024\]](https://arxiv.org/abs/2404.10760)[\[code\]](https://github.com/zhangzjn/ader) ⭐ 350 | 🐛 39 | 🌐 Python | 📅 2025-01-29
* A Unified Model for Multi-class Anomaly Detection [\[NeurIPS 2022\]](https://arxiv.org/pdf/2206.03687.pdf) [\[code\]](https://github.com/zhiyuanyou/UniAD) ⭐ 331 | 🐛 1 | 🌐 Python | 📅 2022-11-22
* UniNet: A Contrastive Learning-guided Unified Framework with Feature Selection for Anomaly Detection [\[CVPR 2025\]](https://pangdatangtt.github.io/)[\[code coming soon\]](https://github.com/pangdatangtt/UniNet) ⭐ 137 | 🐛 14 | 🌐 Python | 📅 2025-08-12
* ResAD: A Simple Framework for Class Generalizable Anomaly Detection [\[NeurIPS 2024\]](https://arxiv.org/abs/2410.20047)[\[code\]](https://github.com/xcyao00/ResAD) ⭐ 94 | 🐛 15 | 🌐 Python | 📅 2026-06-02
* Hierarchical Vector Quantized Transformer for Multi-class Unsupervised Anomaly Detection [\[NeurIPS 2023\]](https://openreview.net/pdf?id=clJTNssgn6)[\[code\]](https://github.com/RuiyingLu/HVQ-Trans) ⭐ 49 | 🐛 6 | 🌐 Python | 📅 2024-04-20
* An Incremental Unified Framework for Small Defect Inspection [\[ECCV2024\]](https://arxiv.org/abs/2312.08917v2)[\[code\]](https://github.com/jqtangust/IUF) ⭐ 49 | 🐛 2 | 🌐 Python | 📅 2025-02-17
* Boosting Global-Local Feature Matching via Anomaly Synthesis for Multi-Class Point Cloud Anomaly Detection [\[TASE 2025\]](https://ieeexplore.ieee.org/abstract/document/10898004) [\[code\]](https://github.com/hustCYQ/GLFM-Multi-class-3DAD) ⭐ 47 | 🐛 1 | 🌐 Python | 📅 2026-01-09
* MoEAD: A Parameter-efficient Model for Multi-class Anomaly Detection [\[ECCV 2024\]](https://eccv2024.ecva.net/virtual/2024/poster/2653)[\[code\]](https://github.com/TheStarOfMSY/MoEAD) ⭐ 36 | 🐛 2 | 🌐 Python | 📅 2024-12-20
* Continuous Memory Representation for Anomaly Detection [\[ECCV 2024\]](https://arxiv.org/abs/2402.18293)[\[homepage\]](https://tae-mo.github.io/crad/)[\[code\]](https://github.com/tae-mo/CRAD) ⭐ 34 | 🐛 1 | 🌐 Python | 📅 2024-11-27
* Center-aware Residual Anomaly Synthesis for Multi-class Industrial Anomaly Detection [\[TII 2025\]](https://arxiv.org/abs/2505.17551v1)[\[code\]](https://github.com/cqylunlun/CRAS) ⭐ 34 | 🐛 1 | 🌐 Python | 📅 2026-03-30
* Hierarchical Gaussian Mixture Normalizing Flow Modeling for Unified Anomaly Detection [\[ECCV 2024\]](https://arxiv.org/abs/2403.13349)[\[code\]](https://github.com/xcyao00/HGAD) ⭐ 30 | 🐛 1 | 🌐 Python | 📅 2026-06-02
* UniFormaly: Towards Task-Agnostic Unified Framework for Visual Anomaly Detection [\[2023\]](https://arxiv.org/abs/2307.12540)[\[code\]](https://github.com/YoojLee/Uniformaly) ⭐ 16 | 🐛 5 | 🌐 Python | 📅 2023-09-15
* Revitalizing Reconstruction Models for Multi-class Anomaly Detection via Class-Aware Contrastive Learning [\[2024\]](https://arxiv.org/abs/2412.04769)[\[code\]](https://github.com/LGC-AD/AD-LGC) ⭐ 11 | 🐛 2 | 🌐 Python | 📅 2025-11-26
* MC3D-AD: A Unified Geometry-aware Reconstruction Model for Multi-category 3D Anomaly Detection [\[IJCAI 2025\]](https://arxiv.org/abs/2505.01969)[\[code\]](https://github.com/jiayi-art/MC3D-AD) ⭐ 8 | 🐛 0 | 🌐 Python | 📅 2026-03-31
* ReFP-AD: Rectified Flow Preconditioning for Energy-Based Anomaly Detection [\[ECCV 2026\]](https://arxiv.org/abs/2608.01793)[\[code\]](https://github.com/CLendering/ReFP-AD) ⭐ 8 | 🐛 0 | 🌐 Python | 📅 2026-08-17
* VLMDiff: Leveraging Vision-Language Models for Multi-Class Anomaly Detection with Diffusion [\[2025\]](https://arxiv.org/abs/2511.08173)[\[code\]](https://github.com/giddyyupp/VLMDiff) ⭐ 7 | 🐛 1 | 🌐 Python | 📅 2026-02-02
* MaskAD: Parallel Masked Autoencoder for Multi-class Unsupervised Anomaly Detection [\[AAAI 2026\]](https://ojs.aaai.org/index.php/AAAI/article/download/38573/42535)[\[code\]](https://github.com/liugang-xd/MaskAD) ⭐ 3 | 🐛 0 | 🌐 Python | 📅 2026-05-31
* Collaborative Reconstruction and Repair for Multi-class Industrial Anomaly Detection [\[Data Intelligence 2025\]](https://arxiv.org/abs/2512.11401)[\[code\]](https://github.com/qiqigeww/CRR-MUAD) ⭐ 2 | 🐛 1 | 🌐 Python | 📅 2025-12-20
* OmniAL A unifiled CNN framework for unsupervised anomaly localization [\[CVPR 2023\]](https://openaccess.thecvf.com/content/CVPR2023/papers/Zhao_OmniAL_A_Unified_CNN_Framework_for_Unsupervised_Anomaly_Localization_CVPR_2023_paper.pdf)
* SelFormaly: Towards Task-Agnostic Unified Anomaly Detection[\[2023\]](https://arxiv.org/abs/2307.12540)
* Removing Anomalies as Noises for Industrial Defect Localization [\[ICCV 2023\]](https://openaccess.thecvf.com/content/ICCV2023/papers/Lu_Removing_Anomalies_as_Noises_for_Industrial_Defect_Localization_ICCV_2023_paper.pdf)
* MSTAD: A masked subspace-like transformer for multi-class anomaly detection [\[2023\]](https://www.sciencedirect.com/science/article/pii/S095070512300936X)
* LafitE: Latent Diffusion Model with Feature Editing for Unsupervised Multi-class Anomaly Detection [\[2023\]](https://arxiv.org/abs/2307.08059)
* DiAD: A Diffusion-based Framework for Multi-class Anomaly Detection [\[AAAI 2024\]](https://ojs.aaai.org/index.php/AAAI/article/view/28690)[\[code\]](https://lewandofskee.github.io/projects/diad)
* Structural Teacher-Student Normality Learning for Multi-Class Anomaly Detection and Localization [\[2024\]](https://arxiv.org/abs/2402.17091)
* Unsupervised anomaly detection and localization with one model for all category [\[KBS 2024\]](https://www.sciencedirect.com/science/article/pii/S0950705124001680)
* Anomaly Detection by Adapting a pre-trained Vision Language Model [\[2024\]](https://arxiv.org/abs/2403.09493)
* DMAD: Dual Memory Bank for Real-World Anomaly Detection [\[2024\]](https://arxiv.org/abs/2403.12362)
* Toward Multi-class Anomaly Detection: Exploring Class-aware Unified Model against Inter-class Interference [\[2024\]](https://arxiv.org/abs/2403.14213)
* Long-Tailed Anomaly Detection with Learnable Class Names [\[CVPR 2024\]](https://arxiv.org/abs/2403.20236)[\[data split\]](https://zenodo.org/records/10854201)
* MambaAD: Exploring State Space Models for Multi-class Unsupervised Anomaly Detection [\[NeurIPS 2024\]](https://arxiv.org/abs/2404.06564)[\[code\]](https://lewandofskee.github.io/projects/MambaAD/)
* Prior Normality Prompt Transformer for Multi-class Industrial Image Anomaly Detection [\[TII 2024\]](https://arxiv.org/abs/2406.11507)
* Learning Multi-view Anomaly Detection [\[2024\]](https://arxiv.org/abs/2407.11935)
* Exploring Plain ViT Reconstruction for Multi-class Unsupervised Anomaly Detection [\[CVIU 2025\]](https://www.sciencedirect.com/science/article/abs/pii/S1077314225000311?via%3Dihub)[\[code\]](https://zhangzjn.github.io/projects/ViTAD/)
* Exploiting Point-Language Models with Dual-Prompts for 3D Anomaly Detection [\[2025\]](https://arxiv.org/abs/2502.11307)
* Learning Invariant Discriminative Patterns for Unified Anomaly Detection [\[ACM MM 2025\]](https://dl.acm.org/doi/abs/10.1145/3746027.3755179)
* DecAD: Decoupling Anomalies in Latent Space for Multi-Class Unsupervised Anomaly Detection [\[ICCV 2025\]](https://openaccess.thecvf.com/content/ICCV2025/html/Wang_DecAD_Decoupling_Anomalies_in_Latent_Space_for_Multi-Class_Unsupervised_Anomaly_ICCV_2025_paper.html)

## 3.8 Logical AD

* Towards Training-free Anomaly Detection with Vision and Language Foundation Models [\[CVPR 2025\]](https://arxiv.org/abs/2503.18325)[\[code\]](https://github.com/zhang0jhon/LogSAD) ⭐ 100 | 🐛 12 | 🌐 Python | 📅 2025-05-19
* Few Shot Part Segmentation Reveals Compositional Logic for Industrial Anomaly Detection [\[AAAI 2024\]](https://ojs.aaai.org/index.php/AAAI/article/view/28703)[\[code\]](https://github.com/oopil/PSAD_logical_anomaly_detection) ⭐ 65 | 🐛 10 | 🌐 Python | 📅 2025-05-09
* CSAD: Unsupervised Component Segmentation for Logical Anomaly Detection [\[BMVC 2024\]](https://arxiv.org/abs/2408.15628)[\[code\]](https://github.com/Tokichan/CSAD) ⭐ 63 | 🐛 1 | 🌐 Python | 📅 2025-08-22
* SALAD -- Semantics-Aware Logical Anomaly Detection [\[ICCV 2025\]](https://arxiv.org/abs/2509.02101)[\[code\]](https://github.com/MaticFuc/SALAD) ⭐ 49 | 🐛 10 | 🌐 Python | 📅 2025-10-03
* Set Features for Fine-grained Anomaly Detection[\[2023\]](https://arxiv.org/abs/2302.12245) [\[code\]](https://github.com/NivC/SINBAD) ⭐ 36 | 🐛 0 | 🌐 Python | 📅 2024-11-10
* Supervised Anomaly Detection for Complex Industrial Images [\[2024\]](https://arxiv.org/abs/2405.04953)[\[code\]](https://github.com/abc-125/segad) ⭐ 35 | 🐛 1 | 🌐 Python | 📅 2025-09-07
* REB: Reducing Biases in Representation for Industrial Anomaly Detection [\[2023\]](https://arxiv.org/abs/2308.12577)[\[code\]](https://github.com/ShuaiLYU/REB) ⭐ 26 | 🐛 9 | 🌐 Jupyter Notebook | 📅 2024-01-22
* AnomalyXFusion: Multi-modal Anomaly Synthesis with Diffusion [\[2024\]](https://arxiv.org/abs/2404.19444)[\[data\]](https://github.com/hujiecpp/MVTec-Caption) ⭐ 22 | 🐛 3 | 🌐 Python | 📅 2024-10-30
* Learning Global-Local Correspondence with Semantic Bottleneck for Logical Anomaly Detection [\[TCSVT 2023\]](https://arxiv.org/abs/2303.05768)[\[code\]](https://github.com/hmyao22/GLCF) ⭐ 14 | 🐛 2 | 🌐 Python | 📅 2023-08-13
* VID-AD: A Dataset for Image-Level Logical Anomaly Detection under Vision-Induced Distraction [\[2026\]](https://arxiv.org/abs/2603.13964)[\[data\]](https://github.com/nkthiroto/VID-AD) ⭐ 11 | 🐛 1 | 🌐 Python | 📅 2026-03-26
* LogiCo: A Unified Framework for Logical and Structural Anomaly Detection [\[ECCV 2026\]](https://arxiv.org/abs/2606.28688)[\[code\]](https://github.com/cnulab/LogiCo) ⭐ 11 | 🐛 1 | 🌐 Python | 📅 2026-06-30
* Revisiting Deep Feature Reconstruction for Logical and Structural Industrial Anomaly Detection[\[TMLR 2024\]](https://arxiv.org/abs/2410.16255)[\[code\]](https://github.com/sukanyapatra1997/ULSAD-2024) ⭐ 8 | 🐛 1 | 🌐 Python | 📅 2024-04-24
* LogicAD: Explainable Anomaly Detection via VLM-based Text Feature Extraction [\[AAAI 2025\]](https://arxiv.org/abs/2501.01767)[\[code\]](https://github.com/jasonjin34/logicAD) ⭐ 7 | 🐛 4 | 🌐 Python | 📅 2025-10-27
* Beyond Dents and Scratches: Logical Constraints in Unsupervised Anomaly Detection and Localization [\[IJCV 2022\]](https://link.springer.com/content/pdf/10.1007/s11263-022-01578-9.pdf)
* EfficientAD: Accurate Visual Anomaly Detection at Millisecond-Level Latencies [\[WACV 2024\]](https://openaccess.thecvf.com/content/WACV2024/papers/Batzner_EfficientAD_Accurate_Visual_Anomaly_Detection_at_Millisecond-Level_Latencies_WACV_2024_paper.pdf)
* Contextual Affinity Distillation for Image Anomaly Detection [\[WACV 2024\]](https://openaccess.thecvf.com/content/WACV2024/papers/Zhang_Contextual_Affinity_Distillation_for_Image_Anomaly_Detection_WACV_2024_paper.pdf)
* Template-guided Hierarchical Feature Restoration for Anomaly Detection [\[ICCV 2023\]](https://openaccess.thecvf.com/content/ICCV2023/papers/Guo_Template-guided_Hierarchical_Feature_Restoration_for_Anomaly_Detection_ICCV_2023_paper.pdf)
* Generating and Reweighting Dense Contrastive Patterns for Unsupervised Anomaly Detection [\[AAAI 2024\]](https://ojs.aaai.org/index.php/AAAI/article/view/27910)
* PUAD: Frustratingly Simple Method for Robust Anomaly Detection [\[2024\]](https://arxiv.org/abs/2402.15143)
* SAM-LAD: Segment Anything Model Meets Zero-Shot Logic Anomaly Detection [\[2024\]](https://arxiv.org/abs/2406.00625)
* SLSG: Industrial Image Anomaly Detection with Improved Feature Embeddings and One-Class Classification [\[PR 2024\]](https://www.sciencedirect.com/science/article/pii/S0031320324006137)
* LogiCode: an LLM-Driven Framework for Logical Anomaly Detection [\[2024\]](https://arxiv.org/pdf/2406.04687)
* Component-aware Unsupervised Logical Anomaly Generation for Industrial Anomaly Detection [\[2025\]](https://arxiv.org/abs/2502.11712)
* Uniad: Integrating geometric and semantic cues for unified anomaly detection [\[ACM MM 2025\]](https://dl.acm.org/doi/abs/10.1145/3746027.3755422)
* Logical Anomaly Detection with Text-based Logic via Component-Aware Contrastive Language-Image Training [\[KDD 25\]](https://dl.acm.org/doi/abs/10.1145/3711896.3737032)

## 3.9 MLLM-based AD

* AnomalyGPT: Detecting Industrial Anomalies using Large Vision-Language Models [\[AAAI 2024\]](https://arxiv.org/abs/2308.15366)[\[code\]](https://github.com/CASIA-IVA-Lab/AnomalyGPT) ⭐ 1,132 | 🐛 61 | 🌐 Python | 📅 2023-12-20[\[project page\]](https://anomalygpt.github.io/)
* MMAD: The Comprehensive Benchmark for Multimodal Large Language Models in Industrial Anomaly Detection [\[ICLR 2025\]](https://openreview.net/forum?id=JDiER86r8v)[\[Code\]](https://github.com/jam-cc/MMAD) ⭐ 271 | 🐛 1 | 🌐 Python | 📅 2026-01-14  [\[Data\]](https://huggingface.co/datasets/jiang-cc/MMAD)
* Towards Generic Anomaly Detection and Understanding: Large-scale Visual-linguistic Model (GPT-4V) Takes the Lead [\[2023\]](https://arxiv.org/abs/2311.02782)[\[code\]](https://github.com/caoyunkang/GPT4V-for-Generic-Anomaly-Detection) ⭐ 129 | 🐛 1 | 📅 2025-03-04
* IAD-R1: Reinforcing Consistent Reasoning in Industrial Anomaly Detection [\[AAAI 2026\]](https://arxiv.org/abs/2508.09178)[\[code\]](https://github.com/Yanhui-Lee/IAD-R1) ⭐ 96 | 🐛 17 | 🌐 Python | 📅 2025-12-09
* AD-Copilot: A Vision-Language Assistant for Industrial Anomaly Detection via Visual In-context Comparison [\[2026\]](https://arxiv.org/pdf/2603.13779)[\[Code\]](https://github.com/jam-cc/AD-Copilot) ⭐ 64 | 🐛 5 | 🌐 Python | 📅 2026-04-25[\[Model\]](https://huggingface.co/jiang-cc/AD-Copilot)[\[Demo\]](https://huggingface.co/spaces/jiang-cc/AD-Copilot)
* Exploring Grounding Potential of VQA-oriented GPT-4V for Zero-shot Anomaly Detection [\[IJCAI WORKSHOP 2024\]](https://arxiv.org/abs/2311.02612)[\[code\]](https://github.com/zhangzjn/GPT-4V-AD) ⭐ 31 | 🐛 4 | 🌐 Python | 📅 2023-11-07
* EMIT: Enhancing MLLMs for Industrial Anomaly Detection via Difficulty-Aware [\[2025\]](https://arxiv.org/abs/2507.21619)[\[code\]](https://github.com/guanwei49/EMIT) ⭐ 29 | 🐛 1 | 🌐 Python | 📅 2026-01-24
* AnomalyAgent: Training-Free Agentic Models for Zero-/Few-Shot Anomaly Detection [\[2026\]](https://arxiv.org/abs/2605.30140)[\[code\]](https://github.com/mala-lab/AnomalyAgent) ⭐ 15 | 🐛 0 | 🌐 Python | 📅 2026-05-28
* M3-AD: Reflection-aware Multi-modal, Multi-category, and Multi-dimensional
  Benchmark and Framework for Industrial Anomaly Detection [\[2026\]](https://arxiv.org/abs/2603.00055)[\[code\]](https://github.com/Yanhui-Lee/M3-AD) ⭐ 14 | 🐛 1 | 📅 2026-02-10
* Reason-IAD: Knowledge-Guided Dynamic Latent Reasoning for Explainable Industrial Anomaly Detection [\[2026\]](https://arxiv.org/abs/2602.09850)[\[code\]](https://github.com/chenpeng052/Reason-IAD) ⭐ 10 | 🐛 3 | 📅 2026-05-07
* Reasoning-Driven Anomaly Detection and Localization with Image-Level Supervision [\[CVPR 2026\]](https://arxiv.org/abs/2603.27179)[\[code\]](https://github.com/YizhouJin313/ReADL) ⭐ 5 | 🐛 2 | 📅 2026-03-11
* Customizing Visual-Language Foundation Models for Multi-modal Anomaly Detection and Reasoning [\[CSCWD 2024\]](https://arxiv.org/abs/2403.11083)
* Do LLMs Understand Visual Anomalies? Uncovering LLM Capabilities in Zero-shot Anomaly Detection [\[ACM MM 2024\]](https://arxiv.org/abs/2404.09654)
* LogiCode: an LLM-Driven Framework for Logical Anomaly Detection [\[T-ASE 2024\]](https://arxiv.org/pdf/2406.04687)
* FabGPT: An Efficient Large Multimodal Model for Complex Wafer Defect Knowledge Queries [\[ICCAD 2024\]](https://arxiv.org/abs/2407.10810)
* VMAD: Visual-enhanced Multimodal Large Language Model for Zero-Shot Anomaly Detection [\[T-ASE\]](https://ieeexplore.ieee.org/document/11088265)[\[2024\]](https://arxiv.org/abs/2409.20146)
* Are Anomaly Scores Telling the Whole Story? A Benchmark for Multilevel Anomaly Detection [\[2024\]](https://arxiv.org/abs/2411.14515)
* Can Multimodal Large Language Models be Guided to Improve Industrial Anomaly Detection? [\[CIE 2025\]](https://arxiv.org/abs/2501.15795)
* EIAD: Explainable Industrial Anomaly Detection Via Multi-Modal Large Language Models [\[ICME 2025\]](https://arxiv.org/abs/2503.14162v1)
* Towards Zero-Shot Anomaly Detection and Reasoning with Multimodal Large Language Models [\[CVPR 2025\]](https://arxiv.org/abs/2502.07601)[\[code\]](https://xujiacong.github.io/Anomaly-OV/)
* AnomalyR1: A GRPO-based End-to-end MLLM for Industrial Anomaly Detection [\[2025\]](https://arxiv.org/abs/2504.11914)
* Detect, Classify, Act: Categorizing Industrial Anomalies with Multi-Modal Large Language Models [\[VAND 2025\]](https://arxiv.org/abs/2505.02626)
* Triad: Empowering LMM-based Anomaly Detection with Vision Expert-guided Visual Tokenizer and Manufacturing Process [\[ICCV 2025\]](https://openaccess.thecvf.com/content/ICCV2025/html/Li_Triad_Empowering_LMM-based_Anomaly_Detection_with_Expert-guided_Region-of-Interest_Tokenizer_and_ICCV_2025_paper.html)
* LR-IAD: Mask-Free Industrial Anomaly Detection with Logical Reasoning [\[2025\]](https://arxiv.org/abs/2504.19524)
* OmniAD: Detect and Understand Industrial Anomaly via Multimodal Reasoning [\[2025\]](https://arxiv.org/abs/2505.22039)
* AD-FM: Multimodal LLMs for Anomaly Detection via Multi-Stage Reasoning and Fine-Grained Reward Optimization [\[AAAI 2026\]](https://arxiv.org/abs/2508.04175)
* AgentIAD: Tool-Augmented Single-Agent for Industrial Anomaly Detection [\[2025\]](https://arxiv.org/abs/2512.13671)
* Judo: A Juxtaposed Domain-oriented Multimodal Reasoner for Industrial Anomaly QA [\[ICLR 2026\]](https://openreview.net/forum?id=XW4mROtaVb)
* MAU-GPT: Enhancing Multi-type Industrial Anomaly Understanding via Anomaly-aware and Generalist Experts Adaptation [\[AAAI 2026\]](https://arxiv.org/abs/2602.07011)
* SAGE: A Visual Language Model for Anomaly Detection via Fact Enhancement and Entropy-aware Alignment [\[2026\]](https://arxiv.org/abs/2507.07939)
* IAD-Unify: A Region-Grounded Unified Model for Industrial Anomaly Segmentation, Understanding, and Generation [\[2026\]](https://arxiv.org/abs/2604.12440)
* MMR-AD: A Large-Scale Multimodal Dataset for Benchmarking General Anomaly Detection with Multimodal Large Language Models [\[CVPR 2026\]](https://arxiv.org/abs/2604.10971)[\[code\]](https://xcyao00.github.io/MMR-AD)
* IndusAgent: Reinforcing Open-Vocabulary Industrial Anomaly Detection with Agentic Tools [\[2026\]](https://arxiv.org/abs/2605.20682)
* Global Logic and Local Search: Dual-Stream Multimodal In-Context Learning for Verifiable Industrial Anomaly Detection [\[ECCV 2026\]](https://arxiv.org/abs/2607.03817)

## 3.10 Video IAD

* Towards Visual Discrimination and Reasoning of Real-World Physical Dynamics: Physics-Grounded Anomaly Detection [\[CVPR 2025\]](https://arxiv.org/abs/2503.03562)[\[code\]](https://github.com/Chopper-233/Physics-AD) ⭐ 23 | 🐛 0 | 🌐 Python | 📅 2026-02-24
* O-VAD: Industrial Video Anomaly Detection through Object-Centric Tracking and Reasoning [\[ECCV 2026\]](https://arxiv.org/abs/2607.18142)[\[code\]](https://o-vad.github.io/)

## Other settings

### TTT binary segmentation

* Test Time Training for Industrial Anomaly Segmentation [\[2024\]](https://arxiv.org/abs/2404.03743)

### MoE with TTA

* Adapted-MoE: Mixture of Experts with Test-Time Adaption for Anomaly Detection[\[2024\]](https://arxiv.org/abs/2409.05611)[\[code coming soon\]]()

### Adversary Attack

* Adversarially Robust Industrial Anomaly Detection Through Diffusion Model [\[2024\]](https://arxiv.org/abs/2408.04839)
* Adversarially Robust Anomaly Detection through Spurious Negative Pair Mitigation [\[ICLR 2025\]](https://openreview.net/forum?id=t8fu5m8R5m)

### Defect Classification

* AnomalyNCD: Towards Novel Anomaly Class Discovery in Industrial Scenarios [\[2024\]](https://arxiv.org/abs/2410.14379)[\[code coming soon\]](https://github.com/HUST-SLOW/AnomalyNCD) ⭐ 165 | 🐛 11 | 🌐 Python | 📅 2025-09-01
* MVREC: A General Few-shot Defect Classification Model Using Multi-View Region-Context [\[AAAI 2025\]](https://arxiv.org/abs/2412.16897)
* Defect Cue-Preserved Structural Feature Refinement for Few-Shot Anomaly Detection [\[CVPR 2026\]](https://cvpr.thecvf.com/virtual/2026/poster/39808)
* UniSpector: Towards Universal Open-set Defect Recognition via Spectral-Contrastive Visual Prompting [\[CVPR 2026\]](https://cvpr.thecvf.com/virtual/2026/poster/38081)
* Towards Open-Vocabulary Industrial Defect Understanding with a Large-Scale Multimodal Dataset [\[CVPR 2026\]](https://arxiv.org/abs/2512.24160)[\[data\]](https://ninaneon.github.io/projectpage)

### Rubustness

* FiCo: Filter or Compensate: Towards Invariant Representation from Distribution Shift for Anomaly Detection [\[AAAI 2025\]](https://arxiv.org/abs/2412.10115)[\[code\]](https://github.com/znchen666/FiCo) ⭐ 10 | 🐛 2 | 🌐 Python | 📅 2024-12-22

### Universal Task

* UniMMAD: Unified Multi-Modal and Multi-Class Anomaly Detection via MoE-Driven Feature Decompression [\[CVPR 2026\]](https://arxiv.org/abs/2509.25934)[\[code\]](https://github.com/yuanzhao-CVLAB/UniMMAD) ⭐ 245 | 🐛 4 | 🌐 Python | 📅 2026-05-20
* One Dinomaly2 Detect Them All: A Unified Framework for Full-Spectrum Unsupervised Anomaly Detection [\[2025\]](https://arxiv.org/abs/2510.17611) [\[code\]](https://github.com/guojiajeremy/Dinomaly2) ⭐ 102 | 🐛 4 | 🌐 Python | 📅 2026-05-30
* Unified Unsupervised Anomaly Detection via Matching Cost Filtering [\[2025\]](https://arxiv.org/abs/2510.03363)[\[code\]](https://github.com/ZHE-SAPI/CostFilter-AD) ⭐ 93 | 🐛 1 | 🌐 Python | 📅 2026-07-03
* UniADC: A Unified Framework for Anomaly Detection and Classification [\[2025\]](https://arxiv.org/abs/2511.06644)[\[code is comming\]](https://github.com/cnulab/UniADC) ⭐ 33 | 🐛 2 | 🌐 Python | 📅 2026-06-30
* AnomalyMoE: Towards a Language-free Generalist Model for Unified Visual Anomaly Detection [\[AAAI 2025\]](https://arxiv.org/abs/2508.06203)[\[code\]](https://github.com/CASIA-LMC-Lab/AnomalyMoE) ⭐ 32 | 🐛 2 | 🌐 Python | 📅 2025-11-17

# 4 Dataset

| Dataset                                                                                                                                            | Class | Normal | Abnormal | Total  | Annotation level  | Source                | Time         |
| -------------------------------------------------------------------------------------------------------------------------------------------------- | ----- | ------ | -------- | ------ | ----------------- | --------------------- | ------------ |
| [3CAD](https://github.com/EnquanYang2022/3CAD) ⭐ 66 \| 🐛 5 \| 🌐 Python \| 📅 2025-04-22                                                          | 8     | 15577  | 11462    | 27039  | Segmentation mask | RGB real              | AAAI, 2025   |
| [AITEX](https://www.cvmart.net/dataSets/detail/300)                                                                                                | 1     | 140    | 105      | 245    | Segmentation mask | RGB real              | 2019         |
| [Anomaly-ShapeNet](https://github.com/Chopper-233/Anomaly-ShapeNet) ⭐ 57 \| 🐛 8 \| 📅 2025-10-17                                                  | 40    | -      | -        | 1600   | Point-level mask  | Point-cloud synthetic | CVPR,2024    |
| [BTAD](http://avires.dimi.uniud.it/papers/btad/btad.zip)                                                                                           | 3     | -      | -        | 2830   | Segmentation mask | RGB real              | 2021         |
| [CID](https://github.com/LightZH/Insulator-Defect-Detection) ⭐ 8 \| 🐛 0 \| 📅 2024-04-30                                                          | 1     | 4060   | 233      | 4293   | Segmentation mask | RGB real              | 2024,TIM     |
| [DAGM](https://www.kaggle.com/datasets/mhskjelvareid/dagm-2007-competition-dataset-optical-inspection)                                             | 10    | -      | -        | 11500  | Segmentation mask | RGB synthetic         | 2007         |
| [DEEPPCB](https://github.com/tangsanli5201/DeepPCB) ⭐ 540 \| 🐛 9 \| 🌐 Python \| 📅 2018-12-19                                                    | 1     | -      | -        | 1500   | Bounding box      | RGB synthetic         | 2019         |
| [DTD-Synthetic](https://drive.google.com/drive/folders/10OyPzvI3H6llCZBxKxFlKWt1Pw1tkMK1)                                                          | 12    | -      | -        | -      | Segmentation mask | RGB synthetic         | WACV,2024    |
| [Eyecandies](https://eyecan-ai.github.io/eyecandies/)                                                                                              | 10    | 13250  | 2250     | 15500  | Segmentation mask | RGBD synthetic image  | ACCV,2022    |
| [Fabirc dataset](http://hub.hku.hk/bitstream/10722/229176/1/content.pdf)                                                                           | 1     | 25     | 25       | 50     | Segmentation mask | RGB synthetic         | PR,2016      |
| [GDXray](https://domingomery.ing.puc.cl/material/gdxray/)                                                                                          | 1     | 0      | 19407    | 19407  | Bounding box      | RGB real              | 2016         |
| [IPAD](https://ljf1113.github.io/IPAD_VAD/)                                                                                                        | 16    | -      | -        | 597979 | Image             | Video real\&synthetic | 2024         |
| [KolekrotSDD](https://www.vicos.si/resources/kolektorsdd/)                                                                                         | 1     | 347    | 52       | 399    | Segmentation mask | RGB real              | JIM,2019     |
| [KolekrotSDD2](https://www.vicos.si/resources/kolektorsdd2/)                                                                                       | 1     | 2979   | 356      | 3335   | Segmentation mask | RGB real              | CiI,2021     |
| [MCBT](https://github.com/ganatma/AL-CycleGAN/tree/main/MCBT) ⭐ 2 \| 🐛 0 \| 🌐 Python \| 📅 2026-04-16 (Manufacturing Complex Background Texture) | -     | 800    | 227      | 1027   | Segmentation mask | RGB real              | CVPRW,2025   |
| [MIAD](https://miad-2022.github.io/)                                                                                                               | 7     | 87500  | 17500    | 105000 | Segmentation mask | RGB synthetic         | 2023         |
| [MPDD](https://github.com/stepanje/MPDD) ⭐ 52 \| 🐛 0 \| 📅 2024-07-24                                                                             | 6     | 1064   | 282      | 1346   | Segmentation mask | RGB real              | ICUMT,2021   |
| [MTD](https://github.com/abin24/Magnetic-tile-defect-datasets.) ⭐ 221 \| 🐛 5 \| 📅 2020-05-09                                                     | 1     | 952    | 392      | 1344   | Segmentation mask | RGB real              | CASE,2018    |
| [MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad)                                                                               | 15    | 4096   | 1258     | 5354   | Segmentation mask | RGB real              | CVPR,2019    |
| [MVTec 3D-AD](https://www.mvtec.com/company/research/datasets/mvtec-3d-ad)                                                                         | 10    | 2904   | 948      | 3852   | Segmentation mask | RGB real              | VISAPP,2021  |
| [MVTec LOCO-AD](https://www.mvtec.com/company/research/datasets/mvtec-loco)                                                                        | 5     | 2347   | 993      | 3340   | Segmentation mask | RGBD real             | IJCV,2022    |
| [NanoTwice](http://web.mi.imati.cnr.it/ettore/NanoTwice)                                                                                           | 1     | 5      | 40       | 45     | Segmentation mask | RGB real              | TII,2016     |
| [NEU surface defect](http://faculty.neu.edu.cn/songkechen/zh_CN/zdylm/263270/list/index.htm)                                                       | 1     | 0      | 1800     | 1800   | Bounding box      | RGB real              | 2013         |
| [PAD](https://github.com/EricLee0224/PAD) ⭐ 104 \| 🐛 5 \| 🌐 Python \| 📅 2024-12-02                                                              | 20    | 5231   | 4902     | 10133  | Segmentation mask | RBG synthetic         | NeurIPS,2023 |
| [Real-IAD](https://realiad4ad.github.io/Real-IAD/)                                                                                                 | 30    | 99721  | 51329    | 151050 | Segmentation mask | RGB real              | CVPR,2024    |
| [Real3D-AD](https://github.com/M-3LAB/Real3D-AD) ⭐ 162 \| 🐛 0 \| 🌐 Python \| 📅 2024-03-13                                                       | 12    | 652    | 602      | 1254   | Point-level mask  | Point-cloud real      | NeurIPS,2023 |
| [RSDD](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=\&arnumber=8063875)                                                                          | 2     | -      | -        | 195    | Segmentation mask | RGB real              | 2017         |
| [Steel defect detection](https://www.kaggle.com/code/ekhtiar/resunet-a-baseline-on-tensorflow/notebook)                                            | 1     | -      | -        | 18076  | Image             | RGB real              | 2019         |
| [Steel tube dataset](https://github.com/huangyebiaoke/steel-pipe-weld-defect-detection) ⭐ 116 \| 🐛 6 \| 🌐 Jupyter Notebook \| 📅 2021-12-06      | 1     | 0      | 3408     | 3408   | Bounding box      | RGB real              | 2021         |
| [VisA](https://github.com/amazon-science/spot-diff) ⭐ 255 \| 🐛 4 \| 🌐 Python \| 📅 2022-09-22                                                    | 12    | 9621   | 1200     | 10821  | Segmentation mask | RGB real              | ECCV,2022    |
| [RAD](https://github.com/hustCYQ/RAD-dataset) ⭐ 12 \| 🐛 1 \| 🌐 Python \| 📅 2024-09-19                                                           | 4     | 213    | 1224     | 1224   | Segmentation mask | RGB real              | CASE,2024    |

* (DEEP PCB)Online PCB defect detector on a new PCB defect dataset [\[2019\]](https://arxiv.org/pdf/1902.06197.pdf) [\[data\]](https://github.com/tangsanli5201/DeepPCB) ⭐ 540 | 🐛 9 | 🌐 Python | 📅 2018-12-19
* The Woven Fabric Defect Detection (WFDD) dataset [\[2024\]](https://arxiv.org/abs/2407.09359)[\[data\]](https://github.com/cqylunlun/GLASS?tab=readme-ov-file#1wfdd-download-link) ⭐ 392 | 🐛 5 | 🌐 Python | 📅 2026-03-30
* (CPLID) Insulator Data Set - Chinese Power Line Insulator Dataset [\[data\]](https://github.com/InsulatorData/InsulatorDataSet) ⭐ 354 | 🐛 5 | 📅 2018-11-19
* MMAD: The First-Ever Comprehensive Benchmark for Multimodal Large Language Models in Industrial Anomaly Detection [\[ICLR 2025\]](https://arxiv.org/abs/2410.09453) [\[data\]](https://github.com/jam-cc/MMAD) ⭐ 271 | 🐛 1 | 🌐 Python | 📅 2026-01-14✨✨✨
* (VisA)SPot-the-Difference Self-supervised Pre-training for Anomaly Detection and Segmentation [\[ECCV 2022\]](https://arxiv.org/pdf/2207.14315.pdf) [\[data\]](https://github.com/amazon-science/spot-diff) ⭐ 255 | 🐛 4 | 🌐 Python | 📅 2022-09-22✨✨✨
* (MTD)Surface defect saliency of magnetic tile [\[2020\]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=\&arnumber=8560423) [\[data\]](https://github.com/abin24/Magnetic-tile-defect-datasets.) ⭐ 221 | 🐛 5 | 📅 2020-05-09
* InsPLAD: A Dataset and Benchmark for Power Line Asset Inspection in UAV Images [\[IJRS 2023\]](https://arxiv.org/abs/2311.01619)[\[data\]](https://github.com/andreluizbvs/InsPLAD) ⭐ 165 | 🐛 0 | 📅 2026-08-04
* Real3D-AD: A Dataset of Point Cloud Anomaly Detection [\[NeurIPS 2023\]](https://openreview.net/pdf?id=zGthDp4yYe)[\[data\]](https://github.com/M-3LAB/Real3D-AD) ⭐ 162 | 🐛 0 | 🌐 Python | 📅 2024-03-13✨✨✨
* Defect Spectrum: A Granular Look of Large-Scale Defect Datasets with Rich Semantics [\[ECCV 2024\]](https://openreview.net/forum?id=RLhS1TrjK3)[\[data\]](https://github.com/EnVision-Research/Defect_Spectrum) ⭐ 135 | 🐛 6 | 🌐 Python | 📅 2024-08-26
* Real-IAD: A Real-World Multi-view Dataset for Benchmarking Versatile Industrial Anomaly Detection [\[CVPR 2024\]](https://arxiv.org/abs/2403.12580)[\[code\]](https://github.com/Tencent/AnomalyDetection_Real-IAD) ⭐ 128 | 🐛 1 | 🌐 Python | 📅 2025-07-16[\[data\]](https://realiad4ad.github.io/Real-IAD/)✨✨✨
* (Steel tube dataset)Deep learning based steel pipe weld defect detection [\[2021\]](https://www.tandfonline.com/doi/pdf/10.1080/08839514.2021.1975391?needAccess=true) [\[data\]](https://github.com/huangyebiaoke/steel-pipe-weld-defect-detection) ⭐ 116 | 🐛 6 | 🌐 Jupyter Notebook | 📅 2021-12-06
* PAD: A Dataset and Benchmark for Pose-agnostic Anomaly Detection [\[NeurIPS 2023\]](https://github.com/EricLee0224/PAD) ⭐ 104 | 🐛 5 | 🌐 Python | 📅 2024-12-02
* (AeBAD)Industrial Anomaly Detection with Domain Shift: A Real-world Dataset and Masked Multi-scale Reconstruction [\[2023\]](https://arxiv.org/abs/2304.02216) [\[data\]](https://github.com/zhangzilongc/MMR) ⭐ 101 | 🐛 5 | 🌐 Python | 📅 2023-11-14
* Multi-Sensor Object Anomaly Detection: Unifying Appearance, Geometry, and Internal Properties [\[CVPR 2025\]](https://zzzbbbzzz.github.io/MulSen_AD/index.html)[\[code\]](https://github.com/ZZZBBBZZZ/MulSen-AD) ⭐ 94 | 🐛 4 | 🌐 Python | 📅 2025-03-20✨✨
* 3CAD: A Large-Scale Real-World 3C Product Dataset for Unsupervised Anomaly [\[AAAI 2025\]](https://arxiv.org/abs/2502.05761)[\[data\]](https://github.com/EnquanYang2022/3CAD) ⭐ 66 | 🐛 5 | 🌐 Python | 📅 2025-04-22✨✨
* (SSGD)SSGD: A smartphone screen glass dataset for defect detection [\[2023\]](https://arxiv.org/abs/2303.06673)[\[github page\]](https://github.com/VincentHancoder/SSGD) ⭐ 64 | 🐛 2 | 🌐 Python | 📅 2024-12-01
* Towards Scalable 3D Anomaly Detection and Localization: A Benchmark via 3D Anomaly Synthesis and A Self-Supervised Learning Network [\[CVPR 2024\]](https://arxiv.org/abs/2311.14897)[\[data\]](https://github.com/Chopper-233/Anomaly-ShapeNet) ⭐ 57 | 🐛 8 | 📅 2025-10-17
* (MPDD)Deep learning-based defect detection of metal parts: evaluating current methods in complex conditions [\[ICUMT 2021\]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=\&arnumber=9631567) [\[data\]](https://github.com/stepanje/MPDD) ⭐ 52 | 🐛 0 | 📅 2024-07-24
* RAD: A Dataset and Benchmark for Real-Life Anomaly Detection with Robotic Observations [\[IJCV 2024\]](https://arxiv.org/html/2410.00713v1)[\[data\]](https://github.com/kaichen-z/RAD) ⭐ 47 | 🐛 0 | 🌐 Python | 📅 2026-08-05
* PKU-GoodsAD: A Supermarket Goods Dataset for Unsupervised Anomaly Detection and Segmentation [\[2023\]](https://github.com/jianzhang96/GoodsAD) ⭐ 41 | 🐛 0 | 📅 2024-10-17[\[data\]](https://github.com/jianzhang96/GoodsAD) ⭐ 41 | 🐛 0 | 📅 2024-10-17✨✨
* Supervised Anomaly Detection for Complex Industrial Images [\[2024\]](https://arxiv.org/abs/2405.04953)[\[data\]](https://github.com/abc-125/segad) ⭐ 35 | 🐛 1 | 🌐 Python | 📅 2025-09-07
* MVTec-Caption: AnomalyXFusion: Multi-modal Anomaly Synthesis with Diffusion [\[2024\]](https://arxiv.org/abs/2404.19444)[\[data\]](https://github.com/hujiecpp/MVTec-Caption) ⭐ 22 | 🐛 3 | 🌐 Python | 📅 2024-10-30
* Anomaly Detection of Integrated Circuits Package Substrates Using the Large Vision Model SAIC: Dataset Construction, Methodology, and Application [\[ICCV 2025\]](https://openaccess.thecvf.com/content/ICCV2025/papers/Yu_Anomaly_Detection_of_Integrated_Circuits_Package_Substrates_Using_the_Large_ICCV_2025_paper.pdf)[\[data\]](https://github.com/Bingyang0410/CPS2D-AD) ⭐ 16 | 🐛 4 | 📅 2025-07-09
* PIAD: Pose and Illumination agnostic Anomaly Detection [\[CVPR 2025\]](https://openaccess.thecvf.com/content/CVPR2025/papers/Yang_PIAD_Pose_and_Illumination_agnostic_Anomaly_Detection_CVPR_2025_paper.pdf) [\[code\]](https://github.com/Kaichen-Yang/piad_baseline) ⭐ 14 | 🐛 1 | 🌐 Python | 📅 2025-11-11[\[data\]](https://kaichen-yang.github.io/piad/)
* M3-AD: Reflection-aware Multi-modal, Multi-category, and Multi-dimensional
  Benchmark and Framework for Industrial Anomaly Detection [\[2026\]](https://arxiv.org/abs/2603.00055)[\[code\]](https://github.com/Yanhui-Lee/M3-AD) ⭐ 14 | 🐛 1 | 📅 2026-02-10
* VID-AD: A Dataset for Image-Level Logical Anomaly Detection under Vision-Induced Distraction [\[2026\]](https://arxiv.org/abs/2603.13964)[\[data\]](https://github.com/nkthiroto/VID-AD) ⭐ 11 | 🐛 1 | 🌐 Python | 📅 2026-03-26
* Image-Pointcloud Fusion based Anomaly Detection using PD-REAL Dataset [\[2023\]](https://arxiv.org/abs/2311.04095)[\[data\]](https://github.com/Andy-cs008/PD-REAL) ⭐ 9 | 🐛 1 | 📅 2026-03-09
* HSS-IAD: A Heterogeneous Same-Sort Industrial Anomaly Detection Dataset [\[ICME 2025\]](https://arxiv.org/abs/2504.12689)[\[data\]](https://github.com/Qiqigeww/HSS-IAD-Dataset) ⭐ 9 | 🐛 1 | 📅 2025-12-16
* Catenary Insulator Defects Detection: A Dataset and an Unsupervised Baseline [\[TIM 2024\]](https://ieeexplore.ieee.org/abstract/document/10504848)[\[code\]](https://github.com/LightZH/Insulator-Defect-Detection) ⭐ 8 | 🐛 0 | 📅 2024-04-30
* 3D-ADAM: A Dataset for 3D Anomaly Detection in Advanced Manufacturing [\[2025\]](https://arxiv.org/abs/2507.07838)[\[data\]](https://huggingface.co/datasets/pmchard/3D-ADAM)[\[code\]](https://github.com/PaulMcHard/3D-ADAMBench) ⭐ 5 | 🐛 1 | 🌐 Python | 📅 2025-05-25
* (MPDD2)Anomaly detection for real-world industrial applications: benchmarking recent self-supervised and pretrained methods [\[ICUMT 2022\]](https://ieeexplore.ieee.org/abstract/document/9943437) [\[data\]](https://github.com/stepanje/MPDD2) ⭐ 3 | 🐛 0 | 📅 2024-07-24
* PeanutAD: A Real-World Dataset for Anomaly Detection in Agricultural Product Processing Line [\[2024\]](https://ieeexplore.ieee.org/abstract/document/10634679)[\[data\]](https://github.com/TCV0257/PeanutAD) ⭐ 2 | 🐛 1 | 📅 2024-06-03
* (MCBT) Manufacturing Complex Background Texture: a real-world industrial anomaly dataset for challenging textured backgrounds, introduced in When Textures Deceive: Weakly Supervised Industrial Anomaly Detection with Adapted-Loss [\[CVPR 2025 W (VAND)\]](https://openaccess.thecvf.com/content/CVPR2025W/VAND/papers/Nakkina_When_Textures_Deceive_Weakly_Supervised_Industrial_Anomaly_Detection_with_Adapted-Loss_CVPRW_2025_paper.pdf)[\[data\]](https://github.com/ganatma/AL-CycleGAN/tree/main/MCBT) ⭐ 2 | 🐛 0 | 🌐 Python | 📅 2026-04-16
* ISP-AD: a large-scale real-world dataset for advancing industrial anomaly detection with synthetic and real defects [\[JIMS 2026\]](https://link.springer.com/article/10.1007/s10845-025-02778-z)[\[code\]](https://github.com/p4ulk/isp-ad) ⭐ 0 | 🐛 0 | 🌐 Python | 📅 2026-07-10[\[data\]](https://zenodo.org/records/14911042)✨
* (NEU surface defect dataset)A noise robust method based on completed local binary patterns for hot-rolled steel strip surface defects [\[2013\]](https://www.sciencedirect.com/science/article/pii/S0169433213016437/pdfft?md5=478bf7f07bbf551a5d991048f9bc16e4\&pid=1-s2.0-S0169433213016437-main.pdf) [\[data\]](http://faculty.neu.edu.cn/songkechen/zh_CN/zdylm/263270/list/index.htm)
* (Steel defect dataset)Severstal: Steel Defect Detection [\[data 2019\]](https://www.kaggle.com/code/ekhtiar/resunet-a-baseline-on-tensorflow/notebook)
* (NanoTwice)Defect detection in SEM images of nanofibrous materials [\[TII 2016\]](https://re.public.polimi.it/bitstream/11311/1024586/1/anomaly_detection_sem.pdf) [\[data\]](http://web.mi.imati.cnr.it/ettore/NanoTwice)
* (GDXray)GDXray: The database of X-ray images for nondestructive testing [\[2015\]](http://dmery.sitios.ing.uc.cl/Prints/ISI-Journals/2015-JNDE-GDXray.pdf) [\[data\]](https://domingomery.ing.puc.cl/material/gdxray/)
* (PCBA-defect) A PCB Dataset for Defects Detection and Classification [\[2019\]](https://arxiv.org/abs/1901.08204)[\[data\]](https://www.kaggle.com/datasets/akhatova/pcb-defects)
* (Fabric dataset)Fabric inspection based on the Elo rating method [\[PR 2016\]](http://hub.hku.hk/bitstream/10722/229176/1/content.pdf)
* (KolektorSDD)Segmentation-based deep-learning approach for surface-defect detection [\[Journal of Intelligent Manufacturing\]](http://arxiv.org/pdf/1903.08536) [\[data\]](https://www.vicos.si/resources/kolektorsdd/)
* (KolektorSDD2)Mixed supervision for surface-defect detection: From weakly to fully supervised learning [\[Computers in Industry 2021\]](https://arxiv.org/pdf/2104.06064.pdf) [\[data\]](https://www.vicos.si/resources/kolektorsdd2/)
* SensumSODF-dataset: Detection of surface defects on pharmaceutical solid oral dosage forms with convolutional neural networks[\[Neural Computing and Applications 2021\]](https://link.springer.com/article/10.1007/s00521-021-06397-6)[\[data\]](https://www.sensum.eu/sensumsodf-dataset/)
* (RSDD)A hierarchical extractor-based visual rail surface inspection system [\[2017\]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=\&arnumber=8063875)
* (Eyecandies)The Eyecandies Dataset for Unsupervised Multimodal Anomaly Detection and Localization [\[ACCV 2022\]](https://arxiv.org/pdf/2210.04570.pdf) [\[data\]](https://eyecan-ai.github.io/eyecandies/)
* (MVTec AD)MVTec AD: A comprehensive real-world dataset for unsupervised anomaly detection [\[CVPR 2019\]](https://openaccess.thecvf.com/content_CVPR_2019/html/Bergmann_MVTec_AD_--_A_Comprehensive_Real-World_Dataset_for_Unsupervised_Anomaly_CVPR_2019_paper.html) [\[IJCV 2021\]](https://link.springer.com/content/pdf/10.1007/s11263-020-01400-4.pdf) [\[data\]](https://www.mvtec.com/company/research/datasets/mvtec-ad)✨✨✨
* (MVTec 3D-AD)The mvtec 3d-ad dataset for unsupervised 3d anomaly detection and localization [\[VISAPP 2021\]](https://arxiv.org/pdf/2112.09045.pdf) [\[data\]](https://www.mvtec.com/company/research/datasets/mvtec-3d-ad)✨✨
* (MVTec LOCO-AD)Beyond Dents and Scratches: Logical Constraints in Unsupervised Anomaly Detection and Localization [\[IJCV 2022\]](https://link.springer.com/content/pdf/10.1007/s11263-022-01578-9.pdf) [\[data\]](https://www.mvtec.com/company/research/datasets/mvtec-loco)✨✨✨
* (BTAD)VT-ADL: A vision transformer network for image anomaly detection and localization [\[2021\]](http://arxiv.org/pdf/2104.10036) [\[data\]](http://avires.dimi.uniud.it/papers/btad/btad.zip)
* (DAGM)DAGM dataset [\[data 2007\]](https://www.kaggle.com/datasets/mhskjelvareid/dagm-2007-competition-dataset-optical-inspection)
* (MIAD)Miad:A maintenance inspection dataset for unsupervised anomaly detection [\[2022\]](https://arxiv.org/abs/2211.13968) [\[data\]](https://miad-2022.github.io/)✨✨
* CVPR 1st workshop on Vision-based InduStrial InspectiON [\[homepage\]](https://vision-based-industrial-inspection.github.io/cvpr-2023/) [\[data\]](https://drive.google.com/drive/folders/1TVp_UXJuXudqhC2L3ZKyIDcmQ_2O3JVi)
* VISION Datasets: A Benchmark for Vision-based InduStrial InspectiON [\[2023\]](https://arxiv.org/abs/2306.07890) [\[data\]](https://huggingface.co/datasets/VISION-Workshop/VISION-Datasets)✨✨✨
* CrashCar101: Procedural Generation for Damage Assessment [\[WACV 2024\]](https://crashcar.compute.dtu.dk/static/2435.pdf)[\[data\]](https://crashcar.compute.dtu.dk/)
* (DTD-Synthetic) Zero-shot versus Many-shot: Unsupervised Texture Anomaly Detection [\[WACV 2023\]](https://ieeexplore.ieee.org/document/10030870)[\[data\]](https://drive.google.com/drive/folders/10OyPzvI3H6llCZBxKxFlKWt1Pw1tkMK1)
* IPAD: Industrial Process Anomaly Detection Dataset [\[2024\]](https://arxiv.org/abs/2404.15033)[\[data\]](https://ljf1113.github.io/IPAD_VAD/)
* Texture-AD: An Anomaly Detection Dataset and Benchmark for Real Algorithm Development[\[2024\]](https://arxiv.org/abs/2409.06367)[\[data\]](https://huggingface.co/datasets/texture-ad/Texture-AD-Benchmark)
* CableInspect-AD: An Expert-Annotated Anomaly Detection Dataset [\[NeurIPS 2024\]](https://arxiv.org/abs/2409.20353)[\[data\]](https://mila-iqia.github.io/cableinspect-ad/)
* AD3: Introducing a score for Anomaly Detection Dataset Difficulty assessment using VIADUCT dataset [\[ECCV 2024\]](https://eccv.ecva.net/virtual/2024/poster/2287)[\[data\]](https://fordatis.fraunhofer.de/handle/fordatis/363.2)
* MANTA: A Large-Scale Multi-View and Visual-Text Anomaly Detection Dataset for Tiny Objects [\[2024\]](https://arxiv.org/abs/2412.04867)[\[data\]](https://grainnet.github.io/MANTA)✨✨
* Are Anomaly Scores Telling the Whole Story? A Benchmark for Multilevel Anomaly Detection [\[2024\]](https://arxiv.org/abs/2411.14515)
* Towards Zero-Shot Anomaly Detection and Reasoning with Multimodal Large Language Models [\[2025\]](https://arxiv.org/abs/2502.07601)[\[data\]](https://xujiacong.github.io/Anomaly-OV/)
* Real-IAD D<sup>3</sup>: A Real-World 2D/Pseudo-3D/3D Dataset for Industrial Anomaly Detection [\[CVPR 2025\]](https://arxiv.org/abs/2504.14221)✨✨
* The MVTec AD 2 Dataset: Advanced Scenarios for Unsupervised Anomaly Detection [\[CVPR2025W Challenge\]](https://sites.google.com/view/vand30cvpr2025/challenge)[\[paper\]](https://arxiv.org/abs/2503.21622)[\[data\]](https://www.mvtec.com/company/research/datasets/mvtec-ad-2)✨✨
* Towards Visual Discrimination and Reasoning of Real-World Physical Dynamics: Physics-Grounded Anomaly Detection [\[CVPR 2025\]](https://arxiv.org/abs/2503.03562)✨✨
* CXR-AD: Component X-ray Image Dataset for Industrial Anomaly Detection [\[2025\]](https://arxiv.org/abs/2505.03412)
* Visual Anomaly Detection under Complex View-Illumination Interplay: A Large-Scale Benchmark [\[2025\]](https://arxiv.org/abs/2505.10996)[\[data\]](https://hustcyq.github.io/M2AD/#dataset)✨✨
* Unsupervised Anomaly Segmentation at High Resolution with Patch-Divide-and-Conquer and Self-ensembling [\[ECCVW 2025\]](https://link.springer.com/chapter/10.1007/978-3-031-91856-8_4)[\[data\]](https://www.kaggle.com/datasets/hendrikmeininger/bp-bearings-and-plates-dataset)
* Kaputt: A Large-Scale Dataset for Visual Defect Detection [\[ICCV 2025\]](https://arxiv.org/abs/2510.05903)[\[data\]](https://www.kaputt-dataset.com/)
* Real-IAD Variety: Pushing Industrial Anomaly Detection Dataset to a Modern Era [\[PR 2026\]](https://arxiv.org/abs/2511.00540)[\[data\]](https://realiad4ad.github.io/Real-IAD-Variety)
* Towards High-Resolution 3D Anomaly Detection: A Scalable Dataset and Real-Time Framework for Subtle Industrial Defects [\[AAAI 2026 oral\]](https://arxiv.org/abs/2507.07435)[\[data\]](https://hustcyq.github.io/MiniShift-Simple3D/)
* ADNet: A Large-Scale and Extensible Multi-Domain Benchmark for Anomaly Detection Across 380 Real-World Categories [\[2025\]](https://arxiv.org/abs/2511.20169)[\[data is comming\]](https://grainnet.github.io/ADNet)
* SiM3D: Single-instance Multiview Multimodal and Multisetup 3D Anomaly Detection Benchmark [\[ICCV 2025\]](https://arxiv.org/abs/2506.21549)[\[data\]](https://alex-costanzino.github.io/SiM3D/)
* Toward Long-Tailed Online Anomaly Detection through Class-Agnostic Concepts [\[ICCV 2025\]](https://arxiv.org/abs/2507.16946)[\[data\]](https://zenodo.org/records/16283853)
* Robust AD: A Real World Benchmark Dataset For Robustness in Industrial Anomaly Detection [\[CVPRW 2025\]](https://openaccess.thecvf.com/content/CVPR2025W/VAND/papers/Pemula_Robust_AD_A_Real_World_Benchmark_Dataset_For_Robustness_in_CVPRW_2025_paper.pdf)[\[data\]](https://huggingface.co/datasets/AmazonScience/RobustAD)
* MMR-AD: A Large-Scale Multimodal Dataset for Benchmarking General Anomaly Detection with Multimodal Large Language Models [\[CVPR 2026\]](https://arxiv.org/abs/2604.10971)[\[code\]](https://xcyao00.github.io/MMR-AD)
* ADSeeker: A Knowledge-Grounded Reasoning Framework for Industry Anomaly Detection and Reasoning [\[CVPR 2026\]](https://arxiv.org/abs/2508.03088)
* Omni-AD: A Large-scale and Versatile Benchmark for Industrial Anomaly Detection [\[CVPR 2026\]](https://cvpr.thecvf.com/virtual/2026/poster/39148)
* Towards Open-Vocabulary Industrial Defect Understanding with a Large-Scale Multimodal Dataset [\[CVPR 2026\]](https://arxiv.org/abs/2512.24160)[\[data\]](https://ninaneon.github.io/projectpage)
* Real-IAD MVN: A Multi-View Normal Vector Dataset and Benchmark for High-Fidelity Industrial Anomaly Detection [\[CVPR 2026\]](https://arxiv.org/abs/2605.07149)

## BibTex Citation

If you find this paper and repository useful, please cite our paper☺️.

```
@article{liu2024deep,
  title={Deep industrial image anomaly detection: A survey},
  author={Liu, Jiaqi and Xie, Guoyang and Wang, Jinbao and Li, Shangnian and Wang, Chengjie and Zheng, Feng and Jin, Yaochu},
  journal={Machine Intelligence Research},
  volume={21},
  number={1},
  pages={104--135},
  year={2024},
  publisher={Springer}
}

@article{xie2024iad,
  title={Im-iad: Industrial image anomaly detection benchmark in manufacturing},
  author={Xie, Guoyang and Wang, Jinbao and Liu, Jiaqi and Lyu, Jiayi and Liu, Yong and Wang, Chengjie and Zheng, Feng and Jin, Yaochu},
  journal={IEEE Transactions on Cybernetics},
  year={2024},
  publisher={IEEE}
}

@article{jiang2022survey,
  title={A survey of visual sensory anomaly detection},
  author={Jiang, Xi and Xie, Guoyang and Wang, Jinbao and Liu, Yong and Wang, Chengjie and Zheng, Feng and Jin, Yaochu},
  journal={arXiv preprint arXiv:2202.07006},
  year={2022}
}
```

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=M-3LAB/awesome-industrial-anomaly-detection\&type=Date)](https://star-history.com/#M-3LAB/awesome-industrial-anomaly-detection\&Date)

***

> _Enhansomed by [enhansome](https://github.com/enhansome) on 2026-09-01._
