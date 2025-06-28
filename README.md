# Awesome-Papers-for-Medical-Image

## ✨✨✨ Tumor Segmentation
### 2025
| Title | Abbreviation | Venue | Code | aaa |
|-----|-----|-----|-----|-----|
| [Advancing Generalizable Tumor Segmentation with Anomaly-Aware Open-Vocabulary Attention Maps and Frozen Foundation Diffusion Models](https://openaccess.thecvf.com//content/CVPR2025/papers/Jiang_Advancing_Generalizable_Tumor_Segmentation_with_Anomaly-Aware_Open-Vocabulary_Attention_Maps_and_CVPR_2025_paper.pdf) | DiffuGTS | CVPR 2025 | [code](https://github.com/Yankai96/DiffuGTS) | text-to-image attention maps |







## ✨✨✨ Medical Image Synthesis
### 2025
| Title | Abbreviation | Venue | Code | aaa |
|-----|-----|-----|-----|-----|
| [Toward general text-guided multimodal brain MRI synthesis for diagnosis and medical image analysis](https://www.cell.com/cell-reports-medicine/fulltext/S2666-3791(25)00255-1) | TUMSyn | Cell Reports Medicine 2025 | -- | step1: Clip预训练metadata step2: 注意力+隐式解码器合成影像 |




## ✨✨✨ Tumor Synthesis and Segmentation
### 2025
| Title | Abbreviation | Venue | Code | aaa |
|-----|-----|-----|-----|-----|
| [Medical Image Synthesis via Fine-Grained Image-Text Alignment and Anatomy-Pathology Prompting](https://papers.miccai.org/miccai-2024/paper/3619_paper.pdf) | Chen et al. | MICAAI 2024 | -- | 文本图像，给定词汇gpt生成 |
| [From Pixel to Cancer:Cellular Automata in Computed Tomography](https://link.springer.com/chapter/10.1007/978-3-031-72378-0_4) | Pixel2Cancer | MICAAI 2024 | [code](https://github.com/MrGiovanni/Pixel2Cancer) | 肿瘤图像合成分割（Cellular Automata） |
| [Towards Generalizable Tumor Synthesis](https://openaccess.thecvf.com/content/CVPR2024/papers/Chen_Towards_Generalizable_Tumor_Synthesis_CVPR_2024_paper.pdf) | DiffTumor | CVPR 2024 | [code](https://github.com/MrGiovanni/DiffTumor) | 肿瘤图像合成分割（Diffusion） |
| [Label-Free Liver Tumor Segmentation](https://openaccess.thecvf.com/content/CVPR2023/papers/Hu_Label-Free_Liver_Tumor_Segmentation_CVPR_2023_paper.pdf) | SyntheticTumors | CVPR 2023 | [code](https://github.com/MrGiovanni/SyntheticTumors) | 肝细胞瘤图像合成分割（图像算子，handcrafted） |
| [FreeTumor Advance Tumor Segmentation via Large-Scale Tumor Synthesis](https://arxiv.org/abs/2406.01264) | FreeTumor | ArixV 2025 | [code](https://github.com/Luffy03/FreeTumor) | 肿瘤图像合成分割 |



## ✨✨✨Multi-source Annotations Medical Image Segmentation
### 2025
| Title | Abbreviation | Venue | Code | aaa |
|-----|-----|-----|-----|-----|
| [Learning robust medical image segmentation from multi-source annotations](https://www.sciencedirect.com/science/article/pii/S1361841525000374) | UMA-Net | MedIA 2025 | [code](https://github.com/wangjin2945/UMA-Net) | 对多源标签进行不确定性评估，根据一致性和可靠性将图像样本分为高低质量，对损失加权 |


## ✨✨✨Semi-Supervised Medical Image Segmentation
### 2025
| Title | Abbreviation | Venue | Code | aaa |
|-----|-----|-----|-----|-----|
| [Leveraging labelled data knowledge: A cooperative rectification learning network for semi-supervised 3D medical image segmentation](https://www.sciencedirect.com/science/article/pii/S136184152500009X) | CRLN | MedIA 2025 | [code](https://github.com/Yaan-Wang/CRLN) | memory bank式的矫正伪标签 |



### 2024
| Title | Abbreviation | Venue | Code | aaa |
|-----|-----|-----|-----|-----|
| [Adaptive Bidirectional Displacement for Semi-Supervised Medical Image Segmentation](https://openaccess.thecvf.com/content/CVPR2024/papers/Chi_Adaptive_Bidirectional_Displacement_for_Semi-Supervised_Medical_Image_Segmentation_CVPR_2024_paper.pdf) | ABD | CVPR 2024 | [code](https://github.com/chy-upc/ABD) | 预测值作为可靠性度量，将无标签样本的不可靠区域替换，将有标签样本的不可靠区域互换 |
| [PH-Net: Semi-Supervised Breast Lesion Segmentation via Patch-wise Hardness](https://openaccess.thecvf.com/content/CVPR2024/papers/Jiang_PH-Net_Semi-Supervised_Breast_Lesion_Segmentation_via_Patch-wise_Hardness_CVPR_2024_paper.pdf) | PH-Net | CVPR 2024 | [code](https://github.com/jjjsyyy/PH-Net) | 熵度量patch不确定性，topk挑选不确定patch对比学习 |
| [Effective Semi-Supervised Medical Image Segmentation with Probabilistic Representations and Prototype Learning](https://ieeexplore.ieee.org/document/10723767) | PPC | TMI 2024 | [code](https://github.com/IsYuchenYuan/PPC) | 有限的数据，模型难以表征不确定性，对数据进行不确定性建模 |
| [Mutual learning with reliable pseudo label for semi-supervised medical image segmentation](https://www.sciencedirect.com/science/article/pii/S1361841524000367) | Mutual learning with reliable pseudo label | MedIA 2024 | [code](https://github.com/Jiawei0o0/mutual-learning-with-reliable-pseudo-labels) |
| [FRCNet: Frequency and Region Consistency for Semi-supervised Medical Image Segmentation](https://papers.miccai.org/miccai-2024/paper/0245_paper.pdf) | FRCNet | MICCAI 2024 | [code](https://github.com/NKUhealong/FRCNet) |
| [VCLIPSeg: Voxel-wise CLIP-Enhanced model for Semi-Supervised Medical Image Segmentation](https://papers.miccai.org/miccai-2024/paper/1949_paper.pdf) | VCLIPSeg | MICCAI 2024 | -- |


### 2023
| Title | Abbreviation | Venue | Code |
|-----|-----|-----|-----|
| [Co-training with High-Confidence Pseudo Labels for Semi-supervised Medical Image Segmentation](https://www.ijcai.org/proceedings/2023/0467.pdf) | UCMT | IJCAI 2023 | [code](https://github.com/Senyh/UCMT) |

### 2022
| Title | Abbreviation | Venue | Code | Conclusion |
|-----|-----|-----|-----|-----|
| [Inconsistency-Aware Uncertainty Estimation for Semi-Supervised Medical Image Segmentation](https://ieeexplore.ieee.org/document/9558816) | CoraNet | TMI 2022 | [code](https://github.com/koncle/CoraNet) | 作者观察到给误分类施加不同程度的惩罚，会使模型给出不同的预测。基于此现象，提出一种新的不确定性估计。基于给出的确定性和不确定性，分别计算分割损失。迭代训练。 |
| [SimCVD: Simple Contrastive Voxel-Wise Representation Distillation for Semi-Supervised Medical Image Segmentation](https://ieeexplore.ieee.org/document/9740182) | SimCVD | TMI 2022 | -- | -- |


### 2021
| Title | Abbreviation | Venue | Code |
|-----|-----|-----|-----|
| [Transformation-Consistent_Self-Ensembling_Model_for_Semisupervised_Medical_Image_Segmentation](https://ieeexplore.ieee.org/document/9104928) | TCSM_v2 | TNNLS 2021 | -- |
| [Semi-supervised Left Atrium Segmentation with Mutual Consistency Training](https://link.springer.com/chapter/10.1007/978-3-030-87196-3_28) | MC-Net | MICCAI 2021 | -- |
| [Semi-supervised Medical Image Segmentation through Dual-task Consistency](https://ojs.aaai.org/index.php/AAAI/article/view/17066) | DTC | AAAI 2021 | [code](https://github.com/HiLab-git/DTC) |
| [Semi-Supervised Semantic Segmentation with Cross Pseudo Supervision](https://openaccess.thecvf.com/content/CVPR2021/papers/Chen_Semi-Supervised_Semantic_Segmentation_With_Cross_Pseudo_Supervision_CVPR_2021_paper.pdf) | CPS | CVPR 2021 | -- |




### 2020
| Title | Abbreviation | Venue | Code |
|-----|-----|-----|-----|
| [Semi-Supervised Semantic Segmentation with Cross-Consistency Training](https://openaccess.thecvf.com/content_CVPR_2020/papers/Ouali_Semi-Supervised_Semantic_Segmentation_With_Cross-Consistency_Training_CVPR_2020_paper.pdf) | CCT | CVPR 2020 | [code](https://github.com/yassouali/CCT) |

### 2019
| Title | Abbreviation | Venue | Code |
|-----|-----|-----|-----|
| [Uncertainty-Aware Self-ensembling Model for Semi-supervised 3D Left Atrium Segmentation](https://link.springer.com/chapter/10.1007/978-3-030-32245-8_67) | UA-MT | MICCAI 2019 | [code](https://github.com/yulequan/UA-MT) |


## ✨✨✨One-shot Medical Image Segmentation
### 2024
| Title | Abbreviation | Venue | Code | aaa |
|-----|-----|-----|-----|-----|
| [Bi-level Learning of Task-Specific Decoders for Joint Registration and One-Shot Medical Image Segmentation](https://openaccess.thecvf.com/content/CVPR2024/papers/Fan_Bi-level_Learning_of_Task-Specific_Decoders_for_Joint_Registration_and_One-Shot_CVPR_2024_paper.pdf) | Bi-JROS | CVPR 2024 | [code](https://github.com/Coradlut/Bi-JROS) | 双任务之间的梯度响应 |

## ✨✨✨Barely-Supervised Medical Image Segmentation
### 2024
| Title | Abbreviation | Venue | Code |
|-----|-----|-----|-----|
| [Barely-supervised Brain Tumor Segmentation via Employing Segment Anything Model](https://ieeexplore.ieee.org/document/10491099) | BarelySAM | TCSVT 2024 | -- |
| [FM-ABS: Promptable Foundation Model Drives Active Barely Supervised Learning for 3D Medical Image Segmentation](https://papers.miccai.org/miccai-2024/paper/0050_paper.pdf) | FM-ABS | MICCAI 2024 | [code](https://github.com/lemoshu/FM-ABS) |
| [Few Slices Suffice Multi-faceted Consistency Learning with Active Cross-Annotation for Barely-Supervised 3D Medical Image Segmentation](https://link.springer.com/chapter/10.1007/978-3-031-72120-5_27) | MF-ConS | MICCAI 2024 | -- |
| [Rethinking Barely-Supervised Volumetric Medical Image Segmentation from an Unsupervised Domain Adaptation Perspective](https://arxiv.org/abs/2405.09777) | BvA | arXiv 2024 | [code](https://github.com/Senyh/BvA) |
| [Self-paced Sample Selection for Barely-Supervised Medical Image Segmentation](https://link.springer.com/chapter/10.1007/978-3-031-72114-4_56) | SPSS | MICCAI 2024 | [code](https://github.com/SuuuJM/SPSS) |


### 2023
| Title | Abbreviation | Venue | Code |
|-----|-----|-----|-----|
| [Compete to Win: Enhancing Pseudo Labels for Barely-Supervised Medical Image Segmentation](https://ieeexplore.ieee.org/document/10132102) | ComWin | TMI 2023 | [code](https://github.com/Huiimin5/comwin) |
| [PLN: Parasitic-Like Network for Barely Supervised Medical Image Segmentation](https://ieeexplore.ieee.org/document/9906305) | PLN | TMI 2023 | [code](https://github.com/ShumengLI/PLN) |
| [Orthogonal Annotation Benefits Barely-supervised Medical Image Segmentation](https://openaccess.thecvf.com/content/CVPR2023/papers/Cai_Orthogonal_Annotation_Benefits_Barely-Supervised_Medical_Image_Segmentation_CVPR_2023_paper.pdf) | DeSCO | CVPR 2023 | [code](https://github.com/HengCai-NJU/DeSCO) |

### 2022
| Title | Abbreviation | Venue | Code |
|-----|-----|-----|-----|
| [Calibrating Label Distribution for Class-Imbalanced Barely-Supervised Knee Segmentation](https://link.springer.com/chapter/10.1007/978-3-031-16452-1_11) | CLD | MICCAI 2022 | [code](https://github.com/xmed-lab/CLD-Semi) |





## ✨✨✨Referring Medical Image Segmentation
### 2024
| Title | Abbreviation | Venue | Code |
|-----|-----|-----|-----|
| [CausalCLIPSeg: Unlocking CLIP’s Potential in Referring Medical Image Segmentation with Causal Intervention](https://papers.miccai.org/miccai-2024/paper/3127_paper.pdf) | CausalCLIPSeg | MICCAI 2024 | [code](https://github.com/WUTCM-Lab/CausalCLIPSeg) |

## ✨✨✨Weakly-supervised Medical Image Segmentation
### 2024
| Title | Abbreviation | Venue | Code |
|-----|-----|-----|-----|
| [A Weakly-supervised Multi-lesion Segmentation Framework Based on Target-level Incomplete Annotations](https://papers.miccai.org/miccai-2024/paper/1747_paper.pdf) | CD_TIA | MICCAI 2024 | [code](https://github.com/HeyJGJu/CD_TIA) |
| [A Bayesian Approach to Weakly-supervised Laparoscopic Image Segmentation](https://papers.miccai.org/miccai-2024/paper/0219_paper.pdf) | Bayesian_WSS | MICCAI 2024 | [code](https://github.com/MoriLabNU/Bayesian_WSS) |

## ✨✨✨Universal Model
### 2023
| Title | Abbreviation | Venue | Code |
|-----|-----|-----|-----|
| [CLIP-Driven Universal Model for Organ Segmentation and Tumor Detection](https://openaccess.thecvf.com/content/ICCV2023/papers/Liu_CLIP-Driven_Universal_Model_for_Organ_Segmentation_and_Tumor_Detection_ICCV_2023_paper.pdf) | CLIP-Driven_Universal_Model | ICCV 2023 | [code](https://github.com/ljwztc/CLIP-Driven-Universal-Model) |

## ✨✨✨Medical Image Classification
### 2024
| Title | Abbreviation | Venue | Code |
|-----|-----|-----|-----|
| [Confidence Matters: Enhancing Medical Image Classification Through Uncertainty-Driven Contrastive Self-Distillation](https://papers.miccai.org/miccai-2024/paper/1765_paper.pdf) | UDCD | MICCAI 2024 | [code](https://github.com/philsaurabh/UDCD_MICCAI) |

## ✨✨✨Pre-trained Framework
### 2024
| Title | Abbreviation | Venue | Code |
|-----|-----|-----|-----|
| [VoCo: A Simple-yet-Effective Volume Contrastive Learning Framework for 3D Medical Image Analysis](https://openaccess.thecvf.com/content/CVPR2024/papers/Wu_VoCo_A_Simple-yet-Effective_Volume_Contrastive_Learning_Framework_for_3D_Medical_CVPR_2024_paper.pdf) | VoCo | CVPR 2024 | [code](https://github.com/Luffy03/VoCo) |
| [Anatomical Positional Embeddings](https://papers.miccai.org/miccai-2024/paper/3539_paper.pdf) | APE | MICCAI 2024 | [code](https://github.com/mishgon/ape) |


### 2023
| Title | Abbreviation | Venue | Code |
|-----|-----|-----|-----|
| [A Unified Visual Information Preservation Framework for Self-supervised Pre-Training in Medical Image Analysis](https://ieeexplore.ieee.org/document/10005161) | PCRLv2 | TPAMI 2023 | [code](https://github.com/RL4M/PCRLv2) |
| [vox2vec: A Framework for Self-supervised Contrastive Learning of Voxel-level Representations in Medical Images](https://link.springer.com/chapter/10.1007/978-3-031-43907-0_58) | vox2vec | MICCAI 2023 | [code](https://github.com/mishgon/vox2vec) |
