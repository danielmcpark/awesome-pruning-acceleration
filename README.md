# awesome-pruning-acceleration [![Awesome](https://awesome.re/badge.svg)](https://awesome.re)
Hello visitors, I have been interested in an efficient deep neural network design, such as pruning, AutoML, quantization, and focused on the knowledge distillation for successful network generalization. This page organizes for pruning.

## History

- [2021 years](#2021)
- [2020 years](#2020)
- [2019 years](#2019)
- [2018 years](#2018)
- [2017 years](#2017)
- [2016 years](#2016)
- [2015 years](#2015)

### 2021
|   Title  | Issue | Release |
| :--------| :---: | :-----: |
| [Robust Pruning at Initialization](https://arxiv.org/pdf/2002.08797.pdf) | ICLR | - |

### 2020
|   Title  | Issue | Release |
| :--------| :---: | :-----: |
| [What is the State of Neural Network Pruning?](https://proceedings.mlsys.org/paper/2020/file/d2ddea18f00665ce8623e36bd4e3c7c5-Paper.pdf) | MLSys | [GitHub](https://github.com/jjgo/shrinkbench) |
| [The Generalization-Stability Tradeoff In Neural Network Pruning](https://papers.nips.cc/paper/2020/file/ef2ee09ea9551de88bc11fd7eeea93b0-Paper.pdf) | NeurIPS | [GitHub](https://github.com/bbartoldson/GeneralizationStabilityTradeoff) |
| [Position-based Scaled Gradient for Model Quantization and Pruning](https://papers.nips.cc/paper/2020/file/eb1e78328c46506b46a4ac4a1e378b91-Paper.pdf) | NeurIPS | [GitHub](https://github.com/Jangho-Kim/PSG-pytorch) |
| [Sanity-Checking Pruning Methods: Random Tickets can Win the Jackpot](https://papers.nips.cc/paper/2020/file/eae27d77ca20db309e056e3d2dcd7d69-Paper.pdf) | NeurIPS | [GitHub](https://github.com/JingtongSu/sanity-checking-pruning) |
| [Movement Pruning: Adaptive Sparsity by Fine-Tuning](https://papers.nips.cc/paper/2020/file/eae15aabaa768ae4a5993a8a4f4fa6e4-Paper.pdf) | NeurIPS | [GitHub](https://github.com/huggingface/block_movement_pruning) |
| [HYDRA: Pruning Adversarially Robust Neural Networks](https://papers.nips.cc/paper/2020/file/e3a72c791a69f87b05ea7742e04430ed-Paper.pdf) | NeurIPS | [GitHub](https://github.com/inspire-group/hydra) |
| [Pruning Filter in Filter](https://papers.nips.cc/paper/2020/file/ccb1d45fb76f7c5a0bf619f979c6cf36-Paper.pdf) | NeurIPS | [GitHub](https://github.com/fxmeng/Pruning-Filter-in-Filter) |
| [Storage Efficient and Dynamic Flexible Runtime Channel Pruning via Deep Reinforcement Learning](https://papers.nips.cc/paper/2020/file/a914ecef9c12ffdb9bede64bb703d877-Paper.pdf) | NeurIPS | [GitHub](https://github.com/jianda-chen/static_dynamic_rl_pruning) |
| [Directional Pruning of Deep Neural Networks](https://papers.nips.cc/paper/2020/file/a09e75c5c86a7bf6582d2b4d75aad615-Paper.pdf) | NeurIPS | [GitHub](https://github.com/donlan2710/gRDA-Optimizer/tree/master/directional_pruning) |
| [SCOP: Scientific Control for Reliable Neural Network Pruning](https://papers.nips.cc/paper/2020/file/7bcdf75ad237b8e02e301f4091fb6bc8-Paper.pdf) | NeurIPS | - |
| [Neuron-level Structured Pruning using Polarization Regularizer](https://papers.nips.cc/paper/2020/file/703957b6dd9e3a7980e040bee50ded65-Paper.pdf) | NeurIPS | [GitHub](https://github.com/polarizationpruning/PolarizationPruning) |
| [Pruning neural networks without any data by iteratively conserving synaptic flow](https://papers.nips.cc/paper/2020/hash/46a4378f835dc8040c8057beb6a2da52-Abstract.html) | NeurIPS | [GitHub](https://github.com/ganguli-lab/Synaptic-Flow) |
| [Neuron Merging: Compensating for Pruned Neurons](https://papers.nips.cc/paper/2020/file/0678ca2eae02d542cc931e81b74de122-Paper.pdf) | NeurIPS | [GitHub](https://github.com/danielmcpark/neuron-merging) |
| [Filter Pruning and Re-Initialization via Latent Space Clustering](https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=6287639) | IEEE Access | - |
| [TF-NAS: Rethinking Three Search Freedoms of Latency-Constrained Differentiable Neural Architecture Search](https://arxiv.org/abs/2008.05314) | ECCV | [GitHub](https://github.com/AberHu/TF-NAS) |
| [Differentiable Joint Pruning and Quantization for Hardware Efficiency](https://arxiv.org/pdf/2007.10463.pdf) | ECCV | - |
| [DA-NAS: Data Adapted Pruning for Efficient Neural Architecture Search](https://arxiv.org/pdf/2003.12563.pdf) | ECCV | - |
| [Accelerating CNN Training by Pruning Activation Gradients](https://arxiv.org/pdf/1908.00173.pdf) | ECCV | - |
| [DHP: Differentiable Meta Pruning via HyperNetworks](https://arxiv.org/pdf/2003.13683.pdf) | ECCV | [GitHub](https://github.com/ofsoundof/dhp) |
| [DSA: More Efficient Budgeted Pruning via Differentiable Sparsity Allocation](https://arxiv.org/pdf/2004.02164.pdf) | ECCV | [GitHub](https://github.com/walkerning/differentiable-sparsity-allocation) |
| [EagleEye: Fast Sub-net Evaluation for Efficient Neural Network Pruning](https://arxiv.org/abs/2007.02491) | ECCV | [GitHub](https://github.com/anonymous47823493/EagleEye) |
| [PCONV: The Missing but Desirable Sparsity in DNN Weight Pruning for Real-time Execution on Mobile Devices](https://arxiv.org/pdf/1909.05073.pdf) | AAAI | - |
| [Dynamic Network Pruning with Interpretable Layerwise Channel Selection]() | AAAI | - |
| [Reborn Filters: Pruning Convolutional Neural Networks with Limited Data](https://aaai.org/Papers/AAAI/2020GB/AAAI-TangY.1279.pdf) | AAAI | - |
| [Channel Pruning Guided by Classification Loss and Feature Importance](https://arxiv.org/pdf/2003.06757.pdf) | AAAI | - |
| [Pruning from Scratch](https://www.aaai.org/Papers/AAAI/2020GB/AAAI-WangY.403.pdf) | AAAI | - |
| [DropNet: Reducing Neural Network Complexity via Iterative Pruning](https://proceedings.icml.cc/static/paper_files/icml/2020/2026-Paper.pdf) | ICML | [GitHub](https://github.com/tanchongmin/DropNet) |
| [Operation-Aware Soft Channel Pruning using Differentiable Masks](https://proceedings.icml.cc/static/paper_files/icml/2020/1485-Paper.pdf) | ICML | - |
| [Group Sparsity: The Hinge Between Filter Pruning and Decomposition for Network Compression](https://openaccess.thecvf.com/content_CVPR_2020/papers/Li_Group_Sparsity_The_Hinge_Between_Filter_Pruning_and_Decomposition_for_CVPR_2020_paper.pdf) | CVPR | [GitHub](https://github.com/ofsoundof/group_sparsity)|
| [APQ: Joint Search for Network Architecture, Pruning and Quantization Policy](https://openaccess.thecvf.com/content_CVPR_2020/papers/Wang_APQ_Joint_Search_for_Network_Architecture_Pruning_and_Quantization_Policy_CVPR_2020_paper.pdf) | CVPR | - |
| [Learning Filter Pruning Criteria for Deep Convolutional Neural Networks Acceleration](https://openaccess.thecvf.com/content_CVPR_2020/papers/He_Learning_Filter_Pruning_Criteria_for_Deep_Convolutional_Neural_Networks_Acceleration_CVPR_2020_paper.pdf) | CVPR | - |
| [Structured Compression by Weight Encryption for Unstructured Pruning and Quantization](https://openaccess.thecvf.com/content_CVPR_2020/papers/Kwon_Structured_Compression_by_Weight_Encryption_for_Unstructured_Pruning_and_Quantization_CVPR_2020_paper.pdf) | CVPR | - |
| [Multi-Dimensional Pruning: A Unified Framework for Model Compression](https://openaccess.thecvf.com/content_CVPR_2020/papers/Guo_Multi-Dimensional_Pruning_A_Unified_Framework_for_Model_Compression_CVPR_2020_paper.pdf) | CVPR | - |
| [DMCP: Differentiable Markov Channel Pruning for Neural Networks](https://openaccess.thecvf.com/content_CVPR_2020/papers/Guo_DMCP_Differentiable_Markov_Channel_Pruning_for_Neural_Networks_CVPR_2020_paper.pdf) | CVPR | - |
| [HRank: Filter Pruning using High-Rank Feature Map](https://openaccess.thecvf.com/content_CVPR_2020/papers/Lin_HRank_Filter_Pruning_Using_High-Rank_Feature_Map_CVPR_2020_paper.pdf) | CVPR | [GitHub](https://github.com/lmbxmu/HRank) |
| [Neural Network Pruning with Residual-Connections and Limited-Data](https://openaccess.thecvf.com/content_CVPR_2020/papers/Luo_Neural_Network_Pruning_With_Residual-Connections_and_Limited-Data_CVPR_2020_paper.pdf) | CVPR | - |
| [Picking Winning Tickets Before Training by Preserving Gradient Flow](https://openreview.net/forum?id=SkgsACVKPH) | ICLR | [GitHub](https://github.com/alecwangcq/GraSP) |
| [Provable Filter Pruning for Efficient Neural Networks](https://openreview.net/pdf?id=BJxkOlSYDH) | ICLR | [GitHub](https://github.com/lucaslie/provable_pruning) |
| [Data-Independent Neural Pruning via Coresets](https://openreview.net/pdf?id=H1gmHaEKwB) | ICLR | - |
| [Lookahead: A Far-sighted Alternative of Magnitude-based Pruning](https://openreview.net/pdf?id=ryl3ygHYDB) | ICLR | [GitHub](https://github.com/alinlab/lookahead_pruning) |
| [Dynamic Model Pruning with Feedback](https://openreview.net/pdf?id=SJem8lSFwB) | ICLR | - |
| [One-shot Pruning of Recurrent Neural Neworks by Jacobian Spectrum Evaluation](https://openreview.net/pdf?id=r1e9GCNKvH) | ICLR | - |
| [A Signal Propagation Perspective for Pruning Neural Networks at Initialization](https://openreview.net/pdf?id=HJeTo2VFwH) | ICLR | [GitHub](https://github.com/namhoonlee/spp-public)|

### 2019
|   Title  | Issue | Release |
| :--------| :---: | :-----: |
| [MetaPruning: Meta Learning for Automatic Neural Network Channel Pruning](https://arxiv.org/abs/1903.10258) | ICCV | [GitHub](https://github.com/liuzechun/MetaPruning) |
| [Filter Pruning via Geometric Median for Deep Convolutional Neural Networks Acceleration](http://openaccess.thecvf.com/content_CVPR_2019/papers/He_Filter_Pruning_via_Geometric_Median_for_Deep_Convolutional_Neural_Networks_CVPR_2019_paper.pdf) | CVPR | [GitHub](https://github.com/he-y/filter-pruning-geometric-median) |
| [Towards Optimal Structured CNN Pruning via Generative Adversarial Learning (GAL)](http://openaccess.thecvf.com/content_CVPR_2019/papers/Lin_Towards_Optimal_Structured_CNN_Pruning_via_Generative_Adversarial_Learning_CVPR_2019_paper.pdf) | CVPR | [GitHub](https://github.com/ShaohuiLin/GAL) |
| [Network Pruning via Transformable Architecture Search](https://papers.nips.cc/paper/8364-network-pruning-via-transformable-architecture-search) | NeurIPS | [GitHub](https://github.com/D-X-Y/NAS-Projects) |
| [Gate Decorator: Global Filter Pruning Method for Accelerating Deep Convolutional Neural Networks](https://papers.nips.cc/paper/8486-gate-decorator-global-filter-pruning-method-for-accelerating-deep-convolutional-neural-networks) | NeurIPS | [GitHub](https://github.com/youzhonghui/gate-decorator-pruning) |
| [Global Sparse Momentum SGD for Pruning Very Deep Neural Networks](https://papers.nips.cc/paper/8867-global-sparse-momentum-sgd-for-pruning-very-deep-neural-networks.pdf) | NeurIPS | [GitHub](https://github.com/DingXiaoH/GSM-SGD) |
| [The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks](https://openreview.net/pdf?id=rJl-b3RcF7) | ICLR | - |
| [Integral Pruning on Activations and Weights for Efficient Neural Networks](https://openreview.net/forum?id=HyevnsCqtQ) | ICLR | - |
| [SNIP: Single-Shot Network Pruning Based on Connection Sensitivity](https://openreview.net/pdf?id=B1VZqjAcYX) | ICLR | [GitHub](https://github.com/namhoonlee/spp-public) |

### 2018
|   Title  | Issue | Release |
| :--------| :---: | :-----: |
| [Rethinking the Smaller-Norm-Less-Informative Assumption in Channel Pruning of Convolution Layers](https://arxiv.org/abs/1802.00124) | ICLR | [GitHub](https://github.com/jack-willturner/batchnorm-pruning) |
| [Clustering Convolutional Kernels to Compress Deep Neural Networks](http://openaccess.thecvf.com/content_ECCV_2018/papers/Sanghyun_Son_Clustering_Kernels_for_ECCV_2018_paper.pdf) | ECCV | [GitHub](https://github.com/thstkdgus35/clustering-kernels) |
| [NISP: Pruning Networks using Neuron Importance Score Propagation](http://openaccess.thecvf.com/content_cvpr_2018/papers/Yu_NISP_Pruning_Networks_CVPR_2018_paper.pdf) | CVPR | - |
| [Soft Filter Pruning for Accelerating Deep Convolutional Neural Networks](https://www.ijcai.org/Proceedings/2018/0309.pdf) | IJCAI | [GitHub](https://github.com/he-y/soft-filter-pruning) |
| [Accelerating convolutional networks via global & dynamic filter pruning](https://www.ijcai.org/Proceedings/2018/0336.pdf) | IJCAI | - |

### 2017
|   Title  | Issue | Release |
| :--------| :---: | :-----: |
| [Pruning Filters for Efficient ConvNets](https://arxiv.org/abs/1608.08710) | ICLR | [GitHub](https://github.com/Eric-mingjie/rethinking-network-pruning/tree/master/imagenet/l1-norm-pruning) |
| [Pruning Convolutional Neural Networks for Resource Efficient Inference](https://arxiv.org/abs/1611.06440) | ICLR | [GitHub](https://github.com/Tencent/PocketFlow#channel-pruning) |
| [Designing Energy-Efficient Convolutional Neural Networks using Energy-Aware Pruning](https://arxiv.org/abs/1611.05128) | CVPR | - |
| [ThiNet: A Filter Level Pruning Method for Deep Neural Network Compression](http://openaccess.thecvf.com/content_ICCV_2017/papers/Luo_ThiNet_A_Filter_ICCV_2017_paper.pdf) | ICCV | [GitHub](https://github.com/Roll920/ThiNet) |
| [Channel pruning for accelerating very deep neural networks](https://arxiv.org/abs/1707.06168) | ICCV | [GitHub](https://github.com/yihui-he/channel-pruning) |
| [Learning Efficient Convolutional Networks Through Network Slimming](http://openaccess.thecvf.com/content_ICCV_2017/papers/Liu_Learning_Efficient_Convolutional_ICCV_2017_paper.pdf) | ICCV | [GitHub](https://github.com/liuzhuang13/slimming) |
| [Scalpel: Customizing DNN Pruning to the Underlying Hardware Parallelism](https://ieeexplore.ieee.org/document/8192500) | ISCA | [GitHub](https://github.com/jiecaoyu/scalpel-1) |

### 2016
|   Title  | Issue | Release |
| :--------| :---: | :-----: |
| [Deep Compression: Compressing Deep Neural Networks with Pruning, Trained Quantization and Huffman Coding](https://arxiv.org/abs/1510.00149) | ICLR | - |
| [Eyeriss: A Spatial Architecture for Energy-Efficient Dataflow for Convolutional Neural Networks](https://ieeexplore.ieee.org/abstract/document/7551407) | ISCA | - |

### 2015
|   Title  | Issue | Release |
| :--------| :---: | :-----: |
| [Learning both Weights and Connections for Efficient Neural Networks](https://arxiv.org/abs/1506.02626) | NeurIPS | - |
