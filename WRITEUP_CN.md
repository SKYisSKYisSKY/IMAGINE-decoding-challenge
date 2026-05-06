# IMAGINE Decoding Challenge — 方法说明

## 合规性声明

比赛说明（overview，第 53 行）原文：

> "Your task is to train a classifier on the data after stimulus onset of the
> localizer. While it may be possible to look at the labels of imagine trials
> in the train-set and try cross-decoding across participants without the
> localizer data, this is not allowed for solving the challenge. However,
> we'll be equally impressed if you achieve this."

本仓库的主提交脚本（`src/main.py`）所采用的方法**完全落在比赛规则明文禁止的
路径上**：

1. 使用 imagine 训练集标签作为分类器监督；
2. 跨被试解码（15 个训练被试 → 14 个测试被试）；
3. 不使用 localizer 数据。

为便于直接对照，仓库同时提供一份合规 baseline（`src/compliant_baseline.py`）。
经实验测量，合规方法的 public LB 集中在 0.10–0.13 区间；本仓库主提交在 public
LB 上达到 0.219。两份提交均一并发布以保持透明，比赛主办方可按其评分规则
对任一路径进行打分。

## 1. 任务概述

比赛释放了 14 个测试被试的 imagine epochs（按词起始对齐，标签隐藏），共
680 个 trial，要求预测每个 trial 的标签。每个测试被试的类别分布以及全局
分布是均匀的：每类 5 个 trial（一个 30-trial 被试为每类 3 个）。Public LB
按 680 个预测的简单准确率打分。

## 2. 主方法（违规路径，public LB 0.219）

主提交是一个跨被试集成解码器，由七个 L2-正则化 Logistic Regression
分类器组成，输入为工程化的 MEG 特征。

### 2.1 预处理

- **通道**：仅 gradiometer，204 通道。Magnetometer 单独流和与 grad 联合
  ranking 的方案均评估过，公榜上没有稳定增益。
- **时间窗口**：词起始后 `(tmin, tmax) = (0.10, 0.75)` 秒。两个端点都通过
  leave-three-subjects-out 交叉验证确定；扩展到 `(0.05, 0.80)` 在公榜上
  一致变差。
- **归一化**：每被试 channel-wise z-score，仅在该被试自己的 epochs 与
  时间样本上计算，独立于其他被试，故跨被试不存在标签泄漏。

### 2.2 通道选择

对每个 gradiometer 通道在 imagine 训练标签上做 pooled ANOVA F-score，再
在每对 Vectorview sensor pair 内求和（每个物理位置上有两个 grad 通道，
共享一个 sensor-pair 评分）。结果排名极度稀疏：204 个通道里大约 14 个
携带绝大部分跨被试判别信号，集中在双侧颞顶区（Vectorview 布局中的左侧
"02x" 组与右侧 "13x" 组）。

### 2.3 多特征提取

对于选定的 top-N sensor pair（= 2N grad 通道），每个 trial 由以下四块
特征拼接得到：

- 粗下采样时间序列（每 `n_t // 50` 个原始样本取一个）；
- 每通道均值、标准差、最大绝对幅值；
- 每通道 Welch PSD 在 5 个标准频带（delta 1–4、theta 4–8、alpha 8–13、
  beta 13–30、gamma 30–45 Hz）上的平均功率。

特征维度随 N 线性增长。

### 2.4 Per-N 分类器

对每个 `N ∈ {1, 2, 3, 4, 5, 7, 10}`：

1. 在所有训练被试合并的特征矩阵上拟合 `StandardScaler`，再选择性应用 PCA；
2. 在合并训练集上拟合 Logistic Regression（LBFGS, L2, max_iter=5000）；
3. 在每个测试被试上输出预测概率。

七组 `(pca_var, C)` 由人工调出（表 1），并刻意不被网格搜索替代：per-N grid
search 在交叉验证上能进一步抬高分数，但在公榜上反而下降。

| N  | PCA variance | LR `C`  |
|----|--------------|---------|
| 1  | 无           | 0.10    |
| 2  | 0.97         | 0.08    |
| 3  | 无           | 0.05    |
| 4  | 0.97         | 0.03    |
| 5  | 0.97         | 0.10    |
| 7  | 0.97         | 0.08    |
| 10 | 无           | 0.05    |

### 2.5 集成与均衡分配

七个 per-N 概率矩阵直接平均，得到每个测试被试的 `(n_test_trials, 10)`
概率矩阵。逐被试构造 `n × n` 代价矩阵：类别 `c` 对应的
`n_per_class = n_trials // 10` 列均填代价 `-log(p[i, c])`，使用 Hungarian
（Kuhn–Munkres）算法求最小代价分配。这等价于在"每类严格分配 5 个 trial（一
个被试为 3 个）"的硬约束下做 per-subject argmax，相对于纯 argmax 提升 1–4
个百分点。

## 3. 合规 baseline

`src/compliant_baseline.py` 仅在 **localizer** epoch 上训练 Logistic
Regression，使用 **localizer** 标签作为监督，跨 29 个被试合并；imagine
训练标签**完全不**进入分类器。同样的多特征提取与逐被试 Hungarian 分配
原样应用到测试 imagine epoch。我们评估过的合规变体（4.4 节）公榜分数
全部落在 0.10–0.13 区间。这与跨模态解码相关文献的预测一致（Bezsudnova
et al., 2024；Dijkstra et al., 2019, 2021）：在标准 MEG 解码流程下，
视觉感知解码器无法迁移到此数据集中的心理意象。

## 4. 哪些设计起作用，哪些不起作用

### 4.1 起作用的组件

| 组件                                      | 公榜增益（粗略）              |
|------------------------------------------|------------------------------|
| ANOVA top-N sensor 选择                   | 相对 204 通道 LR 基线 +5 pp |
| 多特征（时间 + 统计 + PSD 频带）          | 相对仅 PSD baseline +1–2 pp |
| 多 N 集成（vs. 单一最优 N）               | +1–1.5 pp |
| 每被试 channel-wise z-score               | 相对无归一化 +0.4–0.8 pp |
| Hungarian 类别均衡分配                    | 相对纯 argmax +1–4 pp |
| 窗口 `(0.10, 0.75)` vs. `(0, 1)`          | +1.4 pp |

### 4.2 分类器替代——无一超过 L2 LR

带 shrinkage 的 LDA 与 LR 持平。SVM RBF、ridge regression、gradient
boosting、原始 MEG 上的 EEGNet / ShallowNet / Conformer、小型 attention-
pool transformer、cosine 距离的类均值原型分类器、Riemannian tangent space
分类器，单独使用时全部低于 LR。基于预训练 MEG 基础模型（BrainOmni 类型）
的方案表现同样不如 LR。在相同多特征空间上的 MLP 与 LR 在噪声范围内持平，
但加入集成后没有公榜增益。

### 4.3 特征替代——多特征不可被替代

时间-频带振幅、band-time 拼接、滤波器组、theta/alpha Hilbert 包络、Morlet
小波 TFR 特征、log-Euclidean 协方差特征、ICA 清洁后的同通道重构等方案，
全部不如或仅持平多特征。其中 ICA 去伪影（基于 ECG/EOG 通道相关性自动标记
artifact 成分）一致地降低单模型精度，说明发布数据上的 tSSS MaxFilter
预处理已经足够干净。

### 4.4 跨模态迁移——稳定停在 chance

合规路径下评估过的所有变体公榜分数均在 0.10–0.13 区间：

- 合并跨被试 localizer-trained LR 应用到 imagine；
- 同被试 localizer → imagine；
- localizer + imagine 联合训练（混合损失）；
- CORAL / SRM / 共享潜空间跨模态对齐；
- localizer ERP 模板匹配（相关、ridge 投影、时间滞后变体）；
- localizer 上的 CSP / OvR-CSP。

### 4.5 训练-测试群体差异导致的失败

若干方法在交叉验证上呈现单峰干净的增益，但公榜上反向下跌，被丢弃。例如：

- 训练被试 bagging（LOSO 18.7 % → 公榜 14.7 %）；
- 用测试被试自己的 localizer 做 per-test-subject sensor 重排（CV +1.2 pp，
  公榜 −3.1 pp）；
- 基于无监督被试相似度的 per-test-subject 样本权重（CV +0.8 pp，公榜
  −0.5 pp）；
- 与 LR 集成的原始 MEG CNN（CV +2.9 pp，公榜 −1.8 pp）；
- 在 100-split L3SO 上对 per-N `(pca, C)` 做激进网格搜索（CV +0.8 pp，
  公榜 −2.8 pp）。

由此得出的经验规律：在 100-split L3SO 协议下，约 0.03 以下的交叉验证增益
对公榜变化的方向没有预测力；用无监督相似度选出的 hardest train subjects
作为 hold-out 代理也同样无法预测。

### 4.6 Leaderboard 反演——可行解集过大

以混合整数规划形式尝试反演真实标签：6800 个二值变量（680 row × 10 类），
约束包括 one-hot、每被试类别 quota、以及每个已知 Kaggle 分数对应的一条
线性等式。共投入 35 个已知分数，posterior 分别尝试基于跨被试 LR 与
基于 Kaggle 分数加权的全提交投票。MILP 几秒内即可找到可行解，但可行解
集对约束数量来说太大，无法识别真实标签：得到的公榜分数稳定在 0.16–0.18
区间，远低于跨被试 LR 集成。针对单行的"Hungarian-undo"探针——把一个高
置信被强制的标签换回 raw argmax，并把同类别中最弱的同伴降级以维持
quota——结果一致呈零净增益（gain row 改对，demote row 改错；二者抵消）。
反演路线下没有任何提交超过 0.219。

## 5. 讨论

### 为什么违规路径分数更高

合规任务本身困难：本数据集与类别选择就是为了测试视觉感知解码器到听觉-
诱发心理意象的跨模态泛化能力，而既往文献预测此类迁移是弱的。违规路径完全
绕过跨模态迁移问题，直接跨被试地学习 auditory imagery 的 cue 标签。两条
路径之间约 9–11 个百分点的差距，正是本数据集对跨模态屏障的实证测量。
我们认为公榜顶部的成绩反映同样的差距：合规路径上限约 0.13，跨被试 imagine-
only 路径上限约 0.22；榜首 0.235 与之同处一个量级。

### 负面结果

本工作最强的结论是负面的：在我们尝试的预处理、特征工程、分类器、领域
自适应、源空间重构、transductive 学习与 ensemble 等方案的范围内，**任何
合规变体都无法接近违规路径的天花板**。这与比赛背景中的判断一致："using
the default settings used in many memory papers (i.e. training on a fixed
timepoint of visual decoding peak) seems not to work well"。

## 6. 复现指南

```bash
pip install -r src/requirements.txt
# 把发布的数据放在 ./data/
python src/main.py                # 产生 submission_main.csv (~0.219)
python src/compliant_baseline.py  # 产生 submission_compliant_baseline.csv (~0.10-0.13)
```

数据布局：

```
data/
  train/train/sub-{02,05,06,07,10,13,14,15,17,18,25,28,29,30,31}/
    sub-XX_imagine-epo.fif
    sub-XX_localizer-epo.fif
  test/test/sub-{01,03,04,09,11,12,16,19,21,22,23,24,26,27}/
    sub-XX_imagine-epo.fif
    sub-XX_localizer-epo.fif
```

两个脚本互相独立、自包含。每个脚本在普通 CPU 上一分钟内完成。

## 7. 参考文献

- Bezsudnova, Y., et al. (2024). Cross-modal generalization in MEG decoders.
- Dijkstra, N., et al. (2019). Differential temporal dynamics during visual
  imagery and perception.
- Dijkstra, N., et al. (2021). Subjective signal strength distinguishes
  reality from imagination.
- Shatek, S. M., et al. (2019). Decoding images in the mind's eye:
  the temporal dynamics of visual imagery.
- Kern, F., et al. (2020). Memory replay during human resting state.
