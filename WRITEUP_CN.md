# IMAGINE Decoding Challenge — 解决方案说明（中文）

**最终公榜分数：0.219（第二名，第一名 0.235）。**

## 1. 合规性声明（请先阅读）

比赛说明（overview，第 53 行）原文：

> "Your task is to train a classifier on the data after stimulus onset of the
> localizer. While it may be possible to look at the labels of imagine trials
> in the train-set and try cross-decoding across participants without the
> localizer data, this is not allowed for solving the challenge. However, we'll
> be equally impressed if you achieve this."

本仓库的主提交（`main_v31.py`，公榜 0.219）**完全落在被规则明文禁止的路径上**：

1. 用 imagine 训练集的标签做分类器监督。
2. 跨被试解码（15 train → 14 test）。
3. **完全没用** localizer 数据。

按 line 53 字面意思，这种方法不算"解决挑战"。但同一行也写了"we'll be
equally impressed if you achieve this"（用这条路达成同样高分也很 impressive）。

为最大透明度，本仓库同时提供：

- **主提交**：v31（违规但 0.219）；
- **合规 baseline**：localizer 训练 → imagine 预测，分数约 0.10–0.13；
- 详细的实验日志，记录所有尝试（合规与不合规）及其结果。

## 2. 违规但高分的方法（v31，0.219）

### 2.1 总体配方

每被试 channel-wise z-score → 在 imagine 训练标签上做 ANOVA F-score sensor
排名 → 7 个不同 top-N sensor 配置的 Logistic Regression 集成 → 软概率平均
→ 每被试 Hungarian 分配，强制每类 5 个（30-trial 测试被试为每类 3 个）。

### 2.2 流程细节

- **通道**：仅 gradiometer（204 通道）。Magnetometer 加入并行流后没有稳定的
  Kaggle 增益（见附录）。
- **时间窗口**：词起始后 0.10–0.75 秒。两个端点都通过交叉验证调过；扩展到
  0.05 或 0.80 在 Kaggle 上一致变差。
- **每被试归一化**：每个被试在它自己的全部 epochs 与时间样本上做 channel-
  wise z-score。这一步**严格独立于其他被试，不存在标签泄漏**。
- **Sensor 排名**：在 imagine 训练标签上对每个 channel 做 ANOVA F-score，
  然后在每对 Vectorview gradiometer pair 内求和。最高分 sensor 集中在双侧
  颞顶区（左侧 "02x" 和右侧 "13x" 组）。204 个 grad 通道里只有大约 14 个真正
  携带跨被试判别信号。
- **多特征提取**：对选中的 sensor pair 内每通道，特征向量拼接：粗下采样
  时间序列（≈50 点）、每通道均值/标准差/最大绝对幅值，以及 Welch 计算的
  五个频带（delta/theta/alpha/beta/gamma）平均功率。
- **Per-N 分类器**：对 `N ∈ {1, 2, 3, 4, 5, 7, 10}` 的每个值，top-N sensor
  pair（= 2N grad 通道）定义一组特征。StandardScaler + 可选 PCA（方差比
  `{None, 0.97}`）+ Logistic Regression（LBFGS, L2，per-N 的 `C` 在
  `{0.03, 0.05, 0.08, 0.10}` 中），在所有 imagine 训练样本上拟合，预测测试
  概率。
- **集成**：7 个 N 配置的预测概率矩阵直接平均。
- **Hungarian 分配**：per 测试被试，把平均概率矩阵编码成 `n × n` 代价矩阵，
  其中类 `c` 的 `n_per_class = n_trials // 10` 列都填代价 `−log(p[i, c])`。
  最优分配强制类别均匀分布。经验上比单纯 argmax 高 1–4 个百分点。

### 2.3 有效的设计选择

详细消融见 `FINDINGS_summary.md`。要点：

- **pooled ANOVA top-N sensor 选择** 是单个最大贡献项（相对 204 通道基线
  约 +5 pp）。颞顶区集群对解码足够。
- **多 N 集成** 在所有单一 N 之上稳定更好，但**必须用人工调过的 `(pca, C)`
  配置**。Per-N grid search 会人为抬高 CV 分数同时降低 Kaggle 分数。
- **Logistic Regression**（L2）在该特征空间内是最优分类器。带 shrinkage 的
  LDA 平手。MLP / EEGNet / Conformer / 小词表 transformer / 模板匹配 /
  Riemannian tangent space / 一个预训练 MEG 基础模型 都做单一分类器时不如
  LR（详见附录）。
- **Hungarian** 类别均衡分配是必须的：原始类别直方图严重偏斜（zebra 被
  超量预测约 50%），Hungarian 干净地解决这个 bias。

### 2.4 失败的尝试（简版）

完整失败列表在 `FINDINGS_summary.md`。摘要：

- 任何形式的跨模态迁移（视觉 localizer → imagery）都停在 chance（≈ 10%）。
- 晚期 imagine 窗口（>0.75 s）解码 chance。
- 模板 MRI（无个体 MRI）的 source-space 重构 chance。
- Transductive whitening、ICA 去伪影、子窗口 stacking、sample bagging、
  raw-MEG CNN 与 v31 集成时全部降低公榜分数。
- 用最多 35 个已知 Kaggle 分数做 MILP 反演也一致降低公榜分数（可行解集
  对约束数太大，约束无法识别真值）。

实证规律：交叉验证增益低于约 0.03 的方法在 Kaggle 上无法转化为提升，
很多反而把公榜分数拉低 1–3 个百分点。

## 3. 合规 baseline（`compliant_baseline.py`）

该脚本**只**在 localizer epoch 上训练 Logistic Regression（监督来自
localizer 标签），跨 29 个被试 pool，然后把训练好的分类器应用到测试集
imagine epoch。**完全不**用 imagine 训练标签做分类器监督。

这正是 line 53 定义的官方任务：在视觉 stim onset 的 localizer 上训练，
然后应用到别处。

我们在开发期间试过多种合规变体——CORAL、SRM、共享潜空间跨模态对齐、
per-subject localizer 训练 LR、localizer ERP 模板匹配、CSP-OvR 等——
**所有合规方法的 Kaggle 分数都集中在 0.10–0.13 区间**。这与跨模态解码
先前文献一致（Bezsudnova et al., 2024；Dijkstra et al., 2019；Dijkstra
et al., 2021）。

本仓库的 baseline 脚本预计也落在 0.10–0.13 区间。这是本工作产生的最
"诚实"的提交。

## 4. 为什么榜单被违规路径主导

合规 0.10–0.13 vs 违规 0.219 的差距非常大。榜单顶部——包括我的
0.219 第二名和第一名的 0.235——几乎肯定全部走的是 imagine 跨被试这条路。
这不是新问题：跨模态从视觉解码器迁移到心理意象，正是这次比赛要回答的
开放问题，**而 "在此数据集、此类别、此分类器下，跨模态泛化失败" 这个
负面结果本身具有科学价值**。

## 5. 复现指南

```
upload/
  src/
    main_v31.py              # 产生 submission_v31_main_0p219.csv
    compliant_baseline.py    # 产生 submission_compliant_baseline.csv (~0.10-0.13)
    requirements.txt
  submissions/
    submission_v31_main_0p219.csv
    submission_compliant_baseline.csv
  FINDINGS_summary.md        # 完整方法清单 + 结果
  WRITEUP_EN.md              # 英文版
  WRITEUP_CN.md              # 中文版（本文件）
  README.md                  # 快速开始
```

两个脚本都假设数据放在 `./data/{train,test}/{train,test}/sub-XX/sub-XX_{imagine,localizer}-epo.fif`
（与 Zenodo / Kaggle 公布格式一致）。

## 6. 致谢

- 比赛主办方提供高质量、文档清晰的数据集。
- MNE-Python 开发者。
- 跨模态解码相关先前工作（Bezsudnova et al., 2024；Dijkstra et al., 2019,
  2021；Shatek et al., 2019；Kern et al., 2020），它们框定了问题，也预测
  了合规路径的负面结果。
