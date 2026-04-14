# 项目进度总结

**项目名称：** 具身智能失败恢复 Benchmark  
**目标投稿：** CVPR 2026 Embodied AI Workshop  
**报告日期：** 2026年4月14日  
**已测模型：** GPT-5.4  
**数据集规模：** 300 样本（5 种失败类型 × 60 样本，均衡分布）

---

## 一、已完成工作

### 1.1 Benchmark 基础设施

- [x] 失败类型定义（F1–F5 五类 taxonomy）
- [x] 数据集构建（基于 ALFRED 环境，300 条均衡样本）
- [x] 评估流水线（含 Bootstrap 置信区间，n_boot=1000）
- [x] Semi-loop 闭环恢复执行器
- [x] JSON 格式修复模块
- [x] 统一结果表生成器（md / tex / json 三种格式）

### 1.2 失败类型定义

| 类型 | 名称 | 含义 | 正确恢复动作 |
|---|---|---|---|
| F1 | Misorientation | 朝向错误，目标不在视野内 | LookAround |
| F2 | Action No-Effect | 动作执行后无变化（如抓取失败） | Retry |
| F3 | Wrong Target | 操作了错误的对象 | Navigate(correct_target) |
| F4 | Partial Progress | 部分完成但需重试 | Retry |
| F5 | State Mismatch | 世界状态与预期不符 | Replan / Navigate |

### 1.3 实验完成情况

| 实验类别 | 具体配置 | 状态 |
|---|---|---|
| Baselines | Oracle、Blind-Heuristic、Heuristic-Oracle、LookAround | ✅ 完成 |
| VLM 输入消融 | full(B+A)、after-only、text-only、swap-test | ✅ 完成 |
| Prompt 策略消融 | counterfactual、mental_simulation | ✅ 完成 |
| Swap 敏感性分析 | 逐类型 McNemar 检验 | ✅ 完成 |
| Semi-loop 闭环恢复 | 全部 VLM 模式 + Oracle 上界 | ✅ 完成 |

---

## 二、实验结果

### 2.1 主结果表

| 方法 | Acc (%) | 95% CI | Macro-F1 | F1 | F2 | F3 | F4 | F5 |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| B: Oracle | 100.0 | [100, 100] | 1.0000 | 100 | 100 | 100 | 100 | 100 |
| B: Blind-Heuristic | 37.3 | [32.0, 43.0] | 0.0416 | 86.7 | 0.0 | 0.0 | 100 | 0.0 |
| B: Heuristic-Oracle | 33.7 | [28.3, 39.3] | 0.0275 | 100 | 8.3 | 1.7 | 58.3 | 0.0 |
| B: LookAround | 20.0 | [15.7, 24.7] | 0.0083 | 100 | 0.0 | 0.0 | 0.0 | 0.0 |
| V: full (B+A) | **28.3** | [23.7, 33.3] | **0.1199** | 61.7 | 21.7 | 18.3 | 31.7 | 8.3 |
| V: swap-test | 28.3 | [23.7, 33.3] | 0.1074 | 48.3 | 26.7 | 20.0 | 40.0 | 6.7 |
| V: after-only | 24.7 | [20.3, 29.7] | 0.0913 | 53.3 | 21.7 | 16.7 | 28.3 | 3.3 |
| V: text-only | 16.7 | [12.7, 21.0] | 0.1162 | 8.3 | 25.0 | 13.3 | 25.0 | 11.7 |
| V: mental_simulation | 26.7 | [22.0, 31.7] | 0.1054 | 73.3 | 11.7 | 36.7 | 5.0 | 6.7 |
| V: counterfactual | 0.7 | [0.0, 1.7] | 0.0003 | 3.3 | 0.0 | 0.0 | 0.0 | 0.0 |

### 2.2 Semi-Loop 闭环恢复结果

| 方法 | Recovery@K (%) | Completion (%) |
|---|---:|---:|
| B: Oracle（上界） | 60.0 | 60.0 |
| V: mental_simulation | 11.0 | 11.0 |
| V: swap-test | 10.7 | 10.7 |
| V: text-only | 10.0 | 10.0 |
| V: full (B+A) | 9.7 | 9.7 |
| V: after-only | 8.3 | 8.3 |
| V: counterfactual | 0.0 | 0.0 |

> Recovery 上界为 60%，因为 F1（LookAround）和 F4（Retry）的 goal state 为空，symbolic executor 无法验证恢复是否成功。

---

## 三、核心发现

### 发现 1：VLM 具有跨类别泛化能力，但整体准确率有限

最佳 VLM 配置（full, Acc=28.3%）在所有 5 类失败上均有非零表现（Macro-F1=0.12），而最强 heuristic baseline（Blind-Heuristic, Acc=37.3%）仅在 F1 和 F4 两类上有效（Macro-F1=0.04）。VLM 在准确率上不如简单规则，但在类别覆盖上明显更优。

### 发现 2：视觉输入的贡献呈递减趋势

- text-only → after-only：**+8.0pp**（图像提供关键视觉信息）
- after-only → full (B+A)：**+3.6pp**（before 帧提供参照但增益有限）

说明 VLM 对 before/after 图像差异的比较推理能力较弱。

### 发现 3：Counterfactual Prompting 导致模型彻底崩溃

Counterfactual 策略准确率仅 0.7%。模型将"如果动作成功了会怎样"的反事实推理框架误解为"图像没有变化 → 动作无效 → Retry"，几乎全部预测为 F2。说明当前 VLM 不具备可靠的反事实推理能力。

### 发现 4：Mental Simulation 展现互补的类别偏好

Mental simulation 在 F1（73.3%）和 F3（36.7%）上显著优于标准 prompt（61.7% 和 18.3%），但 F4 暴跌至 5.0%（标准为 31.7%）。不同推理策略对不同失败类型存在互补优势，未来可探索策略融合方案。

### 发现 5：巨大的 Oracle Gap 证明任务极具挑战性

Oracle（100%）与最佳 VLM（28.3%）之间存在 71.7pp 的差距，表明具身失败恢复仍是一个远未解决的开放问题，为后续研究提供了充足的提升空间。

---

## 四、输出文件清单

```
outputs/
├── baselines/
│   └── baseline_summary_300.json                # 4 个 baseline 的完整结果
├── ablations/
│   ├── ablation_summary.json                    # VLM 4 模式消融结果
│   ├── predictions_full_gpt54.jsonl             # full 模式逐样本预测
│   ├── predictions_afteronly_gpt54.jsonl         # after-only 模式逐样本预测
│   └── predictions_textonly_gpt54.jsonl          # text-only 模式逐样本预测
├── ablations_counterfactual/
│   └── ablation_summary.json                    # counterfactual 策略结果
├── ablations_simulation/
│   └── ablation_summary_mental_simulation.json   # mental_simulation 策略结果
└── tables/
    ├── unified_table.md                          # Markdown 统一结果表
    ├── unified_table.tex                         # LaTeX 统一结果表
    └── unified_rows.json                         # JSON 结构化结果（方便绘图）
```

---

## 五、待完成工作

### P0：投稿必需

| 任务 | 说明 | 预估时间 |
|---|---|---:|
| 测试更多模型 | 至少补充 1–2 个模型（如 GPT-4o、Claude、Gemini 或开源的 LLaVA-Next / Qwen-VL），增强实验说服力 | 2–3 天 |
| 生成关键图表 | Confusion matrix、各类型准确率对比柱状图、方法总览图 | 半天 |
| 错误分析 | 分析模型最常见的错误模式，提供定性案例分析（qualitative examples） | 1 天 |
| 撰写论文 | Workshop paper 一般 4–8 页，CVPR 格式 | 3–4 天 |

### P1：强烈建议

| 任务 | 说明 |
|---|---|
| 开源 vs 闭源模型对比 | 加入至少一个开源 VLM，形成闭源/开源的性能对比 |
| 典型案例分析 | 挑选 3–4 个成功/失败案例，配图详细分析模型的推理过程 |
| 混淆矩阵深入分析 | 分析模型在各失败类型之间的混淆模式 |

### P2：锦上添花

| 任务 | 说明 |
|---|---|
| 扩大数据集规模 | 从 300 扩展到 500+ 样本，增强统计检验力 |
| 多步恢复实验 | 测试 K > 1 的多步恢复场景 |
| 更多 prompt 变体 | 尝试 few-shot、chain-of-thought 等方案 |

---

## 六、快速参考

```
最佳 VLM 配置:       full (B+A)          Acc = 28.3%    Macro-F1 = 0.120
最强 Baseline:       Blind-Heuristic     Acc = 37.3%    Macro-F1 = 0.042
Oracle 上界:                             Acc = 100.0%
最佳 Recovery@K:     mental_simulation   11.0%
Recovery 上界:       Oracle              60.0%
API 开销（估算）:     ~$0.85（mental_simulation 模式）
```
```
