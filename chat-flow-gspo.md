# FastWAM + Flow-GSPO RL: Implementation Report

本文档是对当前 FastWAM RL 训练系统的确定性描述。所有内容均与代码实现一一对应，不包含假设、设想或未来计划。

---

## 1. Project Overview

以 `FastWAM` 为基座，冻结 video DiT 与 world representation，只对 action-side modules 做在线 RL，采用 Flow-GSPO 方法完成消融实验。

两个正交的消融维度：

**维度 A — Ratio 计算粒度**（`rl.variant`）：
- `traj_chunk`：trajectory-level advantage + chunk-level ratio
- `traj_traj`：trajectory-level advantage + trajectory-level ratio

**维度 B — Action Chunk 执行策略**（`rl.action_horizon` / `rl.exec_horizon`）：
- `match`（predict=exec）：将 action_horizon 缩短为 replan_steps，预测与执行一致
- `slice`（predict>exec）：保持原始 action_horizon=32 预测，log_prob 只覆盖实际执行的前 N 步

交叉后共 4 组实验。所有实验共用同一套 rollout 数据结构、trainer 主循环、checkpoint 机制和 logging 系统。

---

## 2. Model Architecture & Trainable Scope

### 2.1 FastWAM 模型结构

```
text_encoder → context embeddings
                     │
              proprio_encoder (nn.Linear)
                     │  将机器人状态拼接到 context
                     ▼
         ┌───────────┴───────────┐
         │                       │
   video_expert             action_expert
   (Wan2.2 DiT)             (ActionDiT)
         │                       │
    video tokens           action tokens
         │                       │
         └───────────┬───────────┘
                     │
                    MoT
            (Mixture-of-Transformers)
         共享 self-attention + 各自 FFN
```

- **video_expert**：Wan2.2 视频 DiT，负责从图像提取 world representation。始终冻结。
- **action_expert（ActionDiT）**：独立的 diffusion transformer，输入为 noisy action tokens + timestep + context，输出 flow velocity prediction $v_\theta$。包含 action_encoder、time_embedding、time_projection（6 路 adaLN）、DiTBlocks、head（Linear → action_dim）。
- **MoT（Mixture-of-Transformers）**：将 video 和 action 两个 expert 的 token 拼接，做统一 self-attention，然后拆分回各自分支执行独立 FFN。推理时使用 KV cache：video 分支只做一次 prefill，后续每个 denoising step 只重算 action 分支。
- **proprio_encoder**：单层 `nn.Linear(proprio_dim, text_dim)`，将机器人本体感知（关节角度、夹爪状态）投影到文本嵌入空间，拼接到 context 末尾。

### 2.2 Trainable Scope

通过 `rl.trainable_scope` 配置，三个确定选项：

| Scope | 可训练模块 | 说明 |
|---|---|---|
| `action_expert_only` | ActionDiT 全部参数 + proprio_encoder | 最保守策略，只调动作预测能力 |
| `action_expert_and_mot` | ActionDiT + MoT action 分支 + proprio_encoder | 额外允许 attention 层学习 video→action 的信息提取 |
| `full` | 全部参数 | 连 video_expert 也一起调（一般不使用） |

实现方式（`trainer.py:_configure_trainable_scope`）：先将整个 model 设为 `eval()` + `requires_grad_(False)`，再按 scope 选择性地 `train()` + `requires_grad_(True)` 对应子模块。

---

## 3. Algorithm: Flow-GSPO on FastWAM

以下公式与 OmniVLA-RL 论文一一对应，代码实现严格遵循论文。

### 3.1 Flow ODE → SDE

将确定性 ODE denoising 改为随机 SDE（论文 eq.10-11）。从纯噪声 $A_t^0$ 出发，经过 $K$ 步 Euler-Maruyama 离散化：

$$A_t^{\tau+\delta} = A_t^\tau + \left[v_\theta + \frac{\sigma_\tau^2}{2}(A_t^\tau + (1-\tau)v_\theta)\right]\delta + \sigma_\tau\sqrt{\delta}\,\epsilon$$

其中：
- $\tau = 1 - \sigma$，$\sigma$ 为当前 noise level，从 1 递减到 0
- $\sigma_\tau = \sigma_{\max} \cdot \sigma$，线性噪声调度
- $\delta = \sigma_{i+1} - \sigma_i$（负值）
- $v_\theta$ 为模型预测的 velocity

### 3.2 Transition Probability

每一步 SDE transition 服从各向同性高斯（论文 eq.12-13）：

$$p_\theta(A_t^{\tau+\delta} \mid A_t^\tau, s_t) \sim \mathcal{N}(\mu_\tau, \sigma_\tau^2\delta \cdot I)$$

其中均值：

$$\mu_\tau = A_t^\tau + \left[v_\theta + \frac{\sigma_\tau^2}{2}(A_t^\tau + (1-\tau)v_\theta)\right]\delta$$

代码实现位置：
- Rollout 阶段：`scheduler_continuous.py:step_sde_with_logprob`
- 训练阶段（带梯度）：`fastwam.py:compute_logprob_from_chain`

两处使用完全相同的 drift 公式，包含完整的 score correction 项。

### 3.3 Log-Probability

单个 transition 的 log-density 使用标准多元高斯密度（论文 eq.14）：

$$\log p_\theta = -\frac{1}{2\sigma_\tau^2\delta}\sum_{h,d}(A_{t,h,d}^{\tau+\delta} - \mu_{\tau,h,d})^2 - N\log(\sigma_\tau\sqrt{\delta}) - \frac{N}{2}\log(2\pi)$$

其中 $N = H \times D$（action_horizon × action_dim）为 action block 的总维度。**使用 sum over 所有维度**，不使用 mean。

**Log-prob 截断**：当设置 `exec_horizon` 时，$N$ 变为 `exec_horizon × D`，`diff` 只取前 `exec_horizon` 个 action 维度。这是对转移概率的边际化（见 Section 5 的数学分析）。

Action block 的完整 log-likelihood 为所有 $K$ 步 transition 之和：

$$\log \pi_\theta(A_t \mid s_t) = \sum_{\tau=0}^{K-1}\log p_\theta(A_t^{\tau+\delta} \mid A_t^\tau, s_t)$$

代码：`infer_action_with_logprob` 中 `total_log_prob = sum(log_probs_per_step)`。

### 3.4 Importance Ratio

**Chunk-level ratio**（Exp-2，论文 eq.15）：

$$s_{i,t}(\theta) = \exp\left(\frac{1}{HK}\left[\log\pi_\theta(A_{i,t}|s_{i,t}) - \log\pi_{\theta_{\text{old}}}(A_{i,t}|s_{i,t})\right]\right)$$

代码：`objectives.py:_compute_chunk_level_objective`
```python
log_ratio = (new_log_prob - old_log_prob) / float(chunk.block_size)
ratio = torch.exp(log_ratio)  # block_size = H * K
```

**Trajectory-level ratio**（Exp-3）：

$$s_i(\theta) = \exp\left(\frac{1}{\sum_t HK}\sum_{c}\left[\log\pi_\theta(A_{i,c}|s_{i,c}) - \log\pi_{\theta_{\text{old}}}(A_{i,c}|s_{i,c})\right]\right)$$

即对所有 chunks 的 log_prob 求和后除以总 block_size。代码：`objectives.py:_compute_trajectory_level_objective`
```python
log_ratio = (new_total - old_total) / total_block_size  # total_block_size = C * H * K
```

### 3.5 Advantage

采用 trajectory-level group advantage normalization（论文 eq.16 的 trajectory 版本）。

对同一 task、同一 reset state 采样的 $G$ 条 trajectory，terminal reward 为 $R_i \in \{0, 1\}$（LIBERO sparse success reward）：

$$\hat{A}_i = \frac{R_i - \mu_R}{\sigma_R + \epsilon}$$

**Uniform assignment**（默认）：

$$\hat{A}_{i,t} = \hat{A}_i \quad \forall t$$

**Temporal decay assignment**（可选，`rl.trajectory_assignment=temporal_decay`）：

$$\hat{A}_{i,t} = \gamma^{N_i - t}\hat{A}_i$$

零方差过滤：若 group 内所有 trajectory reward 相同（全 success 或全 failure），则整组 advantage 置零，不参与训练。

代码：`advantages.py:compute_gspo_trajectory_advantages` 和 `compute_gspo_trajectory_decay_advantages`。

### 3.6 Objective

Clipped surrogate + KL 正则（论文 eq.17-18）：

$$\mathcal{L}(\theta) = -\frac{1}{G}\sum_i\left[\min\left(s_i\hat{A}_i,\;\text{clip}(s_i, 1-\varepsilon, 1+\varepsilon)\hat{A}_i\right) - \beta\,D_{\text{KL}}\right]$$

KL 近似：

$$D_{\text{KL}} \approx s - 1 - \log s = \exp(\log\_ratio) - 1 - \log\_ratio$$

代码：`objectives.py`，其中 `clip_range` 对应 $\varepsilon$，`kl_coef` 对应 $\beta$。

---

## 4. Ablation Experiments

### 4.1 维度 A：Ratio 计算粒度

**`traj_chunk`**：
- Advantage：trajectory-level group normalization（uniform assignment）
- Ratio：per-chunk，每 chunk 独立计算 $\exp(\frac{\log\pi_\theta - \log\pi_{\theta_{\text{old}}}}{HK})$
- Objective：对每个有非零 advantage 的 chunk 独立做 clipped surrogate + KL

特点：advantage 从 trajectory 级别统一分配给所有 chunk，但 ratio 在 chunk 级别独立 clipping，保留 per-step 梯度稳定性。

**`traj_traj`**：
- Advantage：trajectory-level group normalization（与 traj_chunk 相同）
- Ratio：per-trajectory，对所有 chunks 的 log_prob 求和后归一化
- Objective：对每条有非零 advantage 的 trajectory 做一次 clipped surrogate + KL

特点：reward 单位与 ratio 单位完全统一到 trajectory 级别。

### 4.2 维度 B：Action Chunk 执行策略

**问题背景**：FastWAM 模型预训练时预测 $H=32$ 个 action step，但实际执行时只取前 `replan_steps=10` 步后即重新规划。这意味着 $32 - 10 = 22$ 个预测的 action 从未在环境中执行，不贡献 reward。

在 GSPO 的 importance ratio 中，`log_prob` 对所有 $H \times K$ 个维度求和。若包含未执行的 action，ratio 会被不相关的概率变化主导——策略可能只改变了被丢弃部分的分布，但 ratio 仍产生大幅偏移，触发不必要的 clipping。

**方案 match（predict=exec）**：将 `action_horizon` 从 32 缩短为 10，模型预测 10 步，执行 10 步。预测与执行完全一致，log_prob 自然覆盖所有 action，无需截断。ActionDiT 的 `action_encoder`、DiTBlock、`head` 均为逐 token 操作，RoPE 缓存预计算至 1024，完全支持 seq_len=10。

**方案 slice（predict>exec）**：保持 `action_horizon=32`，但 log_prob 只对前 `exec_horizon=10` 个 action 维度求和。采样过程仍对全部 32 个 action 执行 SDE denoising（因为 self-attention 需要），但 ratio 计算只考虑实际执行的子集。此方案参考 RLinf 的 GR00T 实现。

### 4.3 完整实验矩阵

| 实验编号 | variant | action_horizon | exec_horizon | 说明 |
|---|---|---|---|---|
| Exp-2a | `traj_chunk` | 10 (match) | null | predict=exec=10, chunk ratio |
| Exp-2b | `traj_chunk` | 32 (default) | 10 (slice) | predict=32 exec=10, sliced log_prob, chunk ratio |
| Exp-3a | `traj_traj` | 10 (match) | null | predict=exec=10, trajectory ratio |
| Exp-3b | `traj_traj` | 32 (default) | 10 (slice) | predict=32 exec=10, sliced log_prob, trajectory ratio |

启动命令：
```bash
# Exp-2a: match + chunk ratio
python scripts/train_rl.py EVALUATION.task_suite_name=libero_spatial \
  rl.variant=traj_chunk rl.action_horizon=10 rl.exec_horizon=null ckpt=<pt>

# Exp-2b: slice + chunk ratio
python scripts/train_rl.py EVALUATION.task_suite_name=libero_spatial \
  rl.variant=traj_chunk rl.action_horizon=null rl.exec_horizon=10 ckpt=<pt>

# Exp-3a / Exp-3b: 同上，改 rl.variant=traj_traj
```

### 4.4 Block-1 为何被移除

原 Block-1（block-level advantage + chunk-level ratio）已被移除。原因：LIBERO 只提供 sparse terminal reward（0/1 success），所有中间 chunk 的 `chunk_rewards` 全为 0，只有 terminal chunk 可能有 reward。这导致 block-level group normalization 对绝大多数 chunk 产生零方差 group，advantage 全为 0，无法提供有效的梯度信号。

---

## 5. Reward Design & Action Chunk Analysis

### 5.1 Reward

LIBERO 环境只提供 sparse terminal reward：

$$R = \begin{cases}1.0 & \text{task success (done=True)}\\0.0 & \text{otherwise}\end{cases}$$

无中间 step reward、progress reward 或 alignment reward。Trajectory-level advantage 只在 group 内存在成功/失败混合时才有非零值。

### 5.2 Action Chunk Partial Execution 问题

FastWAM 的 action chunking 机制：

```
模型预测: [a_0, a_1, ..., a_9, a_10, ..., a_31]  ← H=32 步
实际执行: [a_0, a_1, ..., a_9]                     ← replan_steps=10 步
丢弃:                         [a_10, ..., a_31]    ← 22 步从不执行
```

在 Flow-GSPO 中，importance ratio 为：

$$s = \exp\left(\frac{1}{HK}\sum_{k,h}\log\frac{p_\theta(x^{k,h})}{p_{\theta_{\text{old}}}(x^{k,h})}\right)$$

其中 $H=32$，但只有前 10 步的 action 影响环境 reward。后 22 步的概率变化会干扰 ratio 信号。

**对比其他实现**：
- OmniVLA-RL：$H=16$，执行全部 16 步，不存在此问题
- RLinf FlowPolicy（pi_0）：预测步数 = 执行步数，不存在此问题
- RLinf GR00T：`action_horizon=16`（内部预测），`action_chunk=4`（实际输出），log_prob 显式切片到 `action_chunk` 维度，与方案 slice 一致

### 5.3 两种方案的 Attention 差异

MoT 中 action tokens 的 attention mask（`fastwam.py:404`）：

```python
# action -> action: 全连接 self-attention
mask[video_seq_len:, video_seq_len:] = True
```

每层 MoT 将 action tokens 与 video KV cache 拼接后做 mixed attention（`mot.py:426-433`）：

```python
k_cat = torch.cat([k_video, k_action], dim=1)  # video + action keys
mixed = self._mixed_attention(q_cat=q_action, k_cat=k_cat, v_cat=v_cat, ...)
```

**Match（action_horizon=10）**：每个 action token 的 attention context = 10 个 action token + video tokens。

**Slice（action_horizon=32）**：每个 action token 的 attention context = **32** 个 action token + video tokens。

前 10 个 token 的 velocity 预测在两种情况下不同，因为 attention 的 key/value 数量不同。这不是 log_prob 截断能弥补的——模型内部的计算已经分叉。因此两个方案测试的是本质不同的问题：

- **Match**：模型在 10-token 短期上下文中做规划，action tokens 之间交互范围小
- **Slice**：模型在 32-token 长期上下文中做规划但只对短期部分算 ratio，action tokens 之间交互范围大

### 5.4 Log-prob 截断的数学基础

SDE 每步的转移概率为各向同性高斯：

$$p(x_{k+1} | x_k) = \mathcal{N}(x_{k+1};\, \mu_k,\, \sigma_k^2 I)$$

协方差矩阵为 $\sigma_k^2 I$（对角），log density 按维度独立分解：

$$\log p(x_{k+1} | x_k) = \sum_{h=1}^{H}\sum_{d=1}^{D}\left[-\frac{(x_{k+1}[h,d]-\mu_k[h,d])^2}{2\sigma_k^2} - \log\sigma_k - \frac{1}{2}\log 2\pi\right]$$

截断到前 $H'$ 个 action 等价于取边际转移 log density：

$$\log p(x_{k+1}[\,:H',:\,] | x_k) = \sum_{h=1}^{H'}\sum_{d=1}^{D}\left[-\frac{(x_{k+1}[h,d]-\mu_k[h,d])^2}{2\sigma_k^2} - \log\sigma_k - \frac{1}{2}\log 2\pi\right]$$

注意：均值 $\mu_k$ 仍通过 self-attention 依赖所有 $H$ 个 action 的状态 $x_k$（neural network 的全局依赖性），但 $x_{k+1}$ 各维度在 log density 中的贡献是独立的（对角协方差保证）。因此截断是合法的边际化操作，不是近似。

注意：截断只影响 log_prob 的计算范围，不影响模型内部的 attention 计算。模型始终对全部 action tokens 执行 SDE denoising。

---

## 6. Observation Mismatch Analysis

在 multi-chunk closed-loop control 中，不同 trajectory 的中间 observation 很快分叉。但 observation mismatch 不破坏 policy gradient 正确性：

$$\frac{p_\theta(\tau)}{p_{\theta_{\text{old}}}(\tau)} = \prod_t\frac{\pi_\theta(A_t|s_t)}{\pi_{\theta_{\text{old}}}(A_t|s_t)}$$

环境转移项在 ratio 中抵消。因此：

- 每个 chunk 的 log-prob 在它自己的 observation 上计算（正确）
- Group comparison 只能在 task/reset 对齐的 trajectory level 做（当前实现正是如此）
- 不能把不同 observation 下的 chunks 当成同状态 group 来比较（当前实现不这样做）

---

## 7. Data Structures

### 7.1 ChunkData（`rollout_buffer.py`）

每个 model inference 产生一个 ChunkData：

| 字段 | 类型 | 用途 |
|---|---|---|
| `obs_image` | Tensor [1,3,H,W] | 重算 log-prob 的视觉输入 |
| `obs_proprio` | Tensor [1,D] or None | 重算 log-prob 的本体感知输入 |
| `context` | Tensor [1,L,text_dim] | 文本 context embeddings |
| `context_mask` | Tensor [1,L] | context mask |
| `chain` | Tensor [K+1,H,D] | 完整 denoising 轨迹 |
| `old_log_prob` | Tensor (scalar) | rollout 时旧策略的 block log-likelihood |
| `block_size` | int | ratio 归一化因子。match 模式下为 `replan_steps × K`；slice 模式下为 `exec_horizon × K`；默认为 `H × K` |
| `exec_horizon` | int or None | log_prob 截断的 action 步数。None 表示覆盖全部预测步 |
| `action` | Tensor [H,D] | 最终 denoised action |
| `chunk_rewards` | list[float] | 本 chunk 内每步的环境 reward |
| `done` | bool | 本 chunk 内 episode 是否结束 |
| `task_id` | str | 任务标识 |
| `task_description` | str | 自然语言任务描述 |
| `group_id` | str | group 标识，格式 `{suite}:{task}:update_{step}:batch_{idx}:reset_{idx}` |
| `reset_id` | str | reset 标识 |
| `initial_state_index` | int | 使用的初始状态索引 |
| `trajectory_id` | str | 父 trajectory 标识 |
| `chunk_index` | int | 在 trajectory 中的 chunk 序号 |
| `env_step_start` | int | 本 chunk 起始环境步 |
| `env_step_end` | int | 本 chunk 结束环境步 |
| `rollout_seed` | int or None | 本 chunk 的采样 seed |
| `rollout_time` | float | 相对 trajectory 开始的时间戳 |
| `task_suite_name` | str | benchmark suite 名称 |
| `reward_components` | dict | 命名 reward 分解 |
| `advantage` | float | 由 advantages.py 填入 |

### 7.2 TrajectoryData（`rollout_buffer.py`）

| 字段 | 类型 | 用途 |
|---|---|---|
| `task_id` | str | 任务标识 |
| `task_description` | str | 自然语言任务描述 |
| `chunks` | list[ChunkData] | 有序 chunk 列表 |
| `trajectory_reward` | float | Terminal reward（0 或 1） |
| `success` | bool | 是否成功 |
| `group_id` | str | group 标识 |
| `reset_id` | str | reset 标识 |
| `initial_state_index` | int | 初始状态索引 |
| `trajectory_id` | str | 唯一 trajectory 标识 |
| `rollout_seed` | int or None | 首个 chunk 的 seed |
| `rollout_time` | float | 采集耗时（秒） |
| `task_suite_name` | str | benchmark suite 名称 |
| `reward_components` | dict | 命名 reward 分解 |
| `trajectory_advantage` | float | trajectory-level advantage |

### 7.3 Chain Storage

Flow-GSPO 不能只存最终 action。必须存完整 denoising chain：

$$A_t^0 \rightarrow A_t^{\tau_1} \rightarrow \cdots \rightarrow A_t^1$$

因为训练时 `compute_logprob_from_chain` 沿整条 SDE path 重算 log-prob（带梯度），需要每一步的中间状态。shape 为 `[K+1, H, D]`。

---

## 8. Training Pipeline

### 8.1 Main Loop

`trainer.py:FastWAMRLTrainer.train()` 的完整流程：

```
for global_step in range(max_updates):
    1. t_rollout = perf_counter()
       _collect_rollout_buffer()
         - 选择 task_batch_size 个 task
         - 每个 task 从 _task_state_cursors 轮选 initial_state
         - 每个 (task, reset) 采样 group_size 条 trajectory
         - 每条 trajectory: env.reset(initial_state) → warm-up → SDE denoising loop
         - 存入 RolloutBuffer
       rollout_time = perf_counter() - t_rollout

    2. assign_advantages()
       - 按 group_id 分组
       - 零方差过滤
       - trajectory-level group normalization

    3. compute_rollout_metrics()

    4. t_optim = perf_counter()
       for epoch in range(num_optimization_epochs):
         compute_gspo_objective()
           - 按 variant 选择 chunk-level 或 trajectory-level objective
           - 对每个 chunk/trajectory 调用 compute_logprob_from_chain (带梯度)
           - 计算 ratio, clipped surrogate, KL
         backward + clip_grad_norm + optimizer.step
         - 累积本 epoch 的 metrics
       optim_time = perf_counter() - t_optim
       - 对所有 epoch 的 metrics 取平均

    5. global_step += 1

    6. log_metrics() (每 log_every 步，含 rollout_time_s 和 optim_time_s)
    7. save_checkpoint() (每 save_every 步)
```

多 epoch 指标处理：当 `num_optimization_epochs > 1` 时，所有 epoch 的 grad_norm、clip_fraction、approx_kl、ratio 等指标累积后取平均，而非只保留最后一轮。

### 8.2 Group Sampling

Group 构造为同一 task、同一 reset state 上的 $G$ 条 trajectory。具体实现：

- 每个 task 维护 `_task_state_cursors[task_id]`，轮转选择 `initial_states` 列表中的不同初始状态
- Group 内所有 trajectory 使用相同的 `initial_state`
- `group_id` 包含完整信息：`{suite}:{task}:update_{step}:batch_{idx}:reset_{idx}`
- 不跨 task 混组

### 8.3 Collector Configuration

`RolloutCollector` 根据 `rl.action_horizon` 和 `rl.exec_horizon` 配置：
- `rl.action_horizon`：若非 null，覆盖传入模型的预测步数（方案 match）
- `rl.exec_horizon`：若非 null，传入 SDE 采样函数，控制 log_prob 只覆盖前 N 个 action（方案 slice）

执行步数（`replan_steps`）始终由 `EVALUATION.replan_steps` 控制，不受这两个配置影响。

---

## 9. Checkpoint & Resume

### 9.1 保存内容

每次 checkpoint 保存两部分：

**Weights**（`checkpoints/weights/update_{step}.pt`）：
- Model 权重（通过 `model.save_checkpoint`）

**State**（`checkpoints/state/update_{step}/trainer_state.pt`）：
```python
{
    "optimizer": optimizer.state_dict(),
    "trainer_state": {
        "global_step": int,
        "task_cursor": int,
        "task_state_cursors": {task_id: cursor},
        "variant": str,
        "weights_path": str,
        "collector_sample_counters": {task_id: counter},
        "rng_state": {
            "cpu": torch.get_rng_state(),
            "numpy": np.random.get_state(),
            "random": random.getstate(),
            "cuda": torch.cuda.get_rng_state_all(),  # if available
        },
    },
}
```

### 9.2 Resume 流程

设置 `rl.resume=<state_dir>` 后，trainer 初始化时按以下顺序恢复：

1. **Model weights**：先加载 init checkpoint（`cfg.ckpt`），然后用 resume state 中的 `weights_path` 覆盖。若 `weights_path` 无效，fallback 到按 step-tag 命名规则查找。
2. **Optimizer state**：`optimizer.load_state_dict(payload["optimizer"])`
3. **Trainer state**：global_step, task_cursor, task_state_cursors
4. **RNG state**：恢复 CPU / NumPy / Python random / CUDA 四路全量随机状态（参照 RLinf 模式）
5. **Collector sample counters**：缓存到 `_saved_sample_counters`，在对应 task 的 collector 惰性创建时恢复，保证 seed 序列连续而非从头重放

---

## 10. Monitoring

### 10.1 Rollout Metrics（`metrics.py:compute_rollout_metrics`）

| 指标 | 含义 |
|---|---|
| `num_trajectories` | 本 batch trajectory 数量 |
| `num_chunks` | 本 batch chunk 数量 |
| `success_rate` | 成功率（SDE exploration） |
| `trajectory_reward_mean/std` | trajectory reward 统计 |
| `trajectory_advantage_mean/std` | trajectory advantage 统计 |
| `chunk_advantage_mean/std` | chunk advantage 统计 |
| `chunk_return_mean/std` | chunk 内 reward 总和统计 |
| `chunks_per_trajectory_mean/max` | 每 trajectory chunk 数统计 |
| `informative_group_fraction` | 非零方差 group 占比 |
| `group_reward_std_mean` | group 内 reward std 的均值 |
| `per_task/{task_id}/success_rate` | 每个 task 的成功率 |
| `per_task/{task_id}/num_trajectories` | 每个 task 的 trajectory 数 |

### 10.2 Training Metrics（`objectives.py` + `trainer.py`）

| 指标 | 含义 |
|---|---|
| `num_objective_terms` | 参与 objective 计算的 chunk/trajectory 数 |
| `policy_objective` | clipped surrogate 均值（多 epoch 平均） |
| `approx_kl` | $s - 1 - \log s$ 的均值（多 epoch 平均） |
| `clip_fraction` | ratio 超出 clip 范围的比例（多 epoch 平均） |
| `ratio_mean/min/max` | ratio 统计（多 epoch 平均） |
| `log_ratio_mean` | log ratio 均值（多 epoch 平均） |
| `grad_norm` | 梯度范数（多 epoch 平均） |
| `rollout_time_s` | rollout 采集耗时（秒） |
| `optim_time_s` | 优化耗时（秒） |

### 10.3 Logging Backend & Wandb

通过 `rl.logging.backends` 配置，支持 `wandb` 和 `tensorboard`。也可直接设置 `wandb.enabled=true` 自动启用 wandb（`metric_logger.py:40-41`）。

Wandb 初始化参数来自顶层 `wandb` 配置（`metric_logger.py:64-77`）：

| 字段 | 默认值 | 用途 |
|---|---|---|
| `wandb.enabled` | `false` | 总开关 |
| `wandb.workspace` | `null` | entity（用户/组织名） |
| `wandb.project` | `fast-wam-rl` | 项目名 |
| `wandb.name` | `flow-gspo-${rl.variant}` | run 名称 |
| `wandb.group` | `${EVALUATION.task_suite_name}` | 分组（按 suite） |
| `wandb.mode` | `online` | `online` / `offline` / `disabled` |

### 10.4 Complete Experiment Commands

四个实验，每个需在 4 个 LIBERO suite 上分别运行。以下以 `libero_spatial` 为例：

```bash
CKPT=<pretrained.pt>

# Exp-2a: traj_chunk + match (predict=exec=10)
python scripts/train_rl.py \
    EVALUATION.task_suite_name=libero_spatial \
    rl.variant=traj_chunk rl.action_horizon=10 rl.exec_horizon=null \
    wandb.enabled=true wandb.name=exp2a-match \
    ckpt=$CKPT

# Exp-2b: traj_chunk + slice (predict=32, log_prob 截断到 10)
python scripts/train_rl.py \
    EVALUATION.task_suite_name=libero_spatial \
    rl.variant=traj_chunk rl.action_horizon=null rl.exec_horizon=10 \
    wandb.enabled=true wandb.name=exp2b-slice \
    ckpt=$CKPT

# Exp-3a: traj_traj + match (predict=exec=10)
python scripts/train_rl.py \
    EVALUATION.task_suite_name=libero_spatial \
    rl.variant=traj_traj rl.action_horizon=10 rl.exec_horizon=null \
    wandb.enabled=true wandb.name=exp3a-match \
    ckpt=$CKPT

# Exp-3b: traj_traj + slice (predict=32, log_prob 截断到 10)
python scripts/train_rl.py \
    EVALUATION.task_suite_name=libero_spatial \
    rl.variant=traj_traj rl.action_horizon=null rl.exec_horizon=10 \
    wandb.enabled=true wandb.name=exp3b-slice \
    ckpt=$CKPT
```

换 suite 时改 `EVALUATION.task_suite_name` 为 `libero_object` / `libero_goal` / `libero_10`（Long）。

### 10.5 Final Evaluation

训练中不进行正式评估。最终 benchmark 评估使用 `eval_libero_single.py`，每个 task 跑 50 trials：

```bash
# 对每个 suite 的 10 个 task 分别评估
for TASK_ID in $(seq 0 9); do
    python experiments/libero/eval_libero_single.py \
        ckpt=<rl_checkpoint.pt> \
        EVALUATION.task_suite_name=libero_spatial \
        EVALUATION.task_id=$TASK_ID \
        EVALUATION.num_trials=50
done
```

汇总 suite-level success rate，对比 FastWAM BC baseline（Spatial 98.2 / Object 100.0 / Goal 97.0 / Long 95.2）。

---

## 11. Code Structure

```
src/fastwam/rl/
├── rollout_buffer.py       # ChunkData, TrajectoryData, RolloutBuffer
├── rollout_collector.py    # RolloutCollector: LIBERO env + SDE sampling + chain storage
├── algorithms.py           # FlowGSPOVariant registry: traj_chunk, traj_traj
├── advantages.py           # trajectory advantage: uniform / temporal_decay
├── objectives.py           # chunk-level & trajectory-level GSPO objective
├── metrics.py              # rollout metrics computation
├── metric_logger.py        # wandb/tensorboard backend
└── trainer.py              # FastWAMRLTrainer: single-process RL main loop

src/fastwam/models/wan22/
├── fastwam.py              # infer_action_with_logprob, compute_logprob_from_chain
├── schedulers/
│   └── scheduler_continuous.py  # step_sde_with_logprob (SDE drift + log-prob)
├── action_dit.py           # ActionDiT (action_expert)
└── mot.py                  # MoT (Mixture-of-Transformers)

configs/train_rl_libero.yaml  # Hydra RL 训练配置
scripts/train_rl.py           # RL 训练入口
```

当前为 single-process rollout trainer。`Accelerator` 仅用于 mixed precision 和 gradient 管理，`num_processes` 必须为 1。

---

## 12. Key Configuration

```yaml
rl:
  variant: traj_chunk          # traj_chunk 或 traj_traj
  trainable_scope: action_expert_only  # action_expert_only / action_expert_and_mot / full
  max_updates: 200             # 对齐 OmniVLA-RL
  group_size: 8                # 每 group 采样的 trajectory 数
  task_batch_size: 1           # 每 update 的 task 数
  num_optimization_epochs: 1
  clip_range: 0.2              # PPO clipping epsilon
  kl_coef: 0.01                # KL 正则系数 beta
  learning_rate: 1e-5
  weight_decay: 0.01           # 对齐 OmniVLA-RL
  max_grad_norm: 1.0
  sigma_max: 0.1               # SDE noise upper bound
  num_inference_steps: 10      # denoising steps K

  # Action chunk 消融维度
  action_horizon: null         # null=默认32, int=覆盖预测步数（match 方案）
  exec_horizon: null           # null=覆盖全部, int=log_prob 只算前N步（slice 方案）

  trajectory_assignment: uniform  # uniform / temporal_decay
  advantage_gamma: 0.99
  log_every: 1
  save_every: 10
  resume: null
```
