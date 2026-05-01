# FastWAM + Flow-GSPO RL: Project Context, Engineering Plan, and Borrowed Design Notes

## 1. Project Summary

本项目的目标是：

- 以 `FastWAM` 作为基座；
- 冻结 `video DiT / world representation`；
- 只对 `action head` 及其必要的 action-side modules 做在线 RL；
- 采用 `Flow-GSPO` 系列方法完成三组消融：
  - `Exp-1`: 原版 `block-level Flow-GSPO`
  - `Exp-2`: `trajectory-level advantage + chunk-level ratio`
  - `Exp-3`: `trajectory-level advantage + trajectory-level ratio`

当前阶段的任务不是证明某一个单独主方法，而是把三种实验都接入同一套可维护、可扩展、可监控的 RL infra。

---

## 2. One-Sentence Problem Definition

`OmniVLA-RL` 已经提供了 `flow action block` 的随机化采样与 `block likelihood / ratio` 建模方式，但 `FastWAM` 面临的是 `multi-chunk closed-loop control`，因此必须重新定义 reward/advantage 的归因单位，并将三种 ablation 放入统一的 rollout-train pipeline。

---

## 3. Hard Constraints

### 3.1 Model-side constraint

训练时优先冻结：

- `video_expert`
- `VAE`
- `text_encoder`
- world-side representation path

第一阶段只优化：

- `action_expert`
- 必要时的 `proprio_encoder`
- 必要时的少量 `MoT` / action-side adapter / LoRA

原因：

- 避免 sparse reward 直接破坏 world representation；
- 使实验结论聚焦于 `WAM representation + action-head RL`；
- 降低 credit assignment 与 representation drift 耦合。

### 3.2 Algorithm-side constraint

三个实验必须共用：

- 同一套 rollout 数据结构；
- 同一套 `collect -> assign advantage -> compute ratio -> optimize -> log` 框架；
- 同一入口脚本与同一配置系统。

不允许为每个实验复制一份 trainer 或 rollout 代码。

### 3.3 Infrastructure-side constraint

代码组织必须优先考虑：

- 简洁；
- 层次清晰；
- 便于在集群上扩展；
- 后续可自然升级到多进程 / async rollout。

---

## 4. Algorithmic Context

### 4.1 What is inherited from OmniVLA-RL

可直接继承的部分：

- 将 flow matching 的 ODE 改写为 SDE；
- 每个 denoising transition 的高斯形式；
- `action block likelihood` 的定义；
- `block-size normalized importance ratio`。

形式上，单个 action block

$$
A_t = [a_{t,0}, \dots, a_{t,H-1}]
$$

的似然为

$$
\pi_\theta(A_t \mid z_t)
=
\prod_{\tau=0}^{K-1}
p_\theta(A_t^{\tau+\delta} \mid A_t^\tau, z_t),
$$

对应 chunk-level ratio 为

$$
\rho_{i,t}^{\text{chunk}}
=
\left(
\frac{\pi_\theta(A_{i,t}\mid z_{i,t})}
{\pi_{\theta_{\text{old}}}(A_{i,t}\mid z_{i,t})}
\right)^{1/(HK)}.
$$

这里：

- `H` 是 action horizon；
- `K` 是 denoising steps。

### 4.2 What must be changed for FastWAM

`OmniVLA-RL` 的原版 advantage 是：

- 同一个状态下采样多个候选 action blocks；
- 用 block reward 做 group normalization。

这在 `FastWAM` 的长程闭环任务里不够，因为：

- success 往往在多个 chunks 之后才出现；
- 中间 block 可能没有稳定的局部 reward；
- 不同 trajectory 中间 observation 很快分叉。

因此 `FastWAM` 必须显式区分：

- `likelihood unit`
- `ratio unit`
- `reward / advantage unit`

---

## 5. Three Required Ablations

### 5.1 Exp-1: Original Block-Level Flow-GSPO

定义：

- reward / advantage: `block-level`
- ratio: `chunk-level`

用途：

- 作为原论文迁移基线；
- 验证 `FastWAM action head` 是否可直接受益于原始 Flow-GSPO。

局限：

- 依赖 block reward；
- 对纯 terminal success 任务可能不足。

### 5.2 Exp-2: Trajectory-Level Advantage + Chunk-Level Ratio

定义：

- reward / advantage: `trajectory-level`
- ratio: `chunk-level`

这是最重要的实验。

形式上，对同 task / same reset distribution 的 $G$ 条 trajectory，

$$
\hat{A}_i^{\text{traj}}
=
\frac{R_i^{\text{traj}}-\mu_R}{\sigma_R+\epsilon},
$$

并将其分配给每个 chunk：

$$
\hat{A}_{i,t} = \hat{A}_i^{\text{traj}}
$$

或

$$
\hat{A}_{i,t} = \gamma^{N_i-t}\hat{A}_i^{\text{traj}}.
$$

用途：

- 解决长程 credit assignment；
- 保留 chunk-level clipping 的稳定性。

### 5.3 Exp-3: Trajectory-Level Advantage + Trajectory-Level Ratio

定义：

- reward / advantage: `trajectory-level`
- ratio: `trajectory-level`

形式上，

$$
\rho_i^{\text{traj}}
=
\left(
\frac{
\prod_t \pi_\theta(A_{i,t}\mid z_{i,t})
}{
\prod_t \pi_{\theta_{\text{old}}}(A_{i,t}\mid z_{i,t})
}
\right)^{1/(\sum_t HK)}.
$$

用途：

- 检验“reward unit 与 ratio unit 完全统一”是否值得。

风险：

- 方差更高；
- clipping 更容易过强；
- 单个坏 chunk 会污染整条 trajectory ratio。

---

## 6. Final Theoretical Conclusion on Observation Mismatch

必须明确：

- `observation mismatch` 不会破坏 policy gradient；
- 它破坏的是伪造的 `same-state group comparison`。

原因是：

$$
\frac{p_\theta(\tau)}{p_{\theta_{\text{old}}}(\tau)}
=
\prod_t
\frac{\pi_\theta(A_t\mid s_t)}
{\pi_{\theta_{\text{old}}}(A_t\mid s_t)},
$$

环境转移项会抵消。

因此：

- 每个 chunk 的 log-prob 应在它自己的 observation 上计算；
- group comparison 只能在 task/reset 对齐的 trajectory level 做；
- 不能把不同 observation 下的 chunks 当成同状态 group 来比较。

---

## 7. Current FastWAM RL Code Status

当前仓库内已经落下的 RL 相关代码主要在：

- `src/fastwam/rl/rollout_buffer.py`
- `src/fastwam/rl/advantages.py`
- `src/fastwam/rl/rollout_collector.py`
- `src/fastwam/rl/algorithms.py`
- `src/fastwam/rl/objectives.py`
- `src/fastwam/rl/metrics.py`
- `src/fastwam/rl/metric_logger.py`
- `src/fastwam/rl/trainer.py`
- `configs/train_rl_libero.yaml`
- `scripts/train_rl.py`

这些文件的当前职责为：

- `rollout_buffer.py`: 轨迹/块级数据结构；
- `advantages.py`: block / trajectory advantage assignment；
- `rollout_collector.py`: LIBERO rollout + SDE sampling + denoising chain 存储；
- `algorithms.py`: 三种 ablation 的配置注册；
- `objectives.py`: chunk-level 与 trajectory-level objective；
- `metrics.py`: rollout / optimization 指标；
- `metric_logger.py`: `wandb/tensorboard` 后端；
- `trainer.py`: 单进程 RL 主循环；
- `train_rl_libero.yaml`: Hydra 训练配置；
- `train_rl.py`: RL 训练入口。

注意：

- 当前实现是 `single-process rollout trainer`；
- 结构已经按后续并行 rollout 可扩展方向拆开；
- 还不是 `RLinf` 风格的多 worker / async scheduler 系统。

---

## 8. Rollout Data: What Must Be Stored

这是集群上继续工作的关键上下文。

### 8.1 Minimum chunk-level storage

每个 action chunk 至少要存：

- `obs_image`
- `obs_proprio`
- `context`
- `context_mask`
- `chain`
- `old_log_prob`
- `block_size`
- `action`
- `chunk_rewards`
- `done`
- `task_id`
- `task_description`
- `advantage`

这些字段的意义分别是：

- `obs_* / context*`: 重新计算当前策略 log-prob 所需；
- `chain`: 完整去噪路径，用于 `compute_logprob_from_chain`；
- `old_log_prob`: rollout 时旧策略的 block likelihood；
- `block_size`: 做 $(HK)^{-1}$ 归一化；
- `chunk_rewards`: `Exp-1` 所需的 block reward；
- `task_id`: trajectory group normalization 所需；
- `advantage`: assignment 之后供优化使用。

### 8.2 Minimum trajectory-level storage

每条 trajectory 至少要存：

- `task_id`
- `task_description`
- `chunks`
- `trajectory_reward`
- `success`
- `trajectory_advantage`

### 8.3 Strongly recommended extra fields

后续为了更稳健，建议补充：

- `reset_id`
- `initial_state_index`
- `trajectory_id`
- `chunk_index`
- `env_step_start`
- `env_step_end`
- `seed`
- `rollout_time`
- `task_suite_name`
- `reward_components`

其中 `reward_components` 尤其重要，因为后续可能从纯 terminal reward 扩展到：

- success reward
- progress reward
- stage reward
- alignment reward
- safety penalty

若不单独存，会影响实验可解释性。

### 8.4 Why chain storage is non-negotiable

对 Flow-GSPO，不能只存最终 action。

必须存：

$$
A_t^0 \rightarrow A_t^{\tau_1} \rightarrow \cdots \rightarrow A_t^1
$$

的完整 denoising chain，因为新策略的 block likelihood 是沿整条 SDE path 重算的，而不是只依赖最终 denoised action。

---

## 9. Recommended Training Loop

推荐流程固定为：

1. 选择 task batch；
2. 对每个 task 采样 `group_size=G` 条 trajectory；
3. 存入 `RolloutBuffer`；
4. 依据 variant 做 advantage assignment；
5. 依据 variant 做 ratio aggregation；
6. 计算 clipped objective + KL regularization；
7. 记录 rollout 和训练指标；
8. checkpoint；
9. 进入下一 update。

这个循环中，`collector` 不应知道当前跑的是哪个实验。

只有以下两层知道 variant：

- `advantage assignment`
- `objective / ratio aggregation`

---

## 10. Efficient Inference and Rollout Infrastructure

### 10.1 Basic efficiency principle

真正昂贵的部分是：

- environment interaction；
- video-side encoding；
- repeated action denoising；
- reward computation。

因此工程上必须避免无意义重复。

### 10.2 Immediate efficiency decisions

当前最合理的策略是：

- 冻结 `video DiT`；
- rollout 时每个 observation 只做一次 video prefill；
- action ratio 重算尽量只走 action-side forward；
- 维持短期 on-policy buffer；
- group 按 task/reset 构造，不跨任务乱拼。

### 10.3 Video KV cache reuse

`FastWAM` 当前已经有：

- rollout-time `infer_action_with_logprob`
- update-time `compute_logprob_from_chain`

二者都依赖 video-side prefill / cache。

后续优化重点：

- rollout 阶段复用 observation 对应的 video KV cache；
- update 阶段减少重复构图；
- 尽量让 ratio 重算只重新前向 action branch。

### 10.4 Group sampling strategy

优先级如下：

1. 同一个 reset state 上采样多条 trajectory；
2. 若做不到，则同 task / 同分布 reset；
3. 不要跨 task 混组。

### 10.5 Parallelism roadmap

短期：

- 保持单进程 rollout；
- 先验证三种 ablation 的正确性。

中期：

- 每个 task 一个 env worker；
- 将 rollout collector 从 trainer 中拆出；
- 用进程级并行采样不同 task groups。

长期：

- 向 `RLinf` 风格靠拢；
- 独立 actor / rollout / env / reward worker；
- 支持 async rollout 与 weight sync。

---

## 11. Monitoring and Logging Requirements

### 11.1 Minimum acceptable monitoring

必须记录：

- `rollout/success_rate`
- `rollout/trajectory_reward_mean`
- `rollout/trajectory_reward_std`
- `rollout/chunk_return_mean`
- `rollout/chunk_return_std`
- `rollout/informative_group_fraction`
- `rollout/group_reward_std_mean`
- `train/policy_objective`
- `train/approx_kl`
- `train/clip_fraction`
- `train/ratio_mean`
- `train/ratio_min`
- `train/ratio_max`
- `train/log_ratio_mean`
- `train/grad_norm`

若跑 `Exp-3`，还应特别关注：

- `chunks_per_trajectory_mean`
- trajectory-level clip fraction 是否快速升高；
- ratio 是否更快塌缩到 clipping 边界。

### 11.2 W&B configuration

当前仓库的训练配置已支持 `wandb` 风格字段：

- `wandb.enabled`
- `wandb.workspace`
- `wandb.project`
- `wandb.name`
- `wandb.group`
- `wandb.mode`

RL trainer 里另外加了一层 `rl.logging.backends`，可选：

- `wandb`
- `tensorboard`

推荐默认行为：

- 集群联机训练：`wandb`
- 无外网或调试：`tensorboard`

### 11.3 What to log as artifacts

除 scalar 指标外，建议后续增加：

- sampled rollout video
- per-task success table
- checkpoint metadata
- config snapshot
- variant name
- reward breakdown histogram

---

## 12. External Repositories: What to Borrow

本项目后续仍会继续借鉴：

- `repo/RLinf`
- `repo/ReinFlow`
- `repo/flow_grpo`

下面是当前最重要的借鉴点。

### 12.1 Borrowing from RLinf

最值得借鉴的不是具体算法，而是 infra 结构。

关键文件：

- `RLinf/rlinf/runners/embodied_runner.py`
- `RLinf/rlinf/utils/metric_logger.py`
- `RLinf/rlinf/utils/metric_utils.py`
- `RLinf/rlinf/utils/distributed.py`

应借鉴的核心点：

1. `runner / worker / logger` 解耦
2. rollout metrics 聚合与分 rank 日志
3. 多后端 logger 抽象
4. async logging / async worker 的组织方式
5. 将时间统计、rollout 指标、训练指标分层汇总

对本项目的直接启发：

- `FastWAMRLTrainer` 现在只是单进程版本；
- 后续应把 `collector`、`reward`、`actor update` 的角色继续拆开；
- 尤其要借鉴 `MetricLogger` + `rollout metrics aggregation` 的思路。

不应直接照搬的部分：

- `RLinf` 偏通用大规模 RL 系统；
- 当前 FastWAM 阶段不需要一开始就上完整多 worker scheduler；
- 先验证算法，再逐步扩展 infra。

### 12.2 Borrowing from ReinFlow

关键文件：

- `ReinFlow/script/run.py`
- `ReinFlow/agent/finetune/dppo/train_ppo_diffusion_agent.py`

应借鉴的核心点：

1. 基于 config 的统一实验入口
2. rollout 数据和 update 逻辑的清晰分离
3. diffusion / flow 类策略中保存 chain 的必要性
4. 训练时对 value/logprob 重算的 batch 化处理
5. reward / success / episode summary 的清晰统计

`train_ppo_diffusion_agent.py` 对本项目特别有用，因为它体现了：

- 如何把环境 rollout 收集成固定结构；
- 如何存 `chains_trajs`；
- 如何在 update 阶段分 batch 计算 logprob / value；
- 如何组织 eval/train 切换。

对本项目的直接启发：

- `FastWAM` 应坚持 `rollout buffer` 的显式结构化；
- 后续若 trajectory 很长，ratio 重算也应支持 batch split；
- 训练 summary 需要区分 chunk 粒度与 episode 粒度。

### 12.3 Borrowing from flow_grpo

关键文件：

- `flow_grpo/scripts/train_wan2_1.py`
- `flow_grpo/flow_grpo/stat_tracking.py`
- `flow_grpo/flow_grpo/diffusers_patch/*_with_logprob.py`
- `flow_grpo/flow_grpo/rewards.py`

应借鉴的核心点：

1. `pipeline_with_logprob` 的实现方式
2. SDE step with logprob 的工程写法
3. `Accelerator + wandb` 的训练日志接入
4. group-based reward normalization / stat tracking
5. reward function 与 sampling process 的解耦

对本项目最直接的帮助有两点：

第一，`with_logprob` 系列 patch 展示了如何把生成 pipeline 改造成：

- 返回 sample；
- 返回 full latent / chain；
- 返回 logprob。

这和 `FastWAM` 里 `infer_action_with_logprob` / `compute_logprob_from_chain` 的思路高度一致。

第二，`PerPromptStatTracker` 明确体现了：

- group-based normalization 是独立模块；
- group 的统计状态可以显式维护；
- informative group 的比例本身值得监控。

不应直接照搬的部分：

- `flow_grpo` 的 grouping 单位是 prompt/image generation；
- `FastWAM` 这里的 grouping 单位是 `task/reset/trajectory`；
- 不能机械套用 prompt-based grouping。

---

## 13. Recommended Engineering Principles Going Forward

### 13.1 Keep collector algorithm-agnostic

`rollout_collector.py` 不应该知道当前是：

- `Exp-1`
- `Exp-2`
- `Exp-3`

它只负责：

- rollout；
- 调 action inference；
- 存 chain / old logprob / rewards / task ids。

### 13.2 Keep objective switch minimal

实验切换只应改：

- `advantage_mode`
- `ratio_mode`

不要让 variant 决定：

- collector 行为；
- env API；
- logging 接口；
- checkpoint 格式。

### 13.3 Treat reward as first-class structure

后续若加入：

- progress reward
- stage reward
- VLM process reward
- alignment reward

必须将它们结构化保存，而不是只存一个最终浮点数。

### 13.4 Prefer resumable configs over ad hoc scripts

集群运行时，一切都应优先通过配置字段表达：

- variant
- task suite
- task ids
- trainable scope
- reward mode
- logging backend
- resume checkpoint

不应靠手改脚本切换实验。

---

## 14. Immediate TODOs for the Cluster Workspace

另一工作空间接手后，建议按以下顺序继续：

1. 确认 `FastWAM`、`RLinf`、`ReinFlow`、`flow_grpo` 都已在集群上拉取。
2. 以本文件和 `src/fastwam/rl/*` 为主上下文，检查当前 RL trainer 是否满足运行环境依赖。
3. 先跑最小单任务 smoke test：
   - `rl.variant=block`
   - `task_batch_size=1`
   - `group_size=2`
   - `max_updates` 很小
4. 再跑三种实验的最小版：
   - `block`
   - `traj_chunk`
   - `traj_traj`
5. 观察 rollout metrics 与 ratio metrics 是否合理。
6. 若单进程版本稳定，再开始考虑并行 rollout。

---

## 15. Final Operating Conclusion

当前项目的正确方向不是“只实现一个主方法”，而是：

- 以 `FastWAM` 冻结的 world representation 为基础；
- 用统一 infra 支持三种 `Flow-GSPO` 消融；
- 用 `chain-aware flow likelihood` 做 ratio；
- 用 task/reset 对齐的 grouping 做 advantage；
- 用干净的 rollout/logging/checkpoint 抽象，为后续集群扩展保留空间。

如果另一个工作空间只读一个文件来理解项目，应该优先读本文件，其次再读：

- `src/fastwam/rl/trainer.py`
- `src/fastwam/rl/objectives.py`
- `src/fastwam/rl/rollout_collector.py`
- `configs/train_rl_libero.yaml`
