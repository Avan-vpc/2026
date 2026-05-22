下面这段可以**直接复制给 Trae**，作为完整工程任务书。目标不是一步到位做大系统，而是先做出一个**highway-env 主实验 + 风险感知 RL + 博弈风险评估 + action masking + 仲裁 fallback + 可出论文图表**的稳定框架；CARLA/MetaDrive 只作为后续可视化增强，避免工程风险过高。

---

# 发给 Trae 的完整任务书

## 0. 项目名称

**RIGA-LC: Risk-Informed Game-Augmented Lane-Change Decision-Making**

中文名：

**风险引导的博弈增强换道决策方法**

目标论文方向：

**Risk-Aware Non-Conservative Lane-Change Decision-Making Under Uncertain Mixed Traffic**

核心思想：

```text
强化学习负责学习非保守换道策略；
风险评估负责量化危险程度；
局部博弈模块负责估计目标车道后车让行/阻挡倾向；
action masking 负责阻止危险动作；
规则/MOBIL fallback 负责极端情况下兜底；
highway-env 做主实验，CARLA/MetaDrive 做案例可视化增强。
```

这套路线参考了几类 2025/2026 已发表论文模式：

1. Hu 等 **Risk-Aware Reinforcement Learning for Non-Conservative Motion Planning**：POMDP + Bayesian belief + time-varying risk field + attention + RL，强调不确定环境下的非保守风险感知规划。
2. Hu 等 **Coupled Reinforcement Learning Network Combined With Risk Assessment**：高层意图决策 + 低层行为控制 + 风险评估，属于典型“层级 RL + 风险模块”模式。
3. Wang 等 **Robust Lane Change Decision**：上层 adversarial MARL，下层 yielding/action masking 碰撞规避，用于提升混合交通换道鲁棒性。
4. Deng 等 **Eliminating Uncertainty of Driver’s Social Preferences**：不完全信息博弈 + highD 社会偏好建模 + realistic simulation，可参考其“社会偏好/让行不确定性”处理方式。
5. KIT-MRT **Mosaic**：参考其“rule-based planner + learned planner + centralized trajectory verification/scoring + arbitration graph”的思想，不必直接复现完整工程。Mosaic 的 Composer 会验证并评分候选轨迹，拒绝不安全 proposal，然后在通过验证的轨迹中按 safety gates 和 performance score 选择最优方案。([GitHub][1])

---

# 1. 总体技术路线

## 1.1 最小闭环

先在 **highway-env** 中完成主实验。highway-env 是 Farama 维护的自动驾驶 tactical decision-making 环境集合，适合快速训练 DQN/PPO 等决策模型。([GitHub][2])

完整框架：

```text
highway-env / custom dense lane-change env
        ↓
baseline RL: DQN / PPO
        ↓
风险指标计算：TTC / THW / DRAC / gap / density
        ↓
目标车道后车响应估计：P_yield / P_block
        ↓
risk-aware reward
        ↓
risk-gated action masking
        ↓
Mosaic-style action arbitration:
    RL action
    rule-safe action
    risk-minimizing action
        ↓
final action
        ↓
evaluation + visualization + paper figures
```

## 1.2 不建议第一版做的事情

第一版暂时不要做：

```text
完整 CARLA 训练
复杂 MARL
完整 flow matching
nuPlan / PDM 复现
多车不完全信息博弈
连续控制 TD3/SAC 作为主线
```

原因：工程风险太高，不利于快速出结果。

---

# 2. 推荐开源项目与用途

## 2.1 必用项目

### A. highway-env

用途：

```text
主训练环境；
高速换道、多车道交通、merge 场景；
自定义 reward / observation / action masking。
```

说明：highway-env 提供自动驾驶决策任务集合，文档也强调其面向 autonomous driving decision-making。([HighwayEnv 文档][3])

优先使用环境：

```text
highway-v0
merge-v0
custom dense-lane-change-v0
custom adversarial-lane-change-v0
```

---

### B. Stable-Baselines3

用途：

```text
DQN baseline
PPO baseline
后续可选 SAC / TD3
日志、训练、评估、TensorBoard
```

说明：Stable-Baselines3 是 PyTorch 下可靠 RL 算法实现集合。([稳定基线3文档][4])
DQN 支持离散动作空间，适合 highway-env 原生离散 meta-actions。([稳定基线3文档][5])

---

### C. rl-agents

用途：

```text
参考 highway-env 官方 DQN baseline；
对比 eleurent 原始实现；
借鉴 replay buffer / config / evaluation 结构。
```

highway-env README 中明确提到可使用 `eleurent/rl-agents` 和 Stable-Baselines3 作为 agent 参考。([GitHub][6])

---

## 2.2 强化项目

### D. KIT-MRT/mosaic

用途：

```text
参考“仲裁 + 验证 + 打分”架构；
不一定直接集成；
把思想移植到 highway-env。
```

Mosaic 的核心是 Composer 验证并评分 rule-based 和 learned planner 产生的候选轨迹，拒绝不安全 proposal，再选择得分最高者。([GitHub][1])
其基础思想来自 arbitration graphs：将复杂决策拆成行为组件和仲裁器，行为组件生成 command，仲裁器根据策略选择最优 option。([GitHub][7])

在本项目中对应为：

```text
RL action proposal
rule-based safe action
risk-minimizing action
        ↓
risk verifier
        ↓
action arbitrator
        ↓
final action
```

---

### E. MetaDrive

用途：

```text
第二阶段增强实验；
比 CARLA 更轻；
做可泛化 RL / 多样道路 / 多车交互。
```

MetaDrive 是轻量、可组合驾驶仿真器，支持生成多样道路和交通设置，也支持真实驾驶日志和 RL 研究。([GitHub][8])
官方文档还说明其可生成大量场景，并支持 generalizable RL、多智能体行为建模、安全探索等任务。([MetaDrive][9])

---

### F. CARLA

用途：

```text
只做案例可视化；
不作为主训练平台；
展示 dense lane change / yield / block 三个 case。
```

CARLA 是自动驾驶研究开源仿真器，支持开发、训练和验证自动驾驶系统，并提供开放城市、车辆和环境资产。([GitHub][10])

建议用途：

```text
把 highway-env 中学到/设计好的策略逻辑迁移成简单 Python 控制脚本；
只展示 2–3 个 case；
不追求完整 CARLA RL 训练。
```

---

# 3. 项目目录结构

请按下面结构创建工程：

```text
riga_lc/
├── README.md
├── requirements.txt
├── configs/
│   ├── highway_default.yaml
│   ├── highway_dense.yaml
│   ├── merge_default.yaml
│   ├── risk_params.yaml
│   ├── train_dqn.yaml
│   └── train_ppo.yaml
├── src/
│   ├── envs/
│   │   ├── __init__.py
│   │   ├── highway_custom_env.py
│   │   ├── dense_lane_change_env.py
│   │   └── adversarial_lane_change_env.py
│   ├── risk/
│   │   ├── __init__.py
│   │   ├── metrics.py
│   │   ├── target_gap.py
│   │   ├── game_risk.py
│   │   └── risk_score.py
│   ├── masking/
│   │   ├── __init__.py
│   │   ├── action_mask.py
│   │   └── safety_verifier.py
│   ├── arbitration/
│   │   ├── __init__.py
│   │   ├── rule_policy.py
│   │   ├── risk_min_policy.py
│   │   └── arbitrator.py
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── train_dqn.py
│   │   ├── train_ppo.py
│   │   ├── evaluate.py
│   │   └── callbacks.py
│   ├── analysis/
│   │   ├── metrics_logger.py
│   │   ├── plot_training.py
│   │   ├── plot_ablation.py
│   │   ├── plot_robustness.py
│   │   └── case_visualizer.py
│   └── utils/
│       ├── seed.py
│       ├── config.py
│       └── io.py
├── scripts/
│   ├── 00_smoke_test_env.py
│   ├── 01_train_baseline_dqn.py
│   ├── 02_train_risk_reward_dqn.py
│   ├── 03_train_masked_dqn.py
│   ├── 04_train_full_riga_lc.py
│   ├── 05_evaluate_all.py
│   ├── 06_run_ablation.py
│   ├── 07_run_robustness.py
│   └── 08_export_figures.py
├── results/
│   ├── logs/
│   ├── models/
│   ├── metrics/
│   ├── figures/
│   └── videos/
└── docs/
    ├── method_notes.md
    ├── experiment_plan.md
    └── paper_figures_checklist.md
```

---

# 4. 阶段任务

## Phase 0：环境搭建与 baseline 跑通

### 目标

先跑通 highway-env + Stable-Baselines3 的 DQN/PPO baseline。

### 参考项目

```text
highway-env
Stable-Baselines3
rl-agents
```

### 具体任务

1. 创建 Python 环境。
2. 安装：

```bash
pip install highway-env stable-baselines3[extra] gymnasium matplotlib pandas seaborn pyyaml tensorboard
```

3. 跑通 highway-v0 随机策略。
4. 跑通 DQN baseline。
5. 保存：

```text
episode reward
collision rate
average speed
episode length
lane change count
```

### 输出文件

```text
results/metrics/baseline_dqn.csv
results/models/dqn_baseline.zip
results/figures/phase0_training_reward.png
```

### 可出的图

**Fig. 1：Baseline 训练曲线**

```text
x-axis: training steps
y-axis: episode reward
curves: DQN, PPO
```

**Fig. 2：baseline 行为截图 / GIF**

```text
highway-env 场景
ego vehicle 高亮
周车蓝色
```

---

## Phase 1：风险指标计算模块

### 目标

为每一步状态计算交通风险指标。

### 需要实现

文件：

```text
src/risk/metrics.py
src/risk/target_gap.py
src/risk/risk_score.py
```

### 风险指标

实现以下函数：

```python
compute_ttc(ego, veh)
compute_thw(ego, veh)
compute_drac(ego, veh)
compute_relative_speed(ego, veh)
compute_gap_size(ego, front, rear)
compute_local_density(vehicles, ego, radius=80.0)
```

### 指标定义

#### TTC

```text
如果前后车正在接近，则 TTC = distance / closing_speed
否则 TTC = inf
```

#### THW

```text
THW = longitudinal distance / ego_speed
```

#### DRAC / required deceleration

```text
DRAC = closing_speed^2 / (2 * distance)
```

这个指标用于评估后车或 ego 为避免碰撞所需减速度，比单纯 TTC 更有物理解释性。

### 风险分数

先用可解释加权式：

```python
risk = (
    w_ttc * 1 / clipped_ttc
    + w_thw * 1 / clipped_thw
    + w_drac * clipped_drac
    + w_density * density
)
```

### 输出文件

```text
results/metrics/risk_debug_samples.csv
```

字段：

```text
episode, step, action, ttc_min, thw_min, drac_max, density, risk_score, collision
```

### 可出的图

**Fig. 3：风险指标随时间变化**

```text
case: 一次换道 episode
曲线：TTC_min, THW_min, DRAC_max, risk_score
标记：换道动作发生时刻
```

**Fig. 4：碰撞/非碰撞样本的风险分布**

```text
boxplot:
collision episodes vs non-collision episodes
指标：risk_score, TTC_min, DRAC_max
```

---

## Phase 2：目标车道后车识别与轻量博弈风险模块

### 目标

识别当前换道目标车道的后车，并估计其让行/阻挡倾向。

### 参考论文

Deng 2025 使用不完全信息博弈处理换道中周车社会偏好不确定性，并用 highD 训练 GMM 风险和 SIDM 环境车。本文只借鉴“后车偏好/响应不确定性”思想，不复刻完整模型。

### 需要实现

文件：

```text
src/risk/target_gap.py
src/risk/game_risk.py
```

### 目标车道后车识别

输入：

```text
ego lane
target lane
all vehicles
```

输出：

```text
target_front_vehicle
target_rear_vehicle
gap_size
rear_relative_speed
rear_acceleration
```

### 响应概率

先实现规则/逻辑函数，不训练复杂模型：

```python
P_yield = sigmoid(
    a0
    + a1 * gap_size
    + a2 * ttc_rear
    - a3 * rear_closing_speed
    - a4 * density
    - a5 * rear_aggressive_indicator
)
P_block = 1 - P_yield
```

其中：

```text
gap_size 越大，P_yield 越高
rear closing speed 越大，P_block 越高
density 越高，P_block 越高
rear acceleration > 0，P_block 越高
```

### game-risk 输出

```text
P_yield
P_block
rear_response_risk
target_gap_risk
```

### 可出的图

**Fig. 5：P_yield 与 gap_size / relative_speed 的关系热力图**

```text
x-axis: gap size
y-axis: rear relative speed
color: P_yield
```

**Fig. 6：yield/block case 可视化**

```text
case A: 后车减速让行，P_yield 上升
case B: 后车加速阻挡，P_block 上升
```

---

## Phase 3：Risk-aware reward

### 目标

将风险指标和非保守性指标加入 reward，形成 risk-aware RL。

### 参考论文

Hu 的 Risk-Aware RL 论文强调 POMDP、不确定性建模、风险场与 RL 结合，用于 non-conservative motion planning。
Hu 的 Coupled RL 论文把风险评估与层级 RL 结合，追求风险与效率平衡。

### 需要实现

文件：

```text
src/envs/highway_custom_env.py
src/envs/dense_lane_change_env.py
```

### reward 设计

```python
reward = (
    speed_reward
    + progress_reward
    + lane_change_success_reward
    - collision_penalty
    - risk_penalty
    - drac_penalty
    - unnecessary_waiting_penalty
    - action_oscillation_penalty
)
```

### 新增 reward 项

#### risk_penalty

```text
risk_score 越高，惩罚越大
```

#### unnecessary_waiting_penalty

目标：

```text
避免车辆在安全/可协商 gap 中一直不换道
```

定义：

```text
如果存在 safe/negotiable target gap 且 ego 长时间 keep lane，则惩罚
```

#### rear_disturbance_penalty

如果换道导致后车大幅减速：

```text
rear deceleration < -threshold
```

则惩罚。

### 实验对比

训练：

```text
DQN baseline
DQN + risk reward
DQN + risk reward + game-risk observation
```

### 可出的图

**Fig. 7：不同 reward 版本训练曲线**

```text
DQN baseline
DQN + risk reward
DQN + risk reward + game-risk
```

**Fig. 8：不同 reward 版本安全-效率散点图**

```text
x-axis: collision rate / risk exposure
y-axis: average speed / success rate
```

---

## Phase 4：Action masking / safety shield

### 目标

阻止 RL 选择明显危险动作。
这是安全核心，不要把安全完全交给 reward。

### 参考论文

Wang 2025 robust lane-change 论文中，下层使用 yielding 和 action masking 机制降低碰撞风险。

### 需要实现

文件：

```text
src/masking/action_mask.py
src/masking/safety_verifier.py
```

### Action mask 逻辑

对每个候选动作：

```text
keep lane
lane left
lane right
faster
slower
```

做短时预测。

如果执行某个动作后：

```text
TTC < TTC_min
THW < THW_min
DRAC > DRAC_max
target gap risk > risk_threshold
predicted collision = True
```

则 mask 掉该动作。

### fallback

如果所有换道动作都被 mask：

```text
选择 keep lane 或 slower
```

如果所有动作都危险：

```text
选择 slower
```

### 与 SB3 兼容方式

第一版简单实现：

```text
不改 SB3 内部网络；
在 env.step(action) 前外部 wrapper 修正 action；
如果 agent 选了 unsafe action，则替换为 fallback action，并记录 mask_event。
```

后续增强：

```text
实现 ActionMaskingWrapper；
或切换到支持 action masking 的算法/自定义 policy。
```

### 记录字段

```text
raw_action
final_action
was_masked
mask_reason
risk_before_action
```

### 可出的图

**Fig. 9：action masking 前后危险动作比例**

```text
bar chart:
raw unsafe action rate
masked unsafe action rate
final unsafe action rate
```

**Fig. 10：无 mask vs 有 mask 的碰撞率对比**

```text
DQN
DQN + risk reward
DQN + risk reward + mask
```

---

## Phase 5：Mosaic-style 仲裁器

### 目标

实现一个简化的 arbitration framework：RL、规则策略、风险最小策略同时给建议，仲裁器选择最终动作。

### 参考项目

Mosaic 通过 arbitration graphs 组合 rule-based 与 learned planners，并用 centralized verification/scoring 选择轨迹。([GitHub][1])
Arbitration Graphs 将复杂决策拆成行为组件和仲裁器。([GitHub][7])

### 需要实现

文件：

```text
src/arbitration/rule_policy.py
src/arbitration/risk_min_policy.py
src/arbitration/arbitrator.py
```

### 三类候选 action

#### 1. RL action

来自 DQN/PPO。

#### 2. Rule-safe action

规则策略：

```text
如果目标 gap 安全 → lane change
否则 keep / slower
```

#### 3. Risk-minimizing action

对所有动作做风险预测，选择 risk 最低的动作。

### 仲裁逻辑

```python
if rl_action is safe and risk_score < threshold:
    final_action = rl_action
elif rule_action is safe:
    final_action = rule_action
else:
    final_action = risk_min_action
```

增强版：

```python
score(action) = 
    w_efficiency * efficiency_score
    + w_safety * safety_score
    + w_comfort * comfort_score
    + w_nonconservative * nonconservative_score
```

### 记录字段

```text
rl_action
rule_action
risk_min_action
final_action
chosen_source: rl / rule / risk_min
arbitration_reason
```

### 可出的图

**Fig. 11：仲裁器选择来源比例**

```text
pie/bar chart:
RL selected %
Rule selected %
Risk-min selected %
```

**Fig. 12：Mosaic-style 框架图**

```text
RL proposal
Rule proposal
Risk-min proposal
        ↓
Verifier
        ↓
Arbitrator
        ↓
Final action
```

---

## Phase 6：主方法 RIGA-LC 训练

### 目标

训练完整方法：

```text
DQN/PPO + game-risk observation + risk reward + action masking + arbitration fallback
```

### 推荐算法

第一优先：

```text
DQN / Dueling DQN
```

第二优先：

```text
PPO
```

暂缓：

```text
SAC / TD3
```

原因：highway-env 原生动作是离散 meta-actions，DQN/PPO 更稳。SB3 的算法表说明 DQN 支持 Discrete action，PPO 同时支持多种 action space。([稳定基线3文档][11])

### 训练设置

建议：

```text
seeds: 5
training steps: 100k / 300k / 500k
evaluation episodes: 100 or 200 per seed
traffic densities: low / medium / high
```

保存：

```text
best model
final model
training logs
evaluation metrics
videos
```

### 可出的图

**Fig. 13：主方法训练曲线**

```text
DQN
DQN + risk
DQN + risk + mask
RIGA-LC full
```

**Fig. 14：主方法安全/效率对比柱状图**

```text
collision rate
lane-change success rate
average speed
unnecessary waiting
```

---

## Phase 7：消融实验

### 目标

证明每个模块有效。

### 消融组

```text
DQN baseline
DQN + risk reward
DQN + risk observation
DQN + game-risk
DQN + action masking
DQN + risk reward + masking
Full RIGA-LC
Full w/o game-risk
Full w/o masking
Full w/o arbitration
Full w/o unnecessary waiting penalty
```

### 输出表格

**Table 1：主结果表**

列：

```text
Method
Reward
Collision Rate ↓
Near-miss Rate ↓
Success Rate ↑
Avg Speed ↑
Waiting Steps ↓
TTC Violation ↓
DRAC Violation ↓
Masked Action Rate
```

**Table 2：消融结果表**

列：

```text
Variant
Collision Rate
Success Rate
Avg Reward
Risk Exposure
Unnecessary Waiting
Rear Disturbance
```

### 可出的图

**Fig. 15：消融柱状图**

```text
collision rate
success rate
waiting steps
risk exposure
```

**Fig. 16：安全-效率 Pareto 图**

```text
x-axis: risk exposure / collision rate
y-axis: average speed / success rate
```

---

## Phase 8：鲁棒性实验

### 目标

模拟不确定混合交通和“对抗式”后车行为。

### 场景扰动

实现：

```text
traffic density: low / medium / high
aggressive vehicle ratio: 0%, 25%, 50%, 75%
target rear acceleration disturbance
front vehicle sudden braking
perception noise on relative distance/speed
random lane change by surrounding vehicles
```

### 参考论文

Wang 的 robust lane-change 论文关注混合交通中对抗扰动和复杂交通条件下的鲁棒性。
Hu Risk-Aware RL 论文强调低速密集、高速稀疏等不同交通条件下的不确定性泛化。

### 输出表格

**Table 3：鲁棒性结果表**

列：

```text
Scenario
Method
Collision Rate
Success Rate
Avg Reward
Performance Degradation
Recovery Rate
```

### 可出的图

**Fig. 17：交通密度变化下性能退化曲线**

```text
x-axis: density
y-axis: collision rate / success rate
```

**Fig. 18：aggressive ratio 下的鲁棒性曲线**

```text
x-axis: aggressive vehicle ratio
y-axis: success rate / collision rate
```

---

## Phase 9：典型案例可视化

### 目标

做出论文中最能说服人的 case 图。

### highway-env case

保存 GIF / png：

```text
Case 1：规则方法过度保守，RIGA-LC 成功换道
Case 2：普通 DQN 激进碰撞，RIGA-LC 被 mask 阻止
Case 3：目标后车让行，RIGA-LC 利用 negotiable gap
Case 4：目标后车加速阻挡，RIGA-LC fallback keep/slower
```

### 可出的图

**Fig. 19：四个典型场景轨迹图**

每个 case 画：

```text
ego trajectory
target rear trajectory
target front trajectory
lane boundary
lane-change start/end
risk score curve
```

**Fig. 20：case 时间序列图**

```text
top: vehicle positions
middle: P_yield / P_block
bottom: action / risk / mask event
```

---

## Phase 10：CARLA 或 MetaDrive 增强展示

### 推荐顺序

优先 MetaDrive，后 CARLA。

MetaDrive 更轻量，可生成多样道路和交通设置，并支持 RL 和多智能体行为建模。([GitHub][8])
CARLA 更适合最终展示图，但工程成本更高。CARLA 支持自动驾驶系统开发、训练和验证，并提供开放数字资产。([GitHub][10])

### MetaDrive 任务

```text
复现 2 个 highway-env 中的策略逻辑；
不一定训练；
展示 rule vs RIGA-LC。
```

### CARLA 任务

```text
只做 2–3 个可视化 case；
不训练 RL；
用 Python 脚本控制 ego；
背景车设置为 yield / block。
```

### 可出的图

**Fig. 21：CARLA dense lane-change case**

```text
ego 尝试换道
target rear yield
成功完成换道
```

**Fig. 22：CARLA block case**

```text
target rear 加速
RIGA-LC 不执行换道 / 减速等待
```

---

# 5. 论文图表清单

最终至少需要这些图表。

## 图

```text
Fig. 1 方法总框架图
Fig. 2 highway-env 实验场景示意
Fig. 3 风险指标定义图：TTC / THW / DRAC / target gap
Fig. 4 P_yield / P_block 热力图
Fig. 5 risk-aware reward 结构图
Fig. 6 action masking 机制图
Fig. 7 Mosaic-style arbitration 框架图
Fig. 8 training reward curves
Fig. 9 collision rate comparison
Fig. 10 lane-change success rate comparison
Fig. 11 unnecessary waiting comparison
Fig. 12 safety-efficiency Pareto plot
Fig. 13 ablation results
Fig. 14 robustness under traffic density
Fig. 15 robustness under aggressive rear vehicles
Fig. 16 typical highway-env trajectory case
Fig. 17 P_yield / risk / action time-series case
Fig. 18 CARLA or MetaDrive visualization case
```

## 表

```text
Table I Reference methods and modules
Table II Environment configuration
Table III Reward function components
Table IV Main comparison results
Table V Ablation study
Table VI Robustness test
Table VII Runtime analysis
```

---

# 6. 论文预期主结果

预期完整方法不需要所有指标第一，但应当呈现下面趋势：

```text
相比 MOBIL / TTC rule：
    waiting steps 更少，lane-change success 更高。

相比 DQN / PPO：
    collision rate 更低，TTC/DRAC violation 更少。

相比 risk reward only：
    action masking 显著降低危险动作。

相比 masking only：
    game-risk estimator 提高 negotiable gap 下的效率。

相比 w/o arbitration：
    极端风险场景下 fallback 更稳定。
```

论文核心表述：

```text
RIGA-LC achieves a better safety-efficiency balance:
it avoids unnecessary conservatism in negotiable gaps while preventing unsafe lane-change actions through risk-gated masking and arbitration fallback.
```

中文：

```text
RIGA-LC 在安全与效率之间取得更好的平衡：
在可协商间隙中减少不必要保守，同时通过风险屏蔽和仲裁兜底避免危险换道动作。
```

---

# 7. 交付要求

## 每个阶段都必须保存

```text
代码
配置文件
训练日志
评估 CSV
图像
可复现实验命令
README 说明
```

## 每个实验都必须支持

```bash
python scripts/05_evaluate_all.py --config configs/highway_dense.yaml --model results/models/xxx.zip --episodes 100 --seed 0
```

## 所有结果保存为

```text
results/metrics/*.csv
results/figures/*.png
results/videos/*.gif
```

## 每个 CSV 至少包含

```text
method
seed
episode
reward
collision
success
avg_speed
waiting_steps
lane_change_count
ttc_min
thw_min
drac_max
risk_exposure
masked_action_count
chosen_source
```

---

# 8. 优先级排序

## 必须完成

```text
1. highway-env baseline
2. 风险指标 TTC / THW / DRAC
3. risk-aware reward
4. target rear P_yield / P_block
5. action masking
6. RIGA-LC full method
7. main comparison
8. ablation
9. robustness
10. figures and tables
```

## 有时间再做

```text
1. PPO / SAC / TD3 baseline
2. MetaDrive 复现
3. CARLA 可视化
4. flow matching scenario augmentation
5. highD 数据驱动校准 P_yield / P_block
```

## 暂时不要做

```text
1. 全量 CARLA RL 训练
2. 多车 MARL
3. nuPlan
4. 复杂 flow matching 轨迹生成主线
5. 端到端视觉输入
```

---

# 9. 最终给 Trae 的执行重点

请按下面原则执行：

```text
先跑通，再加模块；
先 highway-env，再 CARLA；
先 DQN/PPO，再考虑 SAC/TD3；
先规则风险模型，再考虑数据驱动校准；
先 action masking，再做复杂安全证明；
先出图表，再做大规模扩展。
```

目标不是搭一个巨大平台，而是做出一篇结构清晰的论文实验系统：

```text
风险感知
不确定性
博弈增强
强化学习
action masking
arbitration fallback
non-conservative lane change
highway-env statistics
CARLA/MetaDrive visualization
```

这套系统如果按阶段完成，已经具备写成 **2026 年应用型 Q1 风格论文** 的基本形态。

[1]: https://github.com/KIT-MRT/mosaic?utm_source=chatgpt.com "KIT-MRT/mosaic: An Extensible Framework for Composing ..."
[2]: https://github.com/Farama-Foundation/HighwayEnv?utm_source=chatgpt.com "Highway-Env"
[3]: https://highway-env.farama.org/index.html?utm_source=chatgpt.com "HighwayEnv Documentation"
[4]: https://stable-baselines3.readthedocs.io/?utm_source=chatgpt.com "Stable-Baselines3 Docs - Reliable Reinforcement Learning ..."
[5]: https://stable-baselines3.readthedocs.io/en/master/modules/dqn.html?utm_source=chatgpt.com "DQN — Stable Baselines3 2.9.0a2 documentation"
[6]: https://raw.githubusercontent.com/Farama-Foundation/HighwayEnv/master/README.md?utm_source=chatgpt.com "https://raw.githubusercontent.com/Farama-Foundatio..."
[7]: https://github.com/KIT-MRT/arbitration_graphs?utm_source=chatgpt.com "Arbitration Graphs"
[8]: https://github.com/metadriverse/metadrive?utm_source=chatgpt.com "MetaDrive: Lightweight driving simulator for everyone"
[9]: https://metadrive-simulator.readthedocs.io/?utm_source=chatgpt.com "MetaDrive Documentation — MetaDrive 0.1.1 documentation"
[10]: https://github.com/carla-simulator/carla?utm_source=chatgpt.com "CARLA simulator"
[11]: https://stable-baselines3.readthedocs.io/en/master/guide/algos.html?utm_source=chatgpt.com "RL Algorithms — Stable Baselines3 2.9.0a2 documentation"
