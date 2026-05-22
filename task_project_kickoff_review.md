# 项目启动梳理

## 1. 当前定位

该仓库当前是论文复现的**任务书/蓝图**，核心内容集中在 `gpt_paper.md`，尚未形成可直接运行的工程。

| 项目 | 当前状态 | 说明 |
| ---- | -------- | ---- |
| 需求文档 | 已有 | `gpt_paper.md` 已给出完整阶段规划 |
| 代码实现 | 缺失 | 尚未发现 `src/`、`scripts/`、`configs/` |
| 环境文件 | 缺失 | 尚未发现 `requirements.txt`、`environment.yml` |
| 数据说明 | 缺失 | 尚未发现 `highD` 获取、放置、预处理说明 |
| 运行入口 | 缺失 | 尚未发现 train/eval 脚本 |
| 结果模板 | 缺失 | 尚未发现 metrics / figures / videos 样例 |

## 2. 复现主线

文档要求的最小闭环为：

1. `highway-env` / 自定义 dense lane-change 环境
2. DQN / PPO baseline
3. 风险指标：TTC / THW / DRAC / gap / density
4. 目标车道后车响应估计：`P_yield` / `P_block`
5. risk-aware reward
6. risk-gated action masking
7. Mosaic-style arbitration
8. evaluation + visualization + paper figures

第一版明确不建议做：

1. 完整 CARLA 训练
2. 复杂 MARL
3. 完整 flow matching 主线
4. nuPlan / PDM 复现
5. 连续控制 SAC / TD3 作为主线

## 3. 阶段优先级

| 优先级 | 模块 | 备注 |
| ------ | ---- | ---- |
| P0 | baseline 跑通 | 先确认 `highway-env + SB3 + DQN/PPO` |
| P1 | 风险指标模块 | TTC / THW / DRAC / density / gap |
| P2 | target rear + game-risk | 先规则函数，不上复杂学习模型 |
| P3 | risk-aware reward | 与环境 reward 深度绑定 |
| P4 | action masking | 安全核心，优先于复杂证明 |
| P5 | arbitration | RL / rule / risk-min 三方仲裁 |
| P6 | full RIGA-LC | 主结果训练与评估 |
| P7 | ablation / robustness / figures | 出论文表格与图 |

## 4. 当前最缺的内容

| 类别 | 缺失项 | 影响 |
| ---- | ------ | ---- |
| 工程骨架 | `src/`、`scripts/`、`configs/` | 无法开始编码 |
| 环境依赖 | `requirements.txt` 或 `environment.yml` | 无法稳定复现环境 |
| 数据约定 | `highD` 目录、许可证、预处理入口 | 后续校准与扩展无法落地 |
| 配置规范 | 训练步数、seed、场景密度、阈值 | 实验不可比 |
| 指标定义 | near-miss、success、waiting、risk exposure 的精确定义 | 表格难以统一 |
| 日志与结果格式 | CSV 字段、目录命名、checkpoint 规则 | 难以复查和出图 |

## 5. 建议先补齐的文件

建议先创建以下最小文件集合：

1. `README.md`
2. `requirements.txt`
3. `configs/highway_default.yaml`
4. `configs/highway_dense.yaml`
5. `configs/risk_params.yaml`
6. `src/envs/highway_custom_env.py`
7. `src/risk/metrics.py`
8. `src/risk/target_gap.py`
9. `src/masking/action_mask.py`
10. `scripts/01_train_baseline_dqn.py`
11. `scripts/05_evaluate_all.py`

## 6. 需要讨论和确认

### A. 指标与定义

1. `success` 的定义是“完成目标换道且无碰撞”，还是还要求保持若干步稳定行驶？
2. `waiting_steps` 的统计起点是什么？从可换道时刻开始，还是从 episode 起点开始？
3. `near-miss` 的阈值用 TTC、THW、DRAC 中哪一个，还是联合判定？
4. `risk_exposure` 是逐步累加的风险分数、超阈值占比，还是取 episode 峰值？

### B. 方法选择

1. `P_yield / P_block` 是否先固定为规则 sigmoid，不做数据驱动拟合？
2. action masking 第一版是否采用“wrapper 外部修正动作”的最简实现？
3. arbitration 第一版是否采用文档中的硬规则逻辑，不先上学习式打分器？

### C. 实验预算

1. 第一轮正式实验用几个 seed？文档建议 5，但可先用 1 做闭环。
2. 第一阶段的完整训练预算，是按 `100k` steps 起步，还是更小规模 warm-up？

### D. 工程规范

1. 配置系统是否统一用 YAML？
2. 结果目录是否严格固定为 `results/metrics`、`results/figures`、`results/videos`？
3. 是否从一开始就强制每个实验输出统一 CSV 字段？

## 7. 推荐执行顺序

1. 明确第一阶段复现边界
2. 创建最小工程骨架
3. 跑通随机策略与 baseline DQN
4. 固化评估 CSV 与可视化导出
5. 再逐步加入 risk / masking / arbitration

## 8. 当前判断

该项目需求方向清晰，论文主线完整，适合按阶段推进；但在正式复现前，仍需先把**复现边界、指标定义、训练预算、最小工程骨架**四件事确认下来，否则后续容易反复返工。

## 9. 已确认决策

根据本轮讨论，当前已确认如下边界：

| 项目 | 决策 | 说明 |
| ---- | ---- | ---- |
| 第一阶段主线 | DQN 最小闭环 | 先不把 PPO 作为第一阶段必做项 |
| 第一阶段环境 | `highway-v0` + `merge-v0` + 1 个自定义 dense lane-change 场景 | 场景范围比最小闭环更完整 |
| `highD` 接入 | 暂不接入 | 第一阶段仅预留接口与参数位 |
| 训练策略 | 先小规模流程测试，再正式训练 | 符合当前项目稳妥推进方式 |

## 10. 下一步建议

建议下一轮直接进入以下落地任务：

1. 创建最小工程骨架与目录
2. 固化 `requirements.txt` 与基础 YAML 配置
3. 先跑通随机策略和 DQN baseline
4. 打通统一评估 CSV 导出
5. 再逐步接入 risk / masking / arbitration
