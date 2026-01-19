<p align="center">
  <a href="../README.md">English</a> · 
  <strong>中文</strong> · 
  <a href="README.pl.md">Polski</a> ·
</p>


# RobotSumo RL 训练系统

本项目实现了一个自主的 RobotSumo 战斗代理，使用强化学习（Actor-Critic 架构）进行训练。系统采用了一个专门的训练环境，具备 **自我对战机制**，学习代理可以与“Master”模型或其历史版本进行对战，从而不断进化和优化其战斗策略。

主要特性包括一个复杂的 **奖励塑形引擎**，旨在促进积极的前进运动、精准瞄准和战略性场地定位，同时惩罚被动行为，如原地旋转或倒退行驶。

### *实时战斗演示及奖励追踪*

https://github.com/user-attachments/assets/ca0baaf4-f6bf-412e-9ca7-3786b3346c5d
<p align="center">
  <em>SAC 代理（绿色） vs A2C 代理（蓝色）</em>
</p>

https://github.com/user-attachments/assets/2b496931-9eda-4c8b-88ca-7286d5fa9b42
<p align="center">
  <em>SAC 代理（绿色） vs PPO 代理（蓝色）</em>
</p>

https://github.com/user-attachments/assets/bdabd7a4-4890-47b2-a4cf-d7549b31da2e
<p align="center">
  <em>A2C 代理（绿色） vs PPO 代理（蓝色）</em>
</p>


## 系统架构

下图展示了闭环控制系统。它区分了 **移动机器人**（物理/传感层）和 **RL 控制器**（决策层）。请注意，目标信号 $\mathbf{r}_t$ 仅在训练阶段使用，通过奖励引擎来塑造策略。

<div align="center">
  <img src="../resources/control_loop.png" width="650px">
</div>

### 功能模块

* **控制器（RL 策略）：** 基于神经网络的代理（例如 SAC、PPO 或 A2C），将当前观测向量映射到连续动作空间。在部署阶段，它作为推理引擎运行。
* **动力学（Dynamics）：** 表示机器人的二阶物理模型。它计算对输入力和力矩的响应，考虑质量、惯性矩和摩擦，同时受外部 **扰动**（SAT 碰撞）的影响。
* **运动学（Kinematics）：** 状态积分模块，将广义速度转换为全局坐标。它保持机器人相对于竞技场原点的姿态。
* **传感器融合（感知层，Sensor Fusion）：** 预处理层，将机器人状态向量、原始全局状态数据以及环境信息（例如对手位置）转换为标准化的自我中心观测向量。

### 信号向量

各模块之间的通信由以下数学向量定义：

* $\mathbf{r}_t$：**奖励/目标信号** – 仅在训练中使用，用于通过奖励塑形函数指导策略优化。
* $\mathbf{a}_t = [v\_{target}, \omega\_{target}]^T$：**动作向量** – 控制命令，表示期望的线速度和角速度。
* $\dot{\mathbf{x}}_t = [\dot{x}, \dot{y}, \dot{\theta}]^T$：**状态导数** – 动力学引擎计算的瞬时广义速度。
* $\mathbf{y}_t = [x, y, \theta]^T$：**物理输出（姿态）** – 机器人在全局坐标系中的当前位置和朝向。
* $\mathbf{s}_t$：**观测向量（`state_vec`）** – 一个 11 维标准化特征向量，包含本体感受线索（速度）和外部空间关系（与对手/边缘的距离）。

## 状态向量规范

输入状态向量（`state_vec`）由 11 个标准化值组成，为代理提供对竞技场情况的全面视图：

| 索引 | 参数 | 描述 | 范围 | 来源 / 传感器 |
| :--- | :--- | :--- | :--- | :--- |
| 0 | `v_linear` | 机器人线速度（前进/后退） | [-1.0, 1.0] | 轮编码器 / IMU 融合 |
| 1 | `v_side` | 机器人横向速度 | [-1.0, 1.0] | IMU（加速度计） / 状态估计 |
| 2 | `omega` | 旋转速度 | [-1.0, 1.0] | 轮编码器 / 陀螺仪（IMU） |
| 3 | `pos_x` | 竞技场 X 坐标 | [-1.0, 1.0] | 里程计 / 定位融合 |
| 4 | `pos_y` | 竞技场 Y 坐标 | [-1.0, 1.0] | 里程计 / 定位融合 |
| 5 | `dist_opp` | 与对手的标准化距离 | [0.0, 1.0] | 距离传感器（红外/超声波） / LiDAR |
| 6 | `sin_to_opp` | 对手角度的正弦值 | [-1.0, 1.0] | 几何计算（基于距离传感器） |
| 7 | `cos_to_opp` | 对手角度的余弦值 | [-1.0, 1.0] | 几何计算（基于距离传感器） |
| 8 | `dist_edge` | 与最近竞技场边缘的距离 | [0.0, 1.0] | 地面传感器（线检测器） / 几何计算 |
| 9 | `sin_to_center` | 相对于竞技场中心的方向正弦值 | [-1.0, 1.0] | 线传感器 / 状态估计 + 几何计算 |
| 10 | `cos_to_center` | 相对于竞技场中心的方向余弦值 | [-1.0, 1.0] | 线传感器 / 状态估计 + 几何计算 |

## 奖励塑形细节

奖励系统旨在强化激进的战斗策略和战略生存：

* **终局奖励（Terminal Rewards）：** 胜利获得高额奖励，掉出场地或超时（平局）将受到严重惩罚。  
* **后退阻止（Backward Block）：** 严格惩罚倒退行为，并取消该步的其他奖励。
* **防旋转（Anti-Spinning）：** 对过度旋转进行惩罚，防止无目的旋转。
* **前进奖励（Forward Progress）：** 前进奖励根据目标精度（面向对手）进行加权。
* **动能交锋（Kinetic Engagement）：** 保持正向速度并正对对手时给予高额奖励，鼓励果断攻击。
* **边缘安全（Edge Safety）：** 主动逻辑惩罚向“深渊”移动的行为，并奖励回到竞技场中心。
* **战斗动力学（Combat Dynamics）：** 高速正面碰撞（推挤）奖励，侧面或背面被撞则惩罚。
* **效率（Efficiency）：** 每步固定时间惩罚，鼓励尽快获胜。

## 环境规范

仿真环境旨在高度还原官方 RobotSumo 比赛标准：

* **竞技场（Arena / Dohyo）：** 按标准半径（77 cm）建模，并定义中心点。环境严格执行边界条件；当机器人底盘任意角超过 `ARENA_DIAMETER_M` 时，比赛立即结束（终局状态）。
* **机器人物理（Robot Physics）：**  
    * **底盘（Chassis）：** 机器人遵循 10x10 cm 的方形尺寸（`ROBOT_SIDE`）。  
    * **动力学（Dynamics）：** 系统实现基于质量的加速度、转动惯量和摩擦模型（包括模拟轮胎抓地力的侧向摩擦）。
* **碰撞系统（Collision System）：** 实时接触处理采用 **分离轴定理（SAT）**。它计算非弹性重叠并施加物理冲量，影响机器人基于质量和恢复系数的前向和横向速度。
* **起始条件（Start Conditions）：** 标准起始距离约为竞技场半径的 70%，支持固定位置和随机 360°朝向，以增强训练稳健性。


## 性能分析与基准测试

比赛结果清晰地展示了战斗策略的演进以及不同强化学习架构的效率。比较显示出峰值性能和收敛速度的明显层次结构。

### 比赛排行榜与效率

| 排名 | 代理 | 模型版本 | ELO 评分 | 所需训练回合数 |
|:----:|:-----:|:--------------:|:----------:|:-----------------:|
| 1-5  | **SAC** | v19 - v23 | **1391 - 1614** | **~378** |
| 6-10 | **PPO** | v41 - v45 | **1128 - 1342** | **~1,049** |
| 11-15| **A2C** | v423 - v427| **791 - 949** | **10,000 - 24,604**|

> [!NOTE]
> **关于收敛速度的说明：** 各架构的样本效率存在巨大差异。SAC 在达到峰值潜力时明显更早，大约只需要 PPO 的 **1/3** 回合数，以及 A2C 的 **1/60** 回合数即可收敛到熟练的战斗水平。

### 顶级模型比较
*比较每种架构表现最优的版本（最终迭代）。*

<table width="100%">
  <tr>
    <td align="center">
      <img src="../resources/peak_elo_comparison_algos.png" width="800px"><br>
      <em>按算法划分的峰值 ELO</em>
    </td>
  </tr>
  <tr>
    <td align="center">
      <img src="../resources/peak_elo_ranking_models.png" width="800px"><br>
      <em>顶级模型</em>
    </td>
  </tr>
</table>

---

### 进化过程（Evolutionary Progress）
*对整个学习过程中定期采样的模型性能进行分析（每种架构 5 个阶段）。*

<table width="100%">
  <tr>
    <td align="center">
      <img src="../resources/sampled_elo_comparison_algos.png" width="800px"><br>
      <em>按算法划分的采样模型平均 ELO</em>
    </td>
  </tr>
  <tr>
    <td align="center">
      <img src="../resources/sampled_elo_ranking_models.png" width="800px"><br>
      <em>采样模型</em>
    </td>
  </tr>
</table>

### 关键结论（Key Takeaways）

* **SAC（Soft Actor-Critic）效率：** SAC 在此环境中无可争议地表现最佳。其离策略最大熵框架使其以最佳样本效率达到最高技能上限（1614 ELO）。
    * *行为说明：* SAC 代理在被扰动时能够巧妙地恢复自身朝向，并积极利用对手即使是微小的定位失误。
* **PPO 稳定性与战术：** PPO 依然是可靠的竞争者，提供稳定的训练和具有竞争力的性能。尽管其峰值 ELO 低于 SAC，但仍是连续控制任务中的稳健选择。
    * *行为说明：* 有趣的是，PPO 代理在“贴身纠缠”情境中表现出色，学会了战术性操作，在近身接触中打乱对手姿态以获取位置优势。
* **A2C 性能差距：** 基本 Advantage Actor-Critic 算法在样本效率和稳定性方面存在显著不足。即使经过大量训练，其表现仍低于更先进架构的起始 ELO，凸显了简单的在策略方法在此任务中的局限性。
* **架构演进：** 本项目强调，现代离策略方法（SAC）在**连续、非线性控制任务**中远优于传统在策略方法。SAC 能够在离策略数据学习中最大化熵，从而实现更复杂、自适应的战斗行为，并显著提高性能上限。


## 简单入门（Simple Start）

要运行仿真并观察代理的动作，请按照以下步骤操作：

### 安装
```bash
make install
```
### 快速演示（交叉对战，例如 SAC vs PPO）
```bash
make cross-play
```

### 其他命令
```bash
make train-sac        # 开始全新 SAC 训练（清除旧模型）
make train-ppo        # 开始全新 PPO 训练（清除旧模型）
make train-a2c        # 开始全新 A2C 训练（清除旧模型）
make test-sac         # 运行专用 SAC 测试脚本
make test-ppo         # 运行专用 PPO 测试脚本
make test-a2c         # 运行专用 A2C 测试脚本
make tournament       # 自动选择前 5 名训练模型并进行 ELO 排名
make clean-models     # 删除所有训练历史和主模型
```
*有关可用自动化目标的完整列表，请参阅 [Makefile](../Makefile).*


## 未来潜在改进（Future Potential Improvements）

* **观测噪声注入**（Observation Noise Injection）：为激光雷达和里程计传感器实现高斯噪声模型，以模拟真实世界传感器的随机性，从而促进策略的泛化能力和稳健性。

* **输入状态扩展**（Input State Expansion）：基于最近的激光雷达样本，扩展输入状态向量，估算对手速度，以改进预测性战斗操作。

* **高级物理建模**（Advanced Physics Modeling）：实现非线性动力学，如车轮打滑、线速度-角速度饱和和电机饱和，以更好地模拟现实物理约束并提高仿真到真实的转化潜力（Sim-to-Real）。

* **自动化分析与统计**（Automated Analytics & Statistics）：创建脚本分析模型决策并生成详细指标（例如每轮平均步数、旋转频率，以及特定碰撞类型，如追尾或侧面碰撞）。

* **消融研究**（Ablation Studies）：对奖励塑形函数进行参数化，开展消融实验，单独评估各组成部分（如定位 vs 攻击性）对 SAC 和 PPO 稳定性及收敛性的贡献。

* **评估框架与回归测试**（Evaluation Harnesses & Regression Testing）：开发一套固定战术场景（如边缘恢复挑战、特定起始朝向）作为回归测试套件，确保新模型版本在优化更高 ELO 的同时不丢失基础技能。

## 引用（Citation）

如果本仓库在您的研究中有所帮助，欢迎引用：

**APA 格式**
> Brzustowicz, S. (2026). RobotSumo-RL: 使用 SAC、PPO、A2C 算法的相扑机器人强化学习 (版本 1.0.0) [源码]. https://github.com/sebastianbrzustowicz/RobotSumo-RL

**BibTeX**
```bibtex
@software{brzustowicz_robotsumo_rl_2026,
  author = {Sebastian Brzustowicz},
  title = {RobotSumo-RL: 使用 SAC、PPO、A2C 算法的相扑机器人强化学习},
  url = {https://github.com/sebastianbrzustowicz/RobotSumo-RL},
  version = {1.0.0},
  year = {2026}
}
```
> [!TIP]
> 您也可以使用侧边栏的 **“Cite this repository”** 按钮自动复制引用或下载原始元数据文件。

## 许可（License）

RobotSumo-RL 源码可用许可（禁止 AI 使用）。
完整条款和限制请参见 [LICENSE](../LICENSE) 文件。

## 作者（Author）

Sebastian Brzustowicz &lt;Se.Brzustowicz@gmail.com&gt;
