# NVIDIA V2D Challenge 调研报告
### 从人类视频到物理可落地的机器人数据 · 供 Louis 评委参考

> 说明：本报告服务对象为 V2D Challenge 的 Industry Innovation Award 评委（Louis Lian, Lightwheel）。报告从三个视角组织——举办方（官方数据与相机标定深挖）、参赛方（方法、流程与 roadmap）、评委（四维 rubric 与难度划分）。除标注「待确认」的内容外，全部事实来自本地已下载的官方数据手册、官方 starter toolkit、评委 Onboarding PDF 与官网公开页，来源列在文末。时间基准：2026-10-09。

---

## 一、一页结论（给 Louis）

- V2D（Video to Data）Challenge 的官方定位是「From Human Video to Grounded Data」，用三个耦合赛道在同一套任务集上，把「人类演示视频 → 可训练机器人数据」这条链路端到端地测一遍。它补的是两件事：measurement（不孤立评测各阶段，而是同一任务上端到端定位断点、追踪误差传播）和 scale（在小样本上开发、在更大 held-out 数据上验证泛化）。
- 三个赛道：Track 1 Reconstruction（单目第三人称视频 → 4D 人-物交互重建）；Track 2 Robotic Grounding（4D HOI → 机器人策略，embodiment 和仿真器固定）；Track 3 Egocentric（第一人称视频 → 端到端策略，方法无关）。
- 每赛道 3 个奖共 9 个：Track Winner、Runner-Up 按排行榜指标自动产生；Industry Innovation Award 独立于排行榜，由特邀行业评审团按四维 rubric 打分。Louis 的工作是后者。
- 作为评委要清楚两件事：第一，排行榜指标（chamfer/plausibility、CHORD 四指标、eval 脚本打包结果）决定 Winner/Runner-Up，评委不参与；第二，评委真正行使判断力的杠杆是 Industry Innovation Award 的四维 rubric。
- 时间线上有一个口径冲突需要在 10/12 会前确认：评委 Onboarding PPT 写 Leaderboard 冻结 11/04，官网公开页与官方 slide 写 11/10。

---

## 二、比赛全貌

### 2.1 赛道定义（官方口径）

| Track | 输入 → 输出 | 一句话 | Starter Toolkit |
|---|---|---|---|
| 1 Reconstruction | 单目第三人称 RGB 视频 → 4D HOI | 能否从单目视频重建 4D 人-物交互？ | video_to_data/reconstruction |
| 2 Robotic Grounding | 第三人称 4D HOI 轨迹 → 策略 | 能容忍多少重建误差？ | video_to_data/robotic_grounding |
| 3 Egocentric | 第一人称视频 → 策略 | 能否从第一人称视频整体重建并迁移？ | nvidia-isaac/video_to_data |

三个赛道共享同一批操作任务（shared task suites），只是从不同入口测这条链路。

![V2D 官方赛道与奖项总览](V2D1.jpg)

*图：V2D1.jpg，官方 slide——三赛道、每赛道三奖、时间线（9/21 启动、11/10 冻结口径）。*

### 2.2 奖项结构

- 每赛道：Track Winner（排行榜第一）、Track Runner-Up（排行榜第二）——纯指标，无需人工评审。
- Industry Innovation Award：独立于排行榜名次，任何有有效最终提交 + 技术报告的队伍（任意赛道）都有资格，由特邀行业评审团按发布时的公开准则评判。准则的两种表述见 4.1。
- 获奖者受邀在 CoRL 2026 tutorial 上做 5 分钟展示；获奖方案将获得大尺度评测并被上游集成进官方代码库；获奖队成为赛后技术报告的核心作者，所有有效提交队受邀合著。

### 2.3 时间线（含口径冲突）

| 事件 | 评委 Onboarding PPT | 官网公开页 / 官方 slide |
|---|---|---|
| 报名开启 | NOW – Through Challenge | NOW – 9/21 |
| 数据发布 / 正式启动 | 9/21 | 9/21 |
| Leaderboard 冻结 | **11/04 周三 5pm ET** | **11/10** |
| 独立打分 + 书面理由截止 | 11/06 周五 5pm ET | — |
| 调解电话会 | 11/08 周日 ~90min（PPT 两版：7pm ET / 2pm ET，待确认） | — |
| CoRL 2026 颁奖 | 11/12 周四 | 11/12 |

冻结日期（11/04 vs 11/10）为两处口径冲突，需在 10/12 评委口径会上确认。

排行榜规则：每赛道一个排行榜，数据发布即开启直到冻结；每队每周最多提交 5 次，最后三天不限次数。评价脚本为官方固定：Track 1 用 eval_reconstruction.py、Track 2 用 eval_robotic_grounding.py、Track 3 用 eval_e2e.py。

---

## 三、举办方视角：官方数据深挖

### 3.1 总体设计思路

- 三个赛道刻意「耦合」：Track 1 的重建输出就是 Track 2 Tier 3 的输入；Track 2 的 Tier 2 噪声是「按 Track 1 误差分布采样」合成的。这意味着官方想测的是误差如何沿链路传播、下游策略能吸收多少——而不是各阶段孤立的最好成绩。
- 数据发布在 9/21，冻结 11/10（或 11/04，待确认），开发窗口约 5–7 周。评委在 10/12 口径会、11/06 打分前，都有充足时间用自己的实验对照参赛方案。

### 3.2 Track 1 数据（单目第三人称重建）

来源：本地 `video_to_data_challenge/track_1/README.md`。

- 来源：FORM-HOI 多视角录制的 30 个精选 episode；每个 episode 只给一个静态第三人称 RGB 视频。
- 规模：30 episodes，16,563 帧（每段 360–877，均值 552），30 fps，22 条原始动作描述，4 个不同物理相机，10 个目标物体（每物体 3 段）。
- 视频规格：1536×1152 H.264（yuv420p），30 fps，无音轨；与源视频字节一致，未重编码。
- 相机分布：back_stereo_camera_left ×5、front_stereo_camera_left ×9、left_stereo_camera_left ×11、right_stereo_camera_left ×5。
- 元数据：只含 LeRobot v2.1 结构列（timestamp, frame_index, episode_index, index, task_index, next.done）。**无人体位姿、无物体位姿、无可见性标注、无物体网格、无 GT**——官方 held-out GT 不随公开数据发布，本地的重建结果无法直接复算官方 chamfer 分数。
- 相机标定：**不提供**。参赛者必须从单目线索自行估计内参与尺度（官方基线做法：GeoCalib 估计内参+重力方向，公制尺度用 SMPL-X 骨长先验 + ground plane 锚定）。

Track 1 的难点定位（官方）：重遮挡、双手协调（bimanual）、长时序（long horizon）。

![Track 1 数据审计总览](outputs/figs/track1_overview.png)

*图：outputs/figs/track1_overview.png，本机数据审计生成。*

### 3.3 Track 2 数据（机器人落地）

来源：本地 `video_to_data_challenge/track_2/` + 本机前期审计（`outputs/report/executive_summary.md`）。

- 三档输入：
  - Tier 1 Clean multi-view capture：干净多视角，视为上游重建的上界。
  - Tier 2 Synthetic corruption：jitter（抖动）、dropout（丢失）、contact error（接触误差），按 Track 1 误差分布采样合成。
  - Tier 3 Off-the-shelf reconstruction：当前最优现成重建方法产出的轨迹（由参赛者自己重建喂入）。
- 本地发布内容：每档 30 个配对 episode（tier1 与 tier2 episode 顺序一致，可做帧对齐的噪声传播分析），每档 30 个视频、29 个 GLB 网格、30 个地面平面 JSON、30 个 G1 支撑面。
- embodiment 与仿真器固定：Unitree G1 全身 + Dex3 灵巧手，Isaac Lab。参赛者竞争的维度是 retarget 与策略学习，不是仿真工程。
- 本机前期实测（CPU 代理实验，非官方分数）：tier2 相对 tier1 原始线性加速度中位放大 1.78×，平滑后 1.40×；这是官方噪声量级的可复算参照。

![Track 2 噪声对比](outputs/figs/track2_noise_comparison.png)

*图：outputs/figs/track2_noise_comparison.png，本机噪声放大 vs 分布散点。*

### 3.4 Track 3 数据（第一人称端到端）

来源：本地 `video_to_data_challenge/track_3/README.md` + `track_3/public/meta/camera_calibration.json`。

- 采集：Vicon 动捕 + 两台 OAK 相机 rig。每个被跟踪物体都有带纹理的 3D 扫描。
- 规模：20 episodes，5,246 帧（均值 262），名义 20.25 fps，10 个任务，3 路 ego 流 / 60 clips，10 个物体（各带纹理网格）。
- 位姿数据：parquet 7 列，`pose = [x, y, z, qw, qx, qy, qz]`（米制，四元数 w 在前），语义是 `world_T_object`，且是在物体扫描自身坐标系里的位姿——把网格放到记录位姿上，就正好在真实物体位置。
- 关键约定：
  - 物体名即网格文件夹名（white_pot → mesh/white_pot/white_pot.glb）。
  - x/y 按 episode 内可见物体位置的均值居中，z 不居中（桌面高度约 0.61 m 保持物理真实）；offset 不存，episode 间无公共原点。
  - visible=false 时位姿全零（零四元数，不是旋转）——用前必须先过滤 visible。

![Track 3 数据审计总览](outputs/figs/track3_overview.png)

![Track 3 加速度分布](outputs/figs/track3_acceleration.png)

*图：outputs/figs/track3_overview.png 与 track3_acceleration.png，本机数据审计/加速度代理分析生成。*

#### 3.4.1 相机标定细节（本报告最细的一层）

标定来源为 OAK 出厂标定（device.readCalibration），单位米。内参为外参矩阵的 fx/fy/cx/cy，畸变系数为 Perspective 模型。

| 流 | 分辨率 | 类型 | fx≈fy | cx | cy | hfov |
|---|---|---|---|---|---|---|
| ego_cam_a | 2028×1520 | 彩色，头戴 | 1149.8 | 994.8 | 767.5 | 108° |
| ego_cam_b | 1280×800 | 单色，立体对右 | 567.3 | 635.6 | 397.5 | 127° |
| ego_cam_c | 1280×800 | 单色，立体对左 | 563.2 | 625.2 | 399.2 | 127° |

- 立体对：ego 的左右为 cam_c/cam_b（因 180° 旋转使基线方向镜像，左右已互换），baseline 0.075 m；标定含 left_to_right、左/右矫正旋转矩阵。可用 cv2.stereoRectify + StereoSGBM 离线矫正，深度公式 depth_m = fx × baseline_m / disparity_px。
- 跨设备时间同步：两个 OAK 设备**不是硬件同步**，ego 设备相对时钟偏差 mean 7.39 ms / max 7.41 ms（记录在 cross_device_skew_ms）。
- 最关键的一条：**该标定文件里没有「相机到动捕世界（mocap world）」的外参**。因此不能直接把网格投影进图像，参赛者必须先解出 ego 相机与动捕世界的相对位姿（本机实验也确认了这一点）。这是 Track 3 显式重建路线最容易低估的坑。
- 另注：exo 流被移入备份，但标定文件保留了 exo rig 的标定（含 exo_cam_b/c 立体对）。公开发布只含 ego 流。

### 3.5 举办方视角小结：数据/标定的三个「信息差」

1. Track 1 完全无标定无 GT——重建质量只能用 held-out chamfer + plausibility 判，本地只能做代理（本机 M4 实验在可见帧上测得 mean trans err 0.077 m / 姿态 err 9.9°）。
2. Track 3 给了内参与立体外参，但缺 ego→动捕世界外参——「有标定」不等于「能直接用」。
3. Track 2 的噪声档是按 Track 1 误差分布生成的——官方把「上游误差」显式做成了比赛的一个输入维度。

---

## 四、参赛方视角：方法与流程

### 4.1 每个赛道参赛者到底要做什么

**Track 1（重建）**：单目视频 + object_prompt（tasks.jsonl）→ 输出人体（SMPL-X/手 MANO）+ 物体位姿轨迹 + 物体几何 + 公制尺度、世界系一致。评估两轴等权：
- Accuracy：到多视角人体网格的 chamfer、到多视角物体网格的 chamfer。
- Physical Plausibility：关节加速度误差、物体加速度误差、接触穿透误差（均相对多视角参照）。

**Track 2（落地）**：给定三档 4D HOI 轨迹 → retarget 到 G1+Dex3 → 在 Isaac Lab 学策略。评估每档的 AUC、SP-SR、MP-SR、MPPE（CHORD 定义），更难档权重更高。

**Track 3（端到端）**：3 路 ego 视频 → 策略。方法无关：显式（重建 + retarget）或隐式（预训练 VLA/WAM）都行。用 eval_e2e.py 打包重建结果 + 策略评估录像。

### 4.2 重建方法（Track 1 的核心，也是 Track 2 Tier 3 的输入生产者）

| 路线 | 代表组件 | 优点 | 坑 | V2D 角色 |
|---|---|---|---|---|
| 经典几何链 | Grounding-DINO→SAM2→FoundationPose→MoGe/DA→CuSFM→BundleSDF | 每步可查错、官方基线同构、可解释 | 尺度错则位姿漂移；加速度/接触轴纯视觉弱 | 官方 MV 基线的同族组件，参赛差异在「精修」 |
| 前馈重建 | VGGT-Ω / Depth Anything 3 / SAM3D / SAM3D-Body | 一次前向替代整条 SfM，速度快 | 新模型指标口径未在 V2D 验证 | 差异化机会：替换官方 CuSFM 相机链路 |
| 3DGS 精修 | gsplat / 3DGS | 高保真网格+纹理 | 需较好相机位姿，单目下要先跟踪后建图 | 把 Plausibility 轴拉回来的差异化点 |
| 数据侧 | FoundationStereo、AprilTag/手眼标定 | 立体精度高 | 需双相机+标定，参赛单目用不上 | 只做自查参照 |

要点：感知是必要条件不是得分条件。Track 1 一半分数在物理合理性（加速度 + 接触穿透），这是动力学问题不是纯视觉问题，纯感知链不自带物理约束，必须加时序平滑、SOMA-X 式全身拟合、接触/穿透后处理。

![重建方法对比](outputs/figs/method_comparison.png)

*图：outputs/figs/method_comparison.png，本机 CPU 代理对比（M1 经典 SfM / M2 深度融合 / M3 前馈网格 / M4 已知 CAD 基线）。*

### 4.3 策略训练方法（Track 2 / Track 3，回答「算法不只是 PPO」）

官方 starter toolkit（github.com/nvidia-isaac/video_to_data）里 robotic_grounding 给的实底：

- 策略层用 Isaac Lab 2.3.1 + RSL-RL（PPO）做 RL 训练，官方提供 `train.py --task Sharpa-V2D-v0` 一类的入口；另有 FlashCHORD 模块用 PPO + FlashSAC 做接触丰富参考跟踪训练（加速训练，官方目标是把训练从「天」压到「小时」）。
- 隐式端到端路线（Track 3）则落在 VLA / WAM / 基础模型：RDT-1B 类模型、GR00T N1.7 微调（toolkit 里有 groot_finetune 模块：采集专家演示 → LeRobot 数据集转换 → GR00T N1.7 微调 → 闭环评估）。
- 所以「除了 PPO 找不到算法」是因为只看策略层：重建层有一整条视觉算法链（检测/分割/深度/位姿/几何），策略层除 PPO 还有 FlashSAC、VLA/WAM 微调、GR00T 微调。

### 4.4 「跑策略的脚本谁写」——给参赛者与评委的准确答案

官方提供的不是「大家跑同一个脚本」，而是：

- 基础设施固定：G1+Dex3 embodiment、Isaac Lab、starter toolkit（含基线、train.py、retarget 工具、reconstruction 的 18 个容器化模块、E2E 工作流 run_e2e.sh）。
- 官方提供 eval 脚本（eval_reconstruction / eval_robotic_grounding / eval_e2e.py）作为**提交打包的唯一通道**——即「评测协议」统一。
- 但 retarget 的具体方案、策略学习的具体算法与配置、误差注入 / 域随机化的设计，都开放给参赛者自己写。参赛竞争的正是这一层。

所以更准确的表述是：评测脚本统一（官方写），训练流水线是「官方给基线 + 参赛者自研」。

补两个澄清：

仿真任务（你指的 benchmark）也是官方提供的。starter toolkit 的 robotic_grounding 里包含 Isaac Lab 环境 + 固定 embodiment（G1/Dex3）+ 任务定义（如 Sharpa-V2D-v0）+ RL 训练入口（scripts/rsl_rl/train.py，含域随机化 domain_randomization.py、eval.py、rollout 导出）。官方口径是「embodiment 与仿真器固定并提供，参赛者比拼 retarget 与策略学习，而不是仿真工程」。选手不写仿真任务本身，但可以在固定环境上改观测、奖励与训练配置——竞争点在策略和 retarget，不在搭环境。

VLA / WAM（Track 3 隐式路线）不是完全从零自训。官方提供端到端 baseline 与微调工具链：toolkit 的 groot_finetune 模块做「采集专家演示 → LeRobot 数据集转换 → GR00T N1.7 微调 → 闭环评估」；参赛者基于预训练基础模型（GR00T N1.7、RDT-1B 等）微调。真正开放的是选显式重建+retarget 还是隐式端到端、选哪个基础模型、怎么微调。

### 4.5 Roadmap（官方 slide V2D2 口径）

| 阶段 | 内容 |
|---|---|
| Sep'26 v0.3 | 多视角第三人称重建流水线（commercial-friendly）；精简的端到端标准工作流（iPhone 录视频 → 微调策略）；4K+ Dex Task 库、多视角 HOI 数据集 |
| Oct'26 v0.4 | 20× 训练加速（天数→小时）；GT / Benchmark / Metrics / Evaluation scripts |
| Nov'26 Challenge | V2D Challenge：9/21 启动、11/10 冻结（或 11/04）、CoRL 2026 颁奖 |

![V2D 官方 Roadmap](V2D2.jpeg)

*图：V2D2.jpeg，官方 slide——Sep v0.3（重建流水线、iPhone→策略工作流、4K+ Dex Task 库）、Oct v0.4（20× 训练加速、GT/基准/指标/评估脚本）、Nov Challenge。*

这条 roadmap 说明：评测脚本与基准在 10 月才到位，开发窗口紧凑，越早锁定「评估协议」的队伍越有利。

---

## 五、评委视角：四维 rubric 与难度划分

### 5.1 四维 rubric（评委 Onboarding PPT 原文，1–5 分）

| 维度 | 1 分 | 3 分 | 5 分 |
|---|---|---|---|
| Economic Viability（经济可行性） | 每单位可用机器人数据的成本与今天相当 | 某一个成本项（算力 / 采集硬件 / 人力）有意义的下降 | 主导成本项数量级下降，且报告给出前后数值 |
| Data Flywheel（数据飞轮） | 输出是一次性产物，无部署回流到更好数据/模型的机制 | 存在反馈信号但闭环需要人工整理 | 部署自动生成或改进训练数据——机器人 rollout 数据回流进流水线 |
| Innovation（创新） | 以预期方式应用已知技术 | 新组合/新技术，明显弯曲成本或复杂度曲线，但增量 | 挑战某个被领域视为固定的假设并实证其不成立，重置可能性边界 |
| Complexity Reduction（复杂度削减） | 与今天最佳实践同阶段、同依赖、同采集约束，难迁移 | 删掉或合并一个阶段，或改用更便宜/约束更少的采集，他队一周可适配 | 组件大幅精简，无约束低成本采集，他队当天可落地 |

注意两处官方口径：
1. 评委用这份 PPT 的四维 rubric（Economic Viability / Data Flywheel / Innovation / Complexity Reduction）打分。
2. 官网公开页对 Industry Innovation Award 发布的准则措辞是 [deployability, data and compute efficiency, novelty, reproducibility]——两者语义可大致映射（deployability≈Complexity Reduction，data/compute efficiency≈Economic Viability，novelty≈Innovation），但措辞不同。评委评分以 Onboarding PPT 为准；若两份材料最终冲突，建议在 10/12 会上确认。

### 5.2 评分流程（评委侧）

独立打分 → panel 调解 → 平局先比 Economic Viability 再比 Data Flywheel → 分数与理由随 CoRL 公告公开。评委工作量约 4–6 小时，自我节奏。

### 5.3 难度划分（三档，初稿）

难度按「参赛方案要成功演示什么」递进：

| 档 | 内容 | 对应赛道/环节 | 难在哪里 | 指标 |
|---|---|---|---|---|
| 1 | Reconstruction 重建成功 | Track 1 重建；Track 3 显式重建部分 | 单目无标定无 GT；公制尺度是命门；物理合理性轴是动力学问题 | chamfer + 加速度/接触穿透 |
| 2 | 端到端 + 策略训练成功 | Track 2（retarget+策略）；Track 3 端到端策略 | 不是 RL 算法本身难（官方给 PPO 基线与固定仿真任务），而是 retarget 保真 + 对上游误差稳健；隐式路线受空间推理短板限制 | CHORD 四指标、eval_e2e 录像 |
| 3 | 数据飞轮（数据反哺） | 跨赛道；是评委 rubric 的 Data Flywheel 维度 | 要形成「部署→失败定位→自动补数据→再训练」的自动闭环，见 §5.5 | rubric 的 Data Flywheel 1–5 |

说明：前两档是「把赛道跑通」（对应排行榜），第三档不是某个赛道，而是四维 rubric 里的 Data Flywheel。把它列为最高难度，是因为它要求方案在任一赛道输出之上再叠加一个自动回流闭环——这是最难实证、也最符合 Industry Innovation Award 定位的部分。

三档与「哪条赛道容易拉开」要分开看：排行榜上最容易拉开的是 Track 1（重建方法差异大），Track 2 拼误差稳健性，Track 3 是端到端整体分。

### 5.4 评委判分注意点（每份提交问四句）

1. 报告是否给出 before/after 成本与吞吐量数值（Economic Viability 的 5 分门槛）？
2. 数据飞轮是否自动闭环，还是「有信号但需人工整理」（3 vs 5 的分界）？
3. 重建是否对真实 held-out / mocap GT 打分，而非自评质量？
4. Track 2 的证据是否用了官方噪声档与下游指标（CHORD），还是只在干净输入上自证？

### 5.5 数据飞轮：定义核实 + 三赛道怎么做 + 为什么难

核实：你的猜测与官方口径一致。评委 rubric 里 Data Flywheel 的定义是「在生产中使用后，系统能否无需额外人工持续变好」：1 分＝输出是一次性产物、无部署回流机制；3 分＝存在反馈信号但闭环需人工整理；5 分＝部署自动生成或改进训练数据，机器人 rollout 数据回流进流水线。你描述的「策略失败 → 判断需要什么数据 → 自动补全 → 再训练」正是 5 分的图景。

V2D 语境下的飞轮，本质是让「下游消费（retarget / 策略 rollout）」反过来定位「上游（重建/数据）」的失败，再自动生成针对性的数据。分赛道看：

Track 1（重建）：输出 4D HOI。闭环＝用下游消费信号定位重建失败 → 把失败 episode 自动送进精修或重采集 → 精修结果回流训练更快的重建模型。失败定位可自动化：chamfer（几何轴）与加速度/接触穿透（物理轴）正好把「几何错误」和「物理不合理」分开——scale 漂移会表现为加速度突变，接触穿透有直接的穿透量。自动补数据的难点在于：真正的精修（HOLD/BIGS 式逐段优化）算力高、重采集需要真人重做动作，都不免费；能自动化的是「把失败 episode 排队并路由到更贵的精修」和「用精修好的数据重训前馈模型让它变快」。

Track 2（落地）：飞轮最自然。策略 rollout 的失败信号最现成：AUC/SP-SR/MP-SR/MPPE 把失败定位到「追踪精度 vs 阶段成功率 vs 位姿误差」，RL 的 reward/价值本身也是信号。闭环＝失败 → 按失败所在的误差通道（jitter/dropout/接触）做针对性域随机化或误差注入 → 生成新合成数据 → 重训 → 再 rollout。这条最接近可实现，因为数据是合成的，不需要真人。

Track 3（第一人称）：最难。失败落在 ego 场景的自遮挡、手物接触、长时序。自动补数据有两个出口：一是让操作者重拍（不免费），二是用失败场景在仿真里渲染出合成 ego 视频（sim-to-ego 数据增强）回流。显式路线可做「失败 episode 的显式重建精修」，隐式路线要靠合成 ego 数据喂 VLA/WAM。

为什么难（你的判断是对的）：① 失败归因难——同一失败可能是重建、retarget、策略任一环节造成，自动定位需要逐层可解释或代理信号；② 自动补数据难——质量 vs 数量，合成数据可能只是放大噪声而非补盲区；③ 算力与成本——精修、重训、端到端微调都贵；④ 真正的 5 分要求「自动闭环、无需人工整理」，多数方案会停在 3 分（有信号但要人工整理）。

![Cosmos Curator 数据管线（NVIDIA 数据侧自动化的行业参照）](V2D3.jpeg)

![Cosmos Curator 三 MVP 流水线](V2D4.jpeg)

*图：V2D3/V2D4.jpeg，NVIDIA Cosmos Curator——视频切分、过滤、VLM 标注、SAM3 分割、风格迁移、去重，是「自动补数据/整理」数据侧的现成例子，可作飞轮叙事的外部参照。*

---

## 六、开放问题（待与用户讨论后定稿）

1. 冻结日期：11/04（评委 PPT）vs 11/10（官网/slide）——是否在 10/12 会上确认。
2. 评委 rubric 两套措辞（PPT 四维 vs 官网 deployability/novelty/reproducibility）以哪套为准。
3. 难度矩阵是否采纳，以及是否补「三条赛道得分区分布」的分析。
4. 是否需要加入「数据飞轮」的深化章节（部署回流机制、失败→补数据→再训练的闭环设计）。
5. 哪些本地实验图/官方图要正式放进报告（候选见 §七）。

---

## 七、本地素材清单（可直接嵌入本报告的图片）

- 官方赛道与奖项总览：`V2D1.jpg`（三赛道、奖项、时间线）
- 官方 Roadmap：`V2D2.jpeg`（Sep v0.3 / Oct v0.4 / Nov Challenge）
- 数据侧示例：`V2D3.jpeg`（Cosmos Curator 数据管线）、`V2D4.jpeg`（三 MVP 流水线）
- 本机审计图（outputs/figs/）：`track1_overview.png`、`track2_overview.png`、`track2_noise_comparison.png`、`track3_overview.png`、`track3_acceleration.png`、`track3_visibility.png`、`method_comparison.png`、`m1_feature_tracks.png`、`m2_mask_coverage.png`、`m4_track3_errors.png`

---

## 八、参考来源

- V2D Challenge 官网：https://nvidia-isaac.github.io/video_to_data/v2d_challenge/
- 数据集：https://huggingface.co/datasets/nvidia/video_to_data_challenge （本地已完整下载 track_1/2/3）
- Starter toolkit：https://github.com/nvidia-isaac/video_to_data （本地已 clone）
- CHORD 项目页（Track 2 指标出处）：https://nvidia-isaac.github.io/video_to_data/chord/ 与 tech report chord.pdf
- 评委 Onboarding PDF（本地）：V2D Challenge_ Judges Intro.pdf / V2DChallenge_JudgesIn (1).pdf
- 本机前期实验：outputs/report/executive_summary.md、outputs/figs/、outputs/metrics/、outputs/logs/
