# Ubtech_sim 机器人仿真环境

## 概述

基于 Isaac Sim 5.1，用于模拟 Global Humanoid Robot Challenge 2026 赛事中的 WalkerS2 机器人控制及相关场景搭建。

## 功能特性

- 场景构建与加载
- WalkerS2 机器人加载与双臂 IK 控制
- 抓取目标自动规划（选臂、姿态计算、实时跟踪）
- 姿态数据记录（CSV 格式）— 作为评分输入
- 摄像头图像采集（HDF5 格式）— 临时作为图像信息检查的手段

## 项目结构

```
Ubtech_sim/
├── source/                  # Python 源码包
│   ├── __init__.py          # 包初始化
│   ├── config_loader.py     # 配置加载与路径解析
│   ├── coordinate_utils.py  # 世界坐标 ↔ Pinocchio 基座坐标变换
│   ├── grasp_planner.py     # 抓取目标规划与实时跟踪
│   ├── DataLogger.py        # 位姿 CSV / 相机 HDF5 数据记录
│   ├── DualArmIK.py         # 基于 Pinocchio 的双臂逆运动学求解器
│   ├── RobotArticulation.py # WalkerS2 机器人 Articulation 控制接口
│   └── SceneBuilder.py      # 场景构建（桌子、箱子、零件、机器人）
├── config/                  # 任务配置 (task1.yaml – task4.yaml)
├── main.py                  # 仿真入口：编排上述模块的流水线
└── README.md                # 本文档
```

## 核心模块

### config_loader
配置加载工具，解析 YAML 文件并将 `root_path` 解析为绝对路径；`apply_scatter_config` 用散布区域覆盖 plane 配置。

**配置文件路径说明**：
- 任务配置文件：`Ubtech_sim/config/task1.yaml` ~ `task4.yaml`
- 资产文件路径：`../assets/resources/`（在配置文件中设置 `root_path`）

### CoordinateTransform
管理世界坐标系与 Pinocchio URDF 根坐标系之间的刚体变换，通过 torso_link 锚点自动计算。

### GraspPlanner
抓取目标规划器：根据零件位姿自动选臂、计算抓取姿态和 TCP 偏移，并在控制回调中实时跟踪目标物体。

### SceneBuilder
负责构建和初始化仿真场景，使用 Replicator 功能，加载 USD 文件和设置场景元素。

**资产文件位置**：
- 机器人模型：`../assets/resources/Collected_s2_v1_ecbg/s2_v1.usd`
- URDF 文件：`../assets/resources/s2.urdf`
- 其他 USD 资产：`../assets/resources/` 目录

### RobotArticulation
实现 WalkerS2 的感知和控制接口搭建，包括关节状态获取、双臂 IK 控制和传感器（六维力、相机）数据获取。

### DualArmIK
基于 Pinocchio 的双臂逆运动学求解器，使用加权阻尼最小二乘法（DLS），支持位置 / 姿态权重分离。

### DataLogger
提供目标物体位姿数据（打分用）和摄像头图像的数据记录功能，支持 CSV 和 HDF5 格式输出。
各目标物体的位姿数据通过 PhysX view 接口获取，放在物理回调中。

## 配置文件

四个任务分别用 `config/task1.yaml` – `config/task4.yaml` 进行参数配置。

**关键配置项**：
- `root_path`: 资产文件根路径，应设置为 `../assets/resources/`
- `scene_usd`: 场景 USD 文件路径（相对于 `root_path`）
- `task_number`: 任务编号（1-4）

## 使用方式

### 1. 配置路径

确保 `config/*.yaml` 中的 `root_path` 指向正确的资产目录：
```yaml
root_path: "../assets/resources/"
```

### 2. 运行仿真

```bash
# 在项目根目录
cd GlobalHumanoidRobotChallenge2026_Baseline

# 运行仿真
python Ubtech_sim/main.py
```

## 依赖关系

- **Isaac Sim 5.1+**: 仿真环境
- **Pinocchio**: 运动学计算
- **PyYAML**: 配置文件解析

