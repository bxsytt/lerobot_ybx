#!/usr/bin/env bash
# =============================================================================
#  GlobalHumanoidRobotChallenge2026 — 自适应容器启动脚本
#
#  所有关键路径和名称均可通过环境变量覆盖，默认值基于当前机器自动检测。
#
#  用法:
#    ./run.sh                    # 交互式启动（自动检测一切）
#    ./run.sh --headless         # 无头模式（无需物理显示）
#    ./run.sh <任意docker参数>   # 额外参数透传给 docker run
#
#  可覆盖的环境变量（示例）:
#    IMAGE_NAME=my_image:v2  ./run.sh
#    HOST_WORKSPACE=/my/path ./run.sh
#    HEADLESS=1               ./run.sh
# =============================================================================
set -euo pipefail

# ========================== 颜色输出工具 ==========================
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; NC='\033[0m'
info()  { echo -e "${GREEN}[INFO]${NC}  $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*" >&2; }

# ========================== 参数解析 ==========================
EXTRA_DOCKER_ARGS=()
for arg in "$@"; do
    case "$arg" in
        --headless) HEADLESS=1 ;;
        *)          EXTRA_DOCKER_ARGS+=("$arg") ;;
    esac
done
HEADLESS="${HEADLESS:-0}"

# ========================== 前置检查 ==========================
# Docker
if ! command -v docker &>/dev/null; then
    error "未找到 docker 命令，请先安装 Docker！"
    exit 1
fi

# 权限：优先使用当前用户的 docker 组，不行再自动 sudo
if [ "$(id -u)" -ne 0 ]; then
    if ! docker info &>/dev/null 2>&1; then
        warn "当前用户无 docker 权限，自动以 sudo 重新执行..."
        exec sudo --preserve-env=DISPLAY,HEADLESS,IMAGE_NAME,CONTAINER_NAME,HOST_WORKSPACE,CONTAINER_WORKSPACE \
             "$0" "$@"
    fi
fi

# NVIDIA GPU
if ! command -v nvidia-smi &>/dev/null; then
    error "未找到 nvidia-smi！Isaac Sim 需要 NVIDIA GPU 驱动。"
    exit 1
fi
if ! nvidia-smi &>/dev/null; then
    error "nvidia-smi 执行失败！请检查 GPU 驱动是否正常。"
    exit 1
fi

# 显示检查（非 headless 模式时必须有可用显示，不会自动降级）
if [ "$HEADLESS" -eq 0 ]; then
    if [ -z "${DISPLAY:-}" ]; then
        error "DISPLAY 未设置！请确保显示器已连接，或使用 --headless 模式。"
        exit 1
    elif ! xset q &>/dev/null 2>&1; then
        error "无法连接 X11 显示 (DISPLAY=$DISPLAY)！请检查显示器和权限，或使用 --headless 模式。"
        exit 1
    fi
fi

# ========================== 核心配置（全部可通过环境变量覆盖）==========================
# 自动检测项目根目录：run.sh 就在项目根目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SCRIPT_DIR}"

IMAGE_NAME="${IMAGE_NAME:-isaac-sim-nijie-lerobot:v5}"
CONTAINER_NAME="${CONTAINER_NAME:-isaac_sim_ubt}"
HOST_WORKSPACE="${HOST_WORKSPACE:-${PROJECT_ROOT}}"
CONTAINER_WORKSPACE="${CONTAINER_WORKSPACE:-/workspace/lerobot_0.5.1}"
SHM_SIZE="${SHM_SIZE:-8g}"

# Isaac Sim 缓存目录（持久化挂载，避免每次启动重新编译 shader / 下载资产）
ISAAC_CACHE_ROOT="${ISAAC_CACHE_ROOT:-${HOME}/.cache/isaac_sim_container}"

# HuggingFace 缓存
HF_CACHE="${HF_CACHE:-${HOME}/.cache/huggingface}"

info "项目根目录  : ${HOST_WORKSPACE}"
info "容器工作目录: ${CONTAINER_WORKSPACE}"
info "镜像名称    : ${IMAGE_NAME}"
info "容器名称    : ${CONTAINER_NAME}"
info "Headless    : $([ "$HEADLESS" -eq 1 ] && echo '是' || echo '否')"

# ========================== 创建持久化目录 ==========================
for d in cache/kit cache/ov cache/pip cache/glcache cache/computecache data documents; do
    mkdir -p "${ISAAC_CACHE_ROOT}/${d}"
done
mkdir -p "${HF_CACHE}"

# ========================== 容器生命周期管理 ==========================
# 每次启动前清除旧容器，确保使用最新代码挂载和干净状态
if docker container inspect "$CONTAINER_NAME" &>/dev/null 2>&1; then
    info "清除旧容器 ${CONTAINER_NAME} ..."
    docker rm -f "$CONTAINER_NAME" &>/dev/null || true
fi

# ========================== 构建 docker run 参数 ==========================
DOCKER_ARGS=(
    --entrypoint /bin/bash
    --name "$CONTAINER_NAME"
    --privileged
    --network host
    --user root
    --gpus all
    --shm-size="$SHM_SIZE"
    --restart unless-stopped
)

# --- 用户组 ---
# video 组必须存在（GPU 访问需要）
VIDEO_GID=$(getent group video 2>/dev/null | cut -d: -f3 || true)
if [ -z "$VIDEO_GID" ]; then
    error "未找到 video 组！GPU 访问需要该组。"
    exit 1
fi
DOCKER_ARGS+=(--group-add "$VIDEO_GID")
# 可选组：不存在则跳过
for grp in plugdev input; do
    GID=$(getent group "$grp" 2>/dev/null | cut -d: -f3 || true)
    if [ -n "$GID" ]; then
        DOCKER_ARGS+=(--group-add "$GID")
    fi
done

# --- 显示相关 ---
if [ "$HEADLESS" -eq 0 ]; then
    info "配置 X11 显示转发..."
    xhost +local:docker &>/dev/null 2>&1 || true
    xhost +SI:localuser:root &>/dev/null 2>&1 || true
    DOCKER_ARGS+=(
        -e DISPLAY="$DISPLAY"
        -e QT_X11_NO_MITSHM=1
        -e XAUTHORITY=/tmp/.docker.xauth
        -v /tmp/.X11-unix:/tmp/.X11-unix:rw
    )
else
    info "Headless 模式，跳过 X11 配置"
fi

# --- 挂载：项目代码 ---
DOCKER_ARGS+=(
    -v "${HOST_WORKSPACE}:${CONTAINER_WORKSPACE}:rw"
    -w "${CONTAINER_WORKSPACE}"
)

# --- 挂载：Isaac Sim 缓存（大幅加速二次启动）---
DOCKER_ARGS+=(
    -v "${ISAAC_CACHE_ROOT}/cache/kit:/root/.cache/kit:rw"
    -v "${ISAAC_CACHE_ROOT}/cache/ov:/root/.cache/ov:rw"
    -v "${ISAAC_CACHE_ROOT}/cache/pip:/root/.cache/pip:rw"
    -v "${ISAAC_CACHE_ROOT}/cache/glcache:/root/.cache/nvidia/GLCache:rw"
    -v "${ISAAC_CACHE_ROOT}/cache/computecache:/root/.cache/nvidia/ComputeCache:rw"
    -v "${ISAAC_CACHE_ROOT}/data:/root/.local/share/ov/data:rw"
    -v "${ISAAC_CACHE_ROOT}/documents:/root/Documents:rw"
)

# --- 挂载：HuggingFace 模型缓存 ---
DOCKER_ARGS+=(
    -v "${HF_CACHE}:/root/.cache/huggingface:rw"
)

# --- 挂载：设备（仅当路径存在时挂载，避免报错）---
for dev_path in /dev/bus/usb /run/udev /var/run/dbus; do
    if [ -e "$dev_path" ]; then
        DOCKER_ARGS+=(-v "${dev_path}:${dev_path}:rw")
    fi
done

# 摄像头设备：自动检测 /dev/video*，而非硬编码
for cam in /dev/video*; do
    if [ -e "$cam" ]; then
        DOCKER_ARGS+=(-v "${cam}:${cam}")
    fi
done

# --- 环境变量 ---
DOCKER_ARGS+=(
    -e NO_AT_BRIDGE=1
    -e "ACCEPT_EULA=Y"
    -e "PRIVACY_CONSENT=Y"
    -e XDG_RUNTIME_DIR=/tmp
    -e PYTHONPATH="${CONTAINER_WORKSPACE}"
)

# ========================== 启动容器 ==========================
info "启动容器 ${CONTAINER_NAME} ..."

# 容器启动后自动执行：editable install 使宿主机源码修改立即生效
INIT_CMD='echo "[INIT] 安装本地源码 (editable mode)..." && pip install -e . --no-deps -q 2>/dev/null; exec /bin/bash'

echo -e "${CYAN}docker run -it ${DOCKER_ARGS[*]} ${EXTRA_DOCKER_ARGS[*]:-} ${IMAGE_NAME} -c \"${INIT_CMD}\"${NC}"
echo ""

docker run -it \
    "${DOCKER_ARGS[@]}" \
    "${EXTRA_DOCKER_ARGS[@]+"${EXTRA_DOCKER_ARGS[@]}"}" \
    "${IMAGE_NAME}" \
    -c "${INIT_CMD}"

# ========================== 退出后清理 ==========================
if [ "$HEADLESS" -eq 0 ]; then
    xhost -local:docker &>/dev/null 2>&1 || true
    xhost -SI:localuser:root &>/dev/null 2>&1 || true
fi
info "容器已退出，脚本执行完成。"
