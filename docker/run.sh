#!/bin/bash
# Isaac Sim 容器启动脚本

set -e  # 遇到错误立即退出

# ====================== 前置检查 ======================
# 检查 docker 命令是否可用
if ! command -v docker &> /dev/null; then
    echo "错误：未找到 docker 命令，请先安装 Docker！"
    exit 1
fi

# 检查是否有 root 权限（docker 特权模式需要 root）
if [ "$(id -u)" -ne 0 ]; then
    echo "提示：需要 root 权限运行脚本，自动添加 sudo..."
    exec sudo "$0" "$@"  # 自动重新以 sudo 执行脚本
fi

# 检查物理显示是否可用
if [ -z "$DISPLAY" ] || ! xset q > /dev/null 2>&1; then
    echo "错误：未检测到可用的物理显示！请确保："
    echo "  1. 显示器已正确连接并开机"
    echo "  2. DISPLAY 环境变量已设置（正常应为 :0 或 :1）"
    echo "  3. 当前用户有 X11 访问权限"
    exit 1
fi

# ====================== 核心配置 ======================
# 镜像名称（可根据需要修改）
IMAGE_NAME="isaacsim5.1_lerobot:v1.0"


# 容器名称（可根据需要修改）
CONTAINER_NAME="isaac_sim_lerobot" 

# 宿主机工作目录（可根据需要修改）
HOST_WORKSPACE="/data/SJJ/UBT/lerobot"

# 容器内工作目录
CONTAINER_WORKSPACE="/workspace"

# ====================== 执行启动命令 ======================
# 1. 授权本地 docker 访问物理 X11 显示（关键：解决权限问题）
echo "步骤1：授权 Docker 访问物理显示（DISPLAY=$DISPLAY）..."
xhost +local:docker > /dev/null 2>&1
# 额外授权当前用户访问（防止权限不足）
xhost +SI:localuser:root > /dev/null 2>&1

# 2. 启动容器（优化物理显示相关配置）
echo "步骤2：启动 Isaac Sim 容器（root 权限 + GPU + 物理显示）..."
docker run -it \
    --name $CONTAINER_NAME \
    --privileged \
    --network host \
    --user root \
    --gpus all \
    --shm-size=8g \
    --group-add $(getent group video | cut -d: -f3) \
    --group-add $(getent group plugdev | cut -d: -f3) \
    -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
    -v /var/run/dbus:/var/run/dbus:rw \
    -v ${HOST_WORKSPACE}:${CONTAINER_WORKSPACE} \
    -v /dev/bus/usb:/dev/bus/usb \
    -v /dev/video0:/dev/video0 \
    -v /dev/video1:/dev/video1 \
    -v /dev:/dev \
    -v /run/udev:/run/udev \
    -e DISPLAY=$DISPLAY \
    -e QT_X11_NO_MITSHM=1 \
    -e XAUTHORITY=/tmp/.docker.xauth \
    -e NO_AT_BRIDGE=1 \
    -e "ACCEPT_EULA=Y" \
    -e "PRIVACY_CONSENT=Y" \
    -e XDG_RUNTIME_DIR=/tmp \
    -w ${CONTAINER_WORKSPACE} \
    ${IMAGE_NAME}

# ====================== 后置处理 ======================
echo "步骤3：容器退出，恢复 X11 访问权限..."
# 回收显示授权（增强安全性）
xhost -local:docker > /dev/null 2>&1
xhost -SI:localuser:root > /dev/null 2>&1
echo "容器已退出，脚本执行完成！"
