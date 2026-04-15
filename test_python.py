import sys
import os

# 1. 打印初始的 Python 路径
print("=== 初始 Python 路径 ===")
for idx, path in enumerate(sys.path):
    print(f"{idx}: {path}")

# 2. 获取当前目录下的 src 绝对路径（适配你的 ubuntu 环境）
# 方式1：使用当前工作目录（推荐，适配你 ls 显示的 /workspace 目录）
src_path = os.path.abspath("./src")
# 方式2：如果需要固定路径，也可以直接写死（根据你的路径）
# src_path = "/workspace/src"

# 3. 将 src 目录添加到 Python 路径（添加到最前面，优先搜索）
if src_path not in sys.path:
    sys.path.insert(0, src_path)
    print(f"\n✅ 已将 {src_path} 加入 Python 路径")
else:
    print(f"\nℹ️ {src_path} 已存在于 Python 路径中")

# 4. 打印添加后的 Python 路径
print("\n=== 添加 src 后的 Python 路径 ===")
for idx, path in enumerate(sys.path):
    print(f"{idx}: {path}")
