# # import pandas as pd
# # import numpy as np

# # # 使用 Docker 内部的正确挂载路径
# # file_path = '/workspace/datasets/packing_box_v5/data/chunk-000/episode_000000.parquet'

# # # 读取数据
# # df = pd.read_parquet(file_path)

# # print('='*1000)
# # print(f'✅ 成功读取数据集！总帧数: {len(df)}')
# # print('='*1000)

# # # 打印前 50 帧的 action
# # print('【前 50 帧的 action 数据】:')
# # for i, action in enumerate(df['action'].head(5000)):
# #     action_list = np.round(action, 4).tolist()
# #     print(f'第 {i} 帧: {action_list}')

# # print('='*1000)

# import pandas as pd
# import numpy as np
# import os

# # ================= 配置区域 =================
# # 1. 指定你的 parquet 文件路径 (注意：这里写的是 Docker 内部的路径)
# # 1. 指定你的 parquet 文件路径 (注意：这里写的是 Docker 内部的真实路径)
# PARQUET_FILE_PATH = '/data/btt/ybx_expri/ubt_3_22/datasets/test/pick_placev1_btt/data/chunk-000/episode_000000.parquet'

# # 2. 指定输出路径 (保存在 /workspace 根目录下，方便你在宿主机直接查看)
# EXCEL_OUTPUT_PATH = '/data/btt/ybx_expri/ubt_3_22/lerobot/scripts/episode_000000_full_data.xlsx'
# CSV_OUTPUT_PATH = '/data/btt/ybx_expri/ubt_3_22/lerobot/scripts/episode_000000_full_data.csv'
# # ============================================

# def main():
#     print('=' * 50)
#     # 检查文件是否存在
#     if not os.path.exists(PARQUET_FILE_PATH):
#         print(f"❌ 找不到文件，请检查路径是否正确: {PARQUET_FILE_PATH}")
#         return

#     print('正在读取 parquet 数据，请稍候...')
#     df = pd.read_parquet(PARQUET_FILE_PATH)

#     print(f'✅ 成功读取数据集！总帧数: {len(df)}，包含以下列: {list(df.columns)}')

#     # 数据预处理：将内部的多维数组（如 action, state）统一转为保留4位小数的字符串
#     # 这样在 Excel 里就会显示为 [0.1234, -0.5678, ...] 的直观格式，且不会报错
#     print('正在格式化数组数据...')
#     for col in df.columns:
#         df[col] = df[col].apply(
#             lambda x: str(np.round(x, 4).tolist()) if isinstance(x, (np.ndarray, list)) else x
#         )

#     # 尝试保存为 Excel
#     try:
#         print('正在导出 Excel 文件...')
#         df.to_excel(EXCEL_OUTPUT_PATH, index=False)
#         print(f'✅ 成功保存为 Excel 文件: {EXCEL_OUTPUT_PATH}')
#     except ModuleNotFoundError:
#         print('⚠️ 环境中未安装 openpyxl，跳过生成 Excel 文件。')
#         print('   (提示: 你可以在 Docker 中运行 pip install openpyxl 来支持导出 Excel)')

#     # 始终保存一份 CSV 作为兜底（CSV 极快且不需要第三方库，直接用 Excel 软件打开即可）
#     print('正在导出 CSV 文件...')
#     df.to_csv(CSV_OUTPUT_PATH, index=False, encoding='utf-8-sig')
#     print(f'✅ 成功保存为 CSV 文件: {CSV_OUTPUT_PATH}')
#     print('=' * 50)

# if __name__ == "__main__":
#     main()


import pandas as pd
import numpy as np
import os

# ================= 配置区域 =================
# 1. 指定你的 parquet 文件路径 (注意：这里写的是 Docker 内部的真实路径)
PARQUET_FILE_PATH = '/data/btt/ybx_expri/lerobot_0.5.1/datasets/task3/v3/data/chunk-000/file-000.parquet'

# 2. 指定输出路径 (保存在 /workspace 根目录下，方便你在宿主机直接查看)
EXCEL_OUTPUT_PATH = '/data/btt/ybx_expri/lerobot_0.5.1/src/lerobot/scripts/episode_000000_full_data.xlsx'
CSV_OUTPUT_PATH = '/data/btt/ybx_expri/lerobot_0.5.1/src/lerobot/scripts/episode_000000_full_data.csv'
# ============================================

def main():
    print('=' * 50)
    # 检查文件是否存在
    if not os.path.exists(PARQUET_FILE_PATH):
        print(f"❌ 找不到文件，请检查路径是否正确: {PARQUET_FILE_PATH}")
        return

    print('正在读取 parquet 数据，请稍候...')
    df = pd.read_parquet(PARQUET_FILE_PATH)

    print(f'✅ 成功读取数据集！总帧数: {len(df)}，包含以下列: {list(df.columns)}')

    # ================= 新增功能：统计夹爪指令 =================
    if 'action' in df.columns:
        print('\n正在统计 action 夹爪控制指令（最后两维）...')
        
        # 提取倒数第二维（左夹爪）和倒数第一维（右夹爪），为了防止浮点精度误差，先 round 保留1位小数
        left_gripper = df['action'].apply(lambda x: round(float(x[-2]), 1))
        right_gripper = df['action'].apply(lambda x: round(float(x[-1]), 1))
        
        # 统计频次
        left_counts = left_gripper.value_counts().sort_index()
        right_counts = right_gripper.value_counts().sort_index()
        
        print(f"【左夹爪 (倒数第2维) 状态分布】:")
        for val, count in left_counts.items():
            print(f"  状态 {val:>4}: 出现了 {count} 次")
            
        print(f"【右夹爪 (倒数第1维) 状态分布】:")
        for val, count in right_counts.items():
            print(f"  状态 {val:>4}: 出现了 {count} 次")
        print('-' * 50)
    # ==========================================================

    # 数据预处理：将内部的多维数组（如 action, state）统一转为保留4位小数的字符串
    # 这样在 Excel 里就会显示为 [0.1234, -0.5678, ...] 的直观格式，且不会报错
    print('\n正在格式化数组数据...')
    for col in df.columns:
        # 注意这里加了一个判断，防止有些列不是可迭代的数组
        df[col] = df[col].apply(
            lambda x: str(np.round(x, 4).tolist()) if isinstance(x, (np.ndarray, list)) else x
        )

    # 尝试保存为 Excel
    try:
        print('正在导出 Excel 文件...')
        df.to_excel(EXCEL_OUTPUT_PATH, index=False)
        print(f'✅ 成功保存为 Excel 文件: {EXCEL_OUTPUT_PATH}')
    except ModuleNotFoundError:
        print('⚠️ 环境中未安装 openpyxl，跳过生成 Excel 文件。')
        print('   (提示: 你可以在 Docker 中运行 pip install openpyxl 来支持导出 Excel)')

    # 始终保存一份 CSV 作为兜底（CSV 极快且不需要第三方库，直接用 Excel 软件打开即可）
    print('正在导出 CSV 文件...')
    df.to_csv(CSV_OUTPUT_PATH, index=False, encoding='utf-8-sig')
    print(f'✅ 成功保存为 CSV 文件: {CSV_OUTPUT_PATH}')
    print('=' * 50)

if __name__ == "__main__":
    main()