import pandas as pd
import os
import shutil

# 路径设置
original_csv = './dataset/val_data4.csv'
new_csv = './dataset/val_data.csv'
target_dir = './dataset/output_tif_patches'
os.makedirs(target_dir, exist_ok=True)

# 加载前10行
df = pd.read_csv(original_csv)
df_subset = df.head(1).copy()

# 要处理的列
columns_to_update = ['S1', 'S2', 'Height', 'BuildFoot', 'LLM']

# 替换路径并移动文件
for col in columns_to_update:
    new_paths = []
    for old_path in df_subset[col]:
        filename = os.path.basename(old_path)
        new_path = os.path.join(target_dir, filename)
        shutil.copy(old_path, new_path)  # 如果你想移动文件用 shutil.move
        new_paths.append(new_path)
    df_subset[col] = new_paths

# 保存新的CSV
df_subset.to_csv(new_csv, index=False)
print(f"新CSV文件已保存为：{new_csv}")
