import pandas as pd
import os

# 设置你的文件夹路径
folder_path = '/Users/shijunshen/Documents/Code/PycharmProjects/ReinforcementLearning/DDPG_SmartFarm/data'

# 获取该文件夹下所有CSV文件
csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

# 读取所有 CSV 文件，并将它们拼接起来
df_list = []

for file in csv_files:
    file_path = os.path.join(folder_path, file)
    df = pd.read_csv(file_path)
    df_list.append(df)

# 拼接所有 DataFrame
combined_df = pd.concat(df_list, ignore_index=True, sort=False)
print("合并后的数据:")
print(combined_df)

# 找出每一列的均值和标准差
mean_values = combined_df.mean()
std_values = combined_df.std()

# 输出均值和标准差
print("\n均值:")
print(mean_values)
print("\n标准差:")
print(std_values)

# 排除的列
exclude_columns = ["sensor_sensorid2type", "Height", "Coverage"]

# 标准化并保存新的文件
for file in csv_files:
    file_path = os.path.join(folder_path, file)
    df = pd.read_csv(file_path)

    # 标准化：对每一列应用标准化
    for col in df.columns:
        if col not in exclude_columns:
            # 按照均值和标准差标准化
            df[col] = (df[col] - mean_values[col]) / std_values[col]

    # 保存新的文件
    new_file_path = os.path.join(folder_path, file.replace(".csv", "_normalization.csv"))
    df.to_csv(new_file_path, index=False)

    print(f"已处理并保存标准化文件：{new_file_path}")