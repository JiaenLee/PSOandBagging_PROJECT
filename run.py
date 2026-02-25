# import subprocess
# import os
# import optimize
# import config
# import sys

# # 模型訓練程式清單
# scripts = ["PSO_Bagging.py"]

# # 逐一執行
# for script in scripts:
#     print(f"正在透過 uv 執行: {script}")
#     subprocess.run([sys.executable, script], check=True)

import subprocess
import os
import sys
import numpy as np
import pandas as pd

# ---------------------------
# 先自動生成測試 CSV

# data_folder = "datasets/離散化資料集/多類別"
# os.makedirs(data_folder, exist_ok=True)
# csv_file = os.path.join(data_folder, "Abalone_3class.csv")
# if not os.path.exists(csv_file): # 如果檔案不存在就生成
#    num_features = 8
#    num_classes = 3
#    samples_per_class = 10

#    data_list = []
#    for class_label in range(num_classes):
#        for _ in range(samples_per_class):
#            features = np.random.randint(0, 5, size=num_features)  # 隨機整數 0~4
#            data_list.append(list(features) + [class_label])

#    columns = [f"feat_{i+1}" for i in range(num_features)] + ["class"]
#    df = pd.DataFrame(data_list, columns=columns)
#    df.to_csv(csv_file, index=False)
#    print(f"測試用 CSV 已生成: {csv_file}")
#else:
#    print(f"CSV 已存在: {csv_file}")
# ---------------------------


# -----------------------------
# 建立必要資料夾，只需一次
os.makedirs("temp_models/PSO_TRENB", exist_ok=True)
os.makedirs("temp_models/Bagging", exist_ok=True)
os.makedirs("temp_models/PSO_Bagging", exist_ok=True)

os.makedirs("accuracy_result", exist_ok=True)
os.makedirs("training_accuracies", exist_ok=True)
os.makedirs("training_pred_vec", exist_ok=True)
os.makedirs("data_filter_reserved", exist_ok=True)
os.makedirs("data_filter_res", exist_ok=True)
os.makedirs("selection_result", exist_ok=True)
# -----------------------------


# ---------------------------
# 執行模型訓練程式
scripts = [
    "PSO_TRENB.py",
    "bagging.py",
    "PSO_Bagging.py"
]

for script in scripts:
    print(f"\n==============================")
    print(f"正在執行: {script}")
    print(f"==============================\n")

    subprocess.run([sys.executable, script], check=True)
# ---------------------------