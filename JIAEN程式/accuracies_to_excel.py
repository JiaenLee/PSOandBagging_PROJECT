import pandas as pd
import json
import os

# 1. 設定路徑
folder_name = "模型分析"
output_filename = "PSO_模型正確率分析.xlsx"
output_path = os.path.join(folder_name, output_filename)
json_file_path = '../training_accuracies/Selected_PSO_TRENB.json'

# 資料來源資料夾 (用來判斷分類)
binary_dir = "../datasets/離散化資料集/二類別"
multi_dir = "../datasets/離散化資料集/多類別"

# 2. 檢查並建立「模型分析」資料夾
if not os.path.exists(folder_name):
    os.makedirs(folder_name)

# 3. 自動掃描資料夾，建立「資料集 -> 分類」的對照表
category_map = {}

def scan_datasets(path, label):
    if os.path.exists(path):
        for f in os.listdir(path):
            if f.endswith(".csv"):
                dataset_name = f.replace(".csv", "")
                category_map[dataset_name] = label

scan_datasets(binary_dir, "二類別")
scan_datasets(multi_dir, "多類別")

# 4. 讀取數據
try:
    with open(json_file_path, 'r', encoding='utf-8') as f:
        json_data = json.load(f)
except FileNotFoundError:
    print(f"錯誤：找不到 {json_file_path}")
    json_data = {}

# 5. 整理數據格式
rows = []
for dataset_name, accuracies in json_data.items():
    # 從對照表找出該資料集屬於哪一類，若找不到則標示為"未知"
    category_type = category_map.get(dataset_name, "未知")
    
    for acc in accuracies:
        rows.append({
            "類別類型": category_type,  # 新增的第一欄
            "資料集名稱": dataset_name, # 原本的第一欄
            "正確率": acc              # 原本的第二欄
        })

# 6. 建立 DataFrame
df = pd.DataFrame(rows)

# 7. 輸出成 Excel 檔
if not df.empty:
    df.to_excel(output_path, index=False)
    print(f"Excel 檔案已成功生成於：{output_path}")
    print(f"目前分類統計：\n{df['類別類型'].value_counts()}")
else:
    print("沒有可用的數據可以生成報告。")