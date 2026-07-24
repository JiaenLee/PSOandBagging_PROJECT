import pandas as pd
import json
import os

# 1. 設定路徑
folder_name = "模型分析"
output_filename = "PSO_training_accuracies分析.xlsx"
output_path = os.path.join(folder_name, output_filename)
json_file_path = '../training_accuracies/PSO_TRENB.json'

# 2. 檢查並建立「模型分析」資料夾
if not os.path.exists(folder_name):
    os.makedirs(folder_name)

# 3. 讀取數據
try:
    with open(json_file_path, 'r', encoding='utf-8') as f:
        json_data = json.load(f)
except FileNotFoundError:
    print(f"錯誤：找不到 {json_file_path}")
    json_data = {}

# 4. 整理數據格式
rows = []
for full_name, accuracies in json_data.items():
    # 直接從名稱抓取：如果是 "二類別_Algerian"，分割成 ["二類別", "Algerian"]
    if "_" in full_name:
        category_type, dataset_name = full_name.split("_", 1)
    else:
        category_type, dataset_name = "未知", full_name
    
    for acc in accuracies:
        rows.append({
            "類別類型": category_type,
            "資料集名稱": dataset_name,
            "正確率": acc
        })

# 5. 建立 DataFrame 並輸出
df = pd.DataFrame(rows)
if not df.empty:
    df.to_excel(output_path, index=False)
    print(f"Excel 檔案已成功生成於：{output_path}")
else:
    print("沒有可用的數據。")