import pandas as pd
import json
import os

# 1. 設定路徑
# 此 Python 檔所在資料夾：PSOandBagging_PROJECT/YiCi程式
script_dir = os.path.dirname(os.path.abspath(__file__))

# 專案根目錄：PSOandBagging_PROJECT
project_dir = os.path.dirname(script_dir)

# Excel 輸出位置：YiCi程式/模型分析
folder_name = os.path.join(script_dir, "模型分析")
output_filename = "Bagging_模型正確率分析.xlsx"
output_path = os.path.join(folder_name, output_filename)

# JSON 檔案位置
json_file_path = os.path.join(
    project_dir,
    "training_accuracies",
    "Bagging.json"
)

# 資料來源資料夾
binary_dir = os.path.join(
    project_dir,
    "datasets",
    "離散化資料集",
    "二類別"
)

multi_dir = os.path.join(
    project_dir,
    "datasets",
    "離散化資料集",
    "多類別"
)

# 2. 建立「YiCi程式/模型分析」資料夾
os.makedirs(folder_name, exist_ok=True)

# 3. 建立「資料集名稱 -> 類別」對照表
category_map = {}

def scan_datasets(path, label):
    if os.path.exists(path):
        for filename in os.listdir(path):
            if filename.lower().endswith(".csv"):
                dataset_name = os.path.splitext(filename)[0]
                category_map[dataset_name] = label
    else:
        print(f"找不到資料夾：{path}")

scan_datasets(binary_dir, "二類別")
scan_datasets(multi_dir, "多類別")

# 4. 讀取 JSON
try:
    with open(json_file_path, "r", encoding="utf-8") as file:
        json_data = json.load(file)
except FileNotFoundError:
    print(f"錯誤：找不到 JSON 檔案：{json_file_path}")
    json_data = {}

# 5. 整理資料
rows = []

for dataset_name, accuracies in json_data.items():
    category_type = category_map.get(dataset_name, "未知")

    for accuracy in accuracies:
        rows.append({
            "類別類型": category_type,
            "資料集名稱": dataset_name,
            "正確率": accuracy
        })

# 6. 建立 DataFrame
df = pd.DataFrame(rows)

# 7. 輸出 Excel
if not df.empty:
    df.to_excel(output_path, index=False)

    print(f"Excel 檔案已成功生成於：{output_path}")
    print(f"目前分類統計：\n{df['類別類型'].value_counts()}")
else:
    print("沒有可用的數據可以生成報告。")