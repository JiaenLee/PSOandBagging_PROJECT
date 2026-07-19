import pandas as pd
import json
import os

# 1. 設定路徑
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)

folder_name = os.path.join(script_dir, "模型分析")
output_path = os.path.join(folder_name, "Bagging_training_accuracies分析.xlsx")
json_file_path = os.path.join(project_dir, "training_accuracies", "Bagging.json")

# 2. 建立資料夾
os.makedirs(folder_name, exist_ok=True)

# 3. 讀取 JSON 並整理資料
try:
    with open(json_file_path, "r", encoding="utf-8") as file:
        json_data = json.load(file)
    
    rows = []
    for full_name, accuracies in json_data.items():
        # 核心修改：直接從 Key 拆分
        if "_" in str(full_name):
            category_type, dataset_name = str(full_name).split("_", 1)
        else:
            category_type, dataset_name = "未知", str(full_name)

        for acc in accuracies:
            rows.append({
                "類別類型": category_type,
                "資料集名稱": dataset_name,
                "正確率": acc
            })

    # 4. 建立 DataFrame 並排序
    df = pd.DataFrame(rows)
    if not df.empty:
        category_order = {"二類別": 0, "多類別": 1, "未知": 2}
        df["sort"] = df["類別類型"].map(category_order).fillna(2)
        df = df.sort_values(by=["sort", "資料集名稱"]).drop(columns=["sort"])
        
        df.to_excel(output_path, index=False)
        print(f"Excel 檔案已成功生成：{output_path}")
    else:
        print("沒有數據可生成報告。")

except FileNotFoundError:
    print(f"找不到 JSON 檔案：{json_file_path}")