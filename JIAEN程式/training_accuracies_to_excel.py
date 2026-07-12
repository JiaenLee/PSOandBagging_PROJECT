import pandas as pd
import json
import os

# 1. 設定路徑
folder_name = "模型分析"
output_filename = "PSO_模型正確率分析.xlsx"
output_path = os.path.join(folder_name, output_filename)
json_file_path = '../training_accuracies/PSO_TRENB.json'

# 資料來源根目錄 (用來找 CSV 檔案判斷類別)
dataset_base_dir = "../datasets/離散化資料集"

# 2. 檢查並建立「模型分析」資料夾
if not os.path.exists(folder_name):
    os.makedirs(folder_name)

# 3. 定義精準判斷類別的函式 (直接看 class 欄位有幾個值)
def determine_real_category(dataset_name):
    # 可能存放的路徑
    check_paths = [
        os.path.join(dataset_base_dir, "二類別", f"{dataset_name}.csv"),
        os.path.join(dataset_base_dir, "多類別", f"{dataset_name}.csv")
    ]
    
    for path in check_paths:
        if os.path.exists(path):
            try:
                # 只讀取 class 欄位來節省時間
                temp_df = pd.read_csv(path, usecols=['class'])
                class_count = temp_df['class'].nunique()
                return "二類別" if class_count == 2 else "多類別"
            except Exception:
                continue
    return "未知"

# 4. 讀取數據
try:
    with open(json_file_path, 'r', encoding='utf-8') as f:
        json_data = json.load(f)
except FileNotFoundError:
    print(f"錯誤：找不到 {json_file_path}")
    json_data = {}

# 5. 整理數據格式
print("正在根據資料集內容精準判斷類別類型...")
rows = []
for dataset_name, accuracies in json_data.items():
    # --- 直接去查 CSV 內容決定類別 ---
    category_type = determine_real_category(dataset_name)
    
    for acc in accuracies:
        rows.append({
            "類別類型": category_type,  # 改為精準判斷的結果
            "資料集名稱": dataset_name,
            "training正確率": acc             
        })

# 6. 建立 DataFrame
df = pd.DataFrame(rows)

# 7. 輸出成 Excel 檔
if not df.empty:
    # 這裡可以根據需要加上排序，讓結果更整齊
    df = df.sort_values(by=['類別類型', '資料集名稱']).reset_index(drop=True)
    
    df.to_excel(output_path, index=False)
    print("-" * 30)
    print(f"Excel 檔案已成功生成於：{output_path}")
    print(f"目前分類統計：\n{df['類別類型'].value_counts()}")
else:
    print("沒有可用的數據可以生成報告。")