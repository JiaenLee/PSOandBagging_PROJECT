import pandas as pd
import os

# 1. 設定路徑（已校正為相對於專案根目錄的正確路徑）
folder_name = "模型分析"
log_csv_path = "accuracy_result/Bagging.csv" 
output_excel_path = os.path.join(folder_name, "Bagging_運行時間分析.xlsx")

# 正確的資料來源資料夾路徑
binary_dir = "datasets/離散化資料集/二類別"
multi_dir = "datasets/離散化資料集/多類別"

# 2. 確保「模型分析」資料夾存在
if not os.path.exists(folder_name):
    os.makedirs(folder_name)

# 3. 建立「資料集 -> 分類」的對照字典
category_map = {}

def scan_datasets(path, label):
    if os.path.exists(path):
        for f in os.listdir(path):
            if f.endswith(".csv"):
                dataset_name = f.replace(".csv", "")
                category_map[dataset_name] = label

scan_datasets(binary_dir, "二類別")
scan_datasets(multi_dir, "多類別")

# 4. 讀取與處理數據
try:
    # 讀取 Bagging 的原始 CSV
    df_log = pd.read_csv(log_csv_path)
    
    # 新增「類別類型」欄位
    # 使用 map 函數根據 Dataset 名稱填入 二類別/多類別，找不到就填"未知"
    df_log['類別類型'] = df_log['Dataset'].map(category_map).fillna("未知")
    
    # 重新排列欄位順序：類別類型排第一，後面接著 Dataset 和 Time
    # 這樣會讓「類別類型」出現在 Excel 的 A 欄
    df_time = df_log[['類別類型', 'Dataset', 'Time']].copy()

    # 重新命名標題（可選，如果您想讓標題變成中文）
    df_time.columns = ['類別類型', '資料集名稱', '運行時間(秒)']

    # 5. 儲存成 Excel
    df_time.to_excel(output_excel_path, index=False)
    
    print("-" * 30)
    print(f"運行時間報告已生成：{output_excel_path}")
    print(df_time.head()) # 印出前幾行確認

except FileNotFoundError:
    print(f"找不到檔案 {log_csv_path}，請確認 Bagging 相關程式是否已執行完成並產出 CSV。")
except Exception as e:
    print(f"發生錯誤：{e}")