import pandas as pd
import os

# 1. 設定路徑
folder_name = "模型分析"
log_csv_path = "../accuracy_result/PSO_TRENB.csv" 
output_excel_path = os.path.join(folder_name, "PSO_運行時間分析.xlsx")

# 2. 確保資料夾存在
if not os.path.exists(folder_name):
    os.makedirs(folder_name)

# 3. 讀取與處理數據
try:
    df_log = pd.read_csv(log_csv_path)
    
    # 建立空的 List 準備存放拆分後的結果
    categories = []
    names = []

    for full_name in df_log['Dataset']:
        if "_" in str(full_name):
            cat, nm = str(full_name).split("_", 1)
            categories.append(cat)
            names.append(nm)
        else:
            categories.append("未知")
            names.append(full_name)

    # 建立新的 DataFrame
    df_time = pd.DataFrame({
        '類別類型': categories,
        '資料集名稱': names,
        '運行時間(秒)': df_log['Time']
    })

    # 4. 儲存成 Excel
    df_time.to_excel(output_excel_path, index=False)
    
    print("-" * 30)
    print(f"運行時間報告已生成：{output_excel_path}")
    print(df_time.head())

except FileNotFoundError:
    print(f"找不到檔案 {log_csv_path}。")
except Exception as e:
    print(f"發生錯誤：{e}")