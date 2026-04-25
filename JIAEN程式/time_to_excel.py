import pandas as pd
import os

# 設定路徑
folder_name = "模型分析"
log_csv_path = "accuracy_result/PSO_TRENB.csv" 
output_excel_path = os.path.join(folder_name, "PSO_運行時間分析.xlsx")

# 確保資料夾存在
if not os.path.exists(folder_name):
    os.makedirs(folder_name)

try:
    # 1. 讀取原始的檔案
    df_log = pd.read_csv(log_csv_path)
    
    # 2. 只需要資料集名稱和時間(欄位名稱是 Dataset 和 Time)
    df_time = df_log[['Dataset', 'Time']].copy()

    # 3. 儲存成 Excel
    df_time.to_excel(output_excel_path, index=False)
    
    print(f"運行時間報告已生成：{output_excel_path}")
    print(df_time)

except FileNotFoundError:
    print(f"找不到檔案 {log_csv_path}，請確認 PSO_TRENB.py 是否已執行完成。")