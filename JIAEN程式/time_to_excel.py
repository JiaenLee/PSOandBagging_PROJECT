import pandas as pd
import os

# 1. 設定路徑
folder_name = "模型分析"
log_csv_path = "../accuracy_result/PSO_TRENB.csv" 
output_excel_path = os.path.join(folder_name, "PSO_運行時間分析.xlsx")

# 資料來源根目錄
dataset_base_dir = "../datasets/離散化資料集"

# 2. 確保「模型分析」資料夾存在
if not os.path.exists(folder_name):
    os.makedirs(folder_name)

# 3. 定義一個函式，根據資料集內容來精準判斷類別
def determine_real_category(dataset_name):
    # 可能出現的路徑 (二類別或多類別資料夾)
    paths = [
        os.path.join(dataset_base_dir, "二類別", f"{dataset_name}.csv"),
        os.path.join(dataset_base_dir, "多類別", f"{dataset_name}.csv")
    ]
    
    for path in paths:
        if os.path.exists(path):
            try:
                # 唯讀模式讀取類別欄位
                temp_df = pd.read_csv(path, usecols=['class'])
                class_count = temp_df['class'].nunique() # 計算不重複的類別數量
                
                if class_count == 2:
                    return "二類別"
                else:
                    return "多類別"
            except Exception:
                continue
                
    return "未知"

# 4. 讀取與處理數據
try:
    df_log = pd.read_csv(log_csv_path)
    
    print("正在精準判斷每個資料集的類別類型...")
    
    # 使用 apply 搭配剛剛寫的函式
    df_log['類別類型'] = df_log['Dataset'].apply(determine_real_category)
    
    # 重新排列欄位順序
    df_time = df_log[['類別類型', 'Dataset', 'Time']].copy()
    df_time.columns = ['類別類型', '資料集名稱', '運行時間(秒)']

    # 5. 儲存成 Excel
    df_time.to_excel(output_excel_path, index=False)
    
    print("-" * 30)
    print(f"運行時間報告已生成：{output_excel_path}")
    print(df_time.head())

except FileNotFoundError:
    print(f"找不到檔案 {log_csv_path}")
except Exception as e:
    print(f"發生錯誤：{e}")