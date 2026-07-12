import pandas as pd
import os

# 1. 設定路徑
folder_name = "模型分析"
log_csv_path = "../accuracy_result/PSO_TRENB.csv" 
output_excel_path = os.path.join(folder_name, "PSO_運行時間分析.xlsx")

# 資料來源根目錄
DATASET_BASE_DIR = "../datasets/離散化資料集"

# 2. 確保「模型分析」資料夾存在
if not os.path.exists(folder_name):
    os.makedirs(folder_name)

def get_ordered_categories():
    """
    模擬 PSO_TRENB.py 的掃描順序，生成精準的標籤序列
    """
    ordered_labels = []
    folders = ["二類別", "多類別"]
    for folder in folders:
        path = os.path.join(DATASET_BASE_DIR, folder)
        if os.path.exists(path):
            # 取得 CSV 並排序 (與主程式 logic 一致)
            files = sorted([f.replace(".csv", "") for f in os.listdir(path) if f.endswith(".csv")])
            for filename in files:
                ordered_labels.append({"Dataset": filename, "類別類型": folder})
    return pd.DataFrame(ordered_labels)

def determine_by_content(dataset_name):
    """
    Fallback 方案：直接讀取 CSV 內容來判斷類別數
    """
    for folder in ["多類別", "二類別"]: # 優先查多類別
        path = os.path.join(DATASET_BASE_DIR, folder, f"{dataset_name}.csv")
        if os.path.exists(path):
            try:
                temp_df = pd.read_csv(path, usecols=['class'])
                count = temp_df['class'].nunique()
                return "多類別" if count > 2 else "二類別"
            except:
                continue
    return "未知"

# 3. 讀取與處理數據
try:
    # 讀取原始運行紀錄
    df_log = pd.read_csv(log_csv_path)
    
    # 取得理論上的正確順序標籤
    df_ordered = get_ordered_categories()

    print("正在精準匹配資料集類別類型...")

    # 如果行數完全對應，直接按順序填入標籤 (最精準，可分辨同名但不同類別的資料集)
    if len(df_log) == len(df_ordered):
        df_log['類別類型'] = df_ordered['類別類型']
    else:
        # 如果行數不對 (可能中途停止)，改用內容查表法
        print("[提示] 紀錄行數與檔案總數不符，改用內容驗證法...")
        df_log['類別類型'] = df_log['Dataset'].apply(determine_by_content)
    
    # 4. 整理格式
    # 重新排列欄位順序
    df_time = df_log[['類別類型', 'Dataset', 'Time']].copy()

    # 重新命名標題為中文
    df_time.columns = ['類別類型', '資料集名稱', '運行時間(秒)']

    # 根據類別和名稱排序，讓報表更整潔
    df_time = df_time.sort_values(by=['類別類型', '資料集名稱']).reset_index(drop=True)

    # 5. 儲存成 Excel
    df_time.to_excel(output_excel_path, index=False)
    
    print("-" * 30)
    print(f"運行時間報告已生成：{output_excel_path}")
    print(f"類別統計：\n{df_time['類別類型'].value_counts()}")

except FileNotFoundError:
    print(f"找不到檔案 {log_csv_path}")
except Exception as e:
    import traceback
    print(f"發生錯誤：\n{traceback.format_exc()}")