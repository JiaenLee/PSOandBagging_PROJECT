import pandas as pd
import os

# 1. 設定路徑
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)

folder_name = os.path.join(script_dir, "模型分析")
log_csv_path = os.path.join(project_dir, "accuracy_result", "PSO_TRENB.csv") 
output_excel_path = os.path.join(folder_name, "PSO_運行時間分析.xlsx")

# 2. 確保資料夾存在
os.makedirs(folder_name, exist_ok=True)

# 3. 讀取與處理數據
try:
    df_log = pd.read_csv(log_csv_path)
    
    categories = []
    names = []

    # 核心修改：遍歷 Dataset 欄位直接拆分
    for full_name in df_log['Dataset']:
        if "_" in str(full_name):
            cat, nm = str(full_name).split("_", 1)
            categories.append(cat)
            names.append(nm)
        else:
            categories.append("未知")
            names.append(full_name)

    df_time = pd.DataFrame({
        '類別類型': categories,
        '資料集名稱': names,
        '運行時間(秒)': df_log['Time']
    })

    # 4. 排序並儲存
    category_order = {"二類別": 0, "多類別": 1, "未知": 2}
    df_time["sort"] = df_time["類別類型"].map(category_order).fillna(2)
    df_time = df_time.sort_values(by=["sort", "資料集名稱"]).drop(columns=["sort"])

    df_time.to_excel(output_excel_path, index=False)
    print(f"運行時間分析已生成：{output_excel_path}")

except FileNotFoundError:
    print(f"找不到檔案 {log_csv_path}")
except Exception as e:
    print(f"發生錯誤：{e}")