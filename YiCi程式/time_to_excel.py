import pandas as pd
import os

# 1. 取得資料夾位置

# time_to_excel.py 所在資料夾：YiCi程式
script_dir = os.path.dirname(os.path.abspath(__file__))

# 專案根目錄：PSOandBagging_PROJECT
project_dir = os.path.dirname(script_dir)

# Excel 輸出資料夾：YiCi程式/模型分析
folder_name = os.path.join(
    script_dir,
    "模型分析"
)

# Bagging.csv 位於專案根目錄的 accuracy_result
log_csv_path = os.path.join(
    project_dir,
    "accuracy_result",
    "Bagging.csv"
)

# Excel 輸出位置
output_excel_path = os.path.join(
    folder_name,
    "Bagging_運行時間分析.xlsx"
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

# 2. 確保 YiCi程式/模型分析 存在
os.makedirs(folder_name, exist_ok=True)

# 3. 建立「資料集名稱 -> 分類」對照表
category_map = {}


def scan_datasets(path, label):
    if not os.path.exists(path):
        print(f"找不到資料夾：{path}")
        return

    for filename in os.listdir(path):
        if filename.lower().endswith(".csv"):
            dataset_name = os.path.splitext(filename)[0]
            category_map[dataset_name] = label


scan_datasets(binary_dir, "二類別")
scan_datasets(multi_dir, "多類別")

# 4. 讀取與處理資料
try:
    df_log = pd.read_csv(log_csv_path)

    # 檢查必要欄位
    required_columns = ["Dataset", "Time"]

    missing_columns = [
        column
        for column in required_columns
        if column not in df_log.columns
    ]

    if missing_columns:
        print(f"Bagging.csv 缺少必要欄位：{missing_columns}")
    else:
        # 加入類別類型
        df_log["類別類型"] = (
            df_log["Dataset"]
            .map(category_map)
            .fillna("未知")
        )

        # 取出需要的欄位
        df_time = df_log[
            ["類別類型", "Dataset", "Time"]
        ].copy()

        # 改成中文欄位名稱
        df_time.columns = [
            "類別類型",
            "資料集名稱",
            "運行時間(秒)"
        ]

        # 5. 儲存 Excel
        df_time.to_excel(
            output_excel_path,
            index=False,
            engine="openpyxl"
        )

        print("-" * 30)
        print(f"運行時間報告已生成：{output_excel_path}")
        print(f"分類統計：\n{df_time['類別類型'].value_counts()}")
        print(df_time.head())

except FileNotFoundError:
    print(
        f"找不到檔案：{log_csv_path}\n"
        "請確認 Bagging 程式已執行並產生 Bagging.csv。"
    )

except PermissionError:
    print(
        f"無法寫入 Excel：{output_excel_path}\n"
        "請先關閉目前已開啟的 Excel 檔案，再重新執行。"
    )

except Exception as e:
    print(f"發生錯誤：{e}")