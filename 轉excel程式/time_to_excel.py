import pandas as pd
import os

def combine_time_to_excel(bagging_csv_path, pso_csv_path, output_xlsx):
    """
    合併 Bagging 與 PSO 的運行時間，並解析資料集名稱標籤。
    """
    if not os.path.exists(bagging_csv_path) or not os.path.exists(pso_csv_path):
        print(f"❌ 錯誤：找不到必要的 CSV 檔案。")
        if not os.path.exists(bagging_csv_path): print(f"缺少：{bagging_csv_path}")
        if not os.path.exists(pso_csv_path): print(f"缺少：{pso_csv_path}")
        return

    # 1. 讀取資料並將 Dataset 設為索引
    df_bag = pd.read_csv(bagging_csv_path).set_index("Dataset")
    df_pso = pd.read_csv(pso_csv_path).set_index("Dataset")

    # 取得兩份檔案中所有資料集的聯集
    all_datasets = sorted(list(set(df_bag.index) | set(df_pso.index)))

    rows = []

    # 2. 處理數據
    for dataset_full in all_datasets:
        # 解析標籤：二類別_Algerian -> 類型, 名稱
        if "_" in str(dataset_full):
            category_type, dataset_name = str(dataset_full).split("_", 1)
        else:
            category_type, dataset_name = "未知", str(dataset_full)

        # 取得時間資訊，若無則留白
        bag_time = df_bag.loc[dataset_full, "Total_Time"] if dataset_full in df_bag.index else ""
        pso_time = df_pso.loc[dataset_full, "Total_Time"] if dataset_full in df_pso.index else ""

        rows.append({
            "資料集類型": category_type,
            "資料集名稱": dataset_name,
            "Bagging運行時間": bag_time,
            "PSO運行時間": pso_time
        })

    # 3. 建立 DataFrame 並排序
    df_final = pd.DataFrame(rows)
    
    # 定義排序順序：二類別 -> 多類別
    category_order = {"二類別": 0, "多類別": 1}
    df_final["_sort"] = df_final["資料集類型"].map(category_order).fillna(2)
    df_final = df_final.sort_values(by=["_sort", "資料集名稱"])
    df_final = df_final.drop(columns=["_sort"])

    # 4. 輸出 Excel
    os.makedirs(os.path.dirname(output_xlsx), exist_ok=True)
    df_final.to_excel(output_xlsx, index=False, engine="openpyxl")
    
    print(f"✅ 運行時間合併完成！")
    print(f"📊 分析檔案已存至：{output_xlsx}")

if __name__ == "__main__":
    # --- 自動路徑處理 ---
    # 獲取目前腳本所在的資料夾 (轉excel程式)
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    # 獲取專案根目錄
    project_root = os.path.dirname(current_script_dir)

    # CSV 原始檔案路徑
    bagging_csv = os.path.join(project_root, "accuracy_result", "Bagging_Accuracy_Log.csv")
    pso_csv = os.path.join(project_root, "accuracy_result", "PSO_Accuracy_Log.csv")
    
    # 輸出的 Excel 路徑
    output_xlsx = os.path.join(project_root, "模型分析", "模型運行時間.xlsx")

    combine_time_to_excel(bagging_csv, pso_csv, output_xlsx)