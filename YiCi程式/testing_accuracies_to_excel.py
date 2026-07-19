import pandas as pd
import os

def csv_to_vertical_xlsx(input_csv, output_xlsx):
    """
    讀取橫向 5-Fold CSV，直接從名稱拆分類別與資料集名。
    """
    if not os.path.exists(input_csv):
        print(f"找不到檔案：{input_csv}")
        return

    # 1. 讀取 CSV
    df = pd.read_csv(input_csv)

    # 2. 檢查必要欄位
    required_columns = ["Dataset", "Fold_1", "Fold_2", "Fold_3", "Fold_4", "Fold_5"]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"CSV 缺少必要欄位：{missing_columns}")
        return

    # 3. 拆分名稱並轉成縱向
    rows = []
    for _, row in df.iterrows():
        full_name = row["Dataset"]
        
        # 核心修改：直接從名稱拆分
        if "_" in str(full_name):
            category_type, dataset_name = str(full_name).split("_", 1)
        else:
            category_type, dataset_name = "未知", full_name

        for fold_idx in range(1, 6):
            rows.append({
                "類別類型": category_type,
                "資料集名稱": dataset_name,
                "testing正確率": row[f"Fold_{fold_idx}"]
            })

    df_final = pd.DataFrame(rows)

    # 4. 排序邏輯
    category_order = {"二類別": 0, "多類別": 1, "未知": 2}
    df_final["類別排序"] = df_final["類別類型"].map(category_order).fillna(2)
    df_final = df_final.sort_values(by=["類別排序", "資料集名稱"])
    df_final = df_final.drop(columns=["類別排序"])

    # 5. 存成 Excel
    os.makedirs(os.path.dirname(output_xlsx), exist_ok=True)
    df_final.to_excel(output_xlsx, index=False, engine="openpyxl")

    print(f"Excel 轉換完成！路徑：{output_xlsx}")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)

    # 以 Bagging 為例，若是 PSO 請修改檔名
    input_file = os.path.join(project_dir, "accuracy_result", "Bagging_fold_testing_accuracies.csv")
    output_file = os.path.join(script_dir, "模型分析", "Bagging_testing_accuracies分析.xlsx")

    csv_to_vertical_xlsx(input_file, output_file)