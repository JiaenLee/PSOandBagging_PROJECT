import pandas as pd
import os

def csv_to_vertical_xlsx(input_csv, output_xlsx):
    """
    讀取橫向 CSV，從 Dataset 欄位(如：二類別_Algerian)拆分出類別類型。
    """
    if not os.path.exists(input_csv):
        print(f"找不到檔案: {input_csv}")
        return

    df = pd.read_csv(input_csv)
    rows = []

    for _, row in df.iterrows():
        full_name = row["Dataset"]
        
        # 直接解析名稱
        if "_" in full_name:
            category_type, dataset_name = full_name.split("_", 1)
        else:
            category_type, dataset_name = "未知", full_name

        for fold_idx in range(1, 6):
            rows.append({
                "類別類型": category_type,
                "資料集名稱": dataset_name,
                "testing正確率": row[f"Fold_{fold_idx}"],
            })

    df_final = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(output_xlsx), exist_ok=True)
    df_final.to_excel(output_xlsx, index=False)
    print(f"Excel 轉換完成！檔案路徑: {output_xlsx}")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)

    input_file = os.path.join(project_dir, "accuracy_result", "PSO_fold_testing_accuracies.csv")
    output_dir = os.path.join(script_dir, "模型分析")
    output_file = os.path.join(output_dir, "PSO_testing_accuracies分析.xlsx")

    csv_to_vertical_xlsx(input_file, output_file)