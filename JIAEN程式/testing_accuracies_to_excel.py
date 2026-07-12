import pandas as pd
import os


def build_category_map(project_dir):
    """建立資料集名稱 -> 類別類型的對照表。"""
    category_map = {}

    binary_dir = os.path.join(project_dir, "datasets", "離散化資料集", "二類別")
    multi_dir = os.path.join(project_dir, "datasets", "離散化資料集", "多類別")

    for path, label in [(binary_dir, "二類別"), (multi_dir, "多類別")]:
        if os.path.exists(path):
            for filename in os.listdir(path):
                if filename.lower().endswith(".csv"):
                    dataset_name = os.path.splitext(filename)[0]
                    category_map[dataset_name] = label

    return category_map


def csv_to_vertical_xlsx(input_csv, output_xlsx, category_map):
    """
    讀取橫向 5-Fold CSV，逐列轉成縱向 5 行，並加入類別類型欄位。
    """
    if not os.path.exists(input_csv):
        print(f"找不到檔案: {input_csv}")
        return

    # 1. 讀取 CSV
    df = pd.read_csv(input_csv)

    rows = []
    for _, row in df.iterrows():
        dataset_name = row["Dataset"]
        category_type = category_map.get(dataset_name, "未知")

        for fold_idx in range(1, 6):
            rows.append({
                "類別類型": category_type,
                "資料集名稱": dataset_name,
                "testing正確率": row[f"Fold_{fold_idx}"],
            })

    df_final = pd.DataFrame(rows)

    # 2. 存成 Excel 檔 (需要 openpyxl 套件)
    os.makedirs(os.path.dirname(output_xlsx), exist_ok=True)
    df_final.to_excel(output_xlsx, index=False)
    print(f"Excel 轉換完成！檔案路徑: {output_xlsx}")


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)

    input_file = os.path.join(project_dir, "accuracy_result", "PSO_fold_testing_accuracies.csv")
    output_dir = os.path.join(script_dir, "模型分析")
    output_file = os.path.join(output_dir, "PSO_testing_accuracies分析.xlsx")

    category_map = build_category_map(project_dir)
    csv_to_vertical_xlsx(input_file, output_file, category_map)