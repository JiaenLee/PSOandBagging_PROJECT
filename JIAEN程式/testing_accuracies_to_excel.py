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
    讀取橫向 5-Fold CSV，旋轉為縱向後存為 XLSX，並加入類別類型欄位。
    """
    if not os.path.exists(input_csv):
        print(f"找不到檔案: {input_csv}")
        return

    # 1. 讀取 CSV
    df = pd.read_csv(input_csv)

    # 2. 使用 melt 功能將 Fold_1~Fold_5 轉為縱向 (正確率一欄)
    df_vertical = df.melt(
        id_vars=["Dataset"],
        value_vars=["Fold_1", "Fold_2", "Fold_3", "Fold_4", "Fold_5"],
        var_name="Fold_Index",
        value_name="正確率"
    )

    # 3. 加入類別類型欄位
    df_vertical["類別類型"] = df_vertical["Dataset"].map(category_map).fillna("未知")

    # 4. 重新命名欄位，讓格式與 training_accuracies 輸出一致
    df_vertical = df_vertical.rename(columns={"Dataset": "資料集名稱"})

    # 6. 只保留統一欄位順序
    df_final = df_vertical[["類別類型", "資料集名稱", "正確率"]]

    # 7. 存成 Excel 檔 (需要 openpyxl 套件)
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