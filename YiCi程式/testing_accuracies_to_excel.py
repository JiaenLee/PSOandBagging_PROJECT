import pandas as pd
import os


def build_category_map(project_dir):
    """建立資料集名稱 -> 類別類型的對照表。"""
    category_map = {}

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

    for path, label in [
        (binary_dir, "二類別"),
        (multi_dir, "多類別")
    ]:
        if os.path.exists(path):
            for filename in os.listdir(path):
                if filename.lower().endswith(".csv"):
                    dataset_name = os.path.splitext(filename)[0]
                    category_map[dataset_name] = label
        else:
            print(f"找不到資料夾：{path}")

    return category_map


def csv_to_vertical_xlsx(input_csv, output_xlsx, category_map):
    """
    讀取 Bagging 橫向 5-Fold CSV，
    將每個資料集的 Fold_1～Fold_5 轉成縱向 5 行，
    並加入類別類型欄位。
    """
    if not os.path.exists(input_csv):
        print(f"找不到檔案：{input_csv}")
        return

    # 1. 讀取 CSV
    df = pd.read_csv(input_csv)

    # 2. 檢查必要欄位
    required_columns = [
        "Dataset",
        "Fold_1",
        "Fold_2",
        "Fold_3",
        "Fold_4",
        "Fold_5"
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        print(f"CSV 缺少必要欄位：{missing_columns}")
        return

    # 3. 將每個資料集的五折正確率轉成縱向
    rows = []

    for _, row in df.iterrows():
        dataset_name = row["Dataset"]
        category_type = category_map.get(dataset_name, "未知")

        for fold_idx in range(1, 6):
            rows.append({
                "類別類型": category_type,
                "資料集名稱": dataset_name,
                "testing正確率": row[f"Fold_{fold_idx}"]
            })

    df_final = pd.DataFrame(rows)

    # 4. 依照類別與資料集名稱排序
    category_order = {
        "二類別": 0,
        "多類別": 1,
        "未知": 2
    }

    df_final["類別排序"] = (
        df_final["類別類型"]
        .map(category_order)
        .fillna(2)
    )

    df_final = df_final.sort_values(
        by=["類別排序", "資料集名稱"],
        key=lambda column: (
            column.str.lower()
            if column.name == "資料集名稱"
            else column
        )
    )

    df_final = df_final.drop(columns=["類別排序"])

    # 5. 確保輸出資料夾存在
    output_dir = os.path.dirname(output_xlsx)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # 6. 存成 Excel
    df_final.to_excel(
        output_xlsx,
        index=False,
        engine="openpyxl"
    )

    print(f"Excel 轉換完成！檔案路徑：{output_xlsx}")
    print(f"分類統計：\n{df_final['類別類型'].value_counts()}")


if __name__ == "__main__":
    # 此程式所在資料夾：YiCi程式
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # 專案根目錄：PSOandBagging_PROJECT
    project_dir = os.path.dirname(script_dir)

    # Bagging 五折測試正確率 CSV
    input_file = os.path.join(
        project_dir,
        "accuracy_result",
        "Bagging_fold_testing_accuracies.csv"
    )

    # Excel 輸出至 YiCi程式/模型分析
    output_dir = os.path.join(
        script_dir,
        "模型分析"
    )

    output_file = os.path.join(
        output_dir,
        "Bagging_testing_accuracies分析.xlsx"
    )

    category_map = build_category_map(project_dir)

    csv_to_vertical_xlsx(
        input_file,
        output_file,
        category_map
    )