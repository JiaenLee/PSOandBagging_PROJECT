import pandas as pd
import os


def csv_to_vertical_xlsx(input_csv, output_xlsx):
    """
    讀取 Bagging 的橫向 5-Fold CSV，
    將 Fold_1～Fold_5 轉成縱向格式後存為 XLSX。
    """
    if not os.path.exists(input_csv):
        print(f"找不到檔案: {input_csv}")
        return

    # 讀取 CSV
    df = pd.read_csv(input_csv)

    # 檢查必要欄位是否存在
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
        print(f"CSV 缺少欄位: {missing_columns}")
        return

    # 將 Fold_1～Fold_5 轉為縱向
    df_vertical = df.melt(
        id_vars=["Dataset"],
        value_vars=[
            "Fold_1",
            "Fold_2",
            "Fold_3",
            "Fold_4",
            "Fold_5"
        ],
        var_name="Fold_Index",
        value_name="Testing正確率"
    )

    # 將 Fold_Index 轉成數字
    df_vertical["Fold_Number"] = (
        df_vertical["Fold_Index"]
        .str.replace("Fold_", "", regex=False)
        .astype(int)
    )

    # 讓同一資料集的五個 Fold 排在一起
    df_vertical = df_vertical.sort_values(
        by=["Dataset", "Fold_Number"]
    )

    # 只保留資料集名稱與 Testing 正確率
    df_final = df_vertical[
        ["Dataset", "Testing正確率"]
    ]

    # 確保輸出資料夾存在
    output_folder = os.path.dirname(output_xlsx)

    if output_folder:
        os.makedirs(output_folder, exist_ok=True)

    # 輸出 Excel
    df_final.to_excel(
        output_xlsx,
        index=False,
        engine="openpyxl"
    )

    print(f"Excel 轉換完成！檔案路徑: {output_xlsx}")


if __name__ == "__main__":
    # 此程式所在位置：YiCi程式
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # YiCi程式的上一層：專案根目錄
    project_dir = os.path.dirname(script_dir)

    # 讀取專案根目錄 accuracy_result 中的 CSV
    input_file = os.path.join(
        project_dir,
        "accuracy_result",
        "Bagging_fold_testing_accuracies.csv"
    )

    # 輸出到 YiCi程式/模型分析
    output_file = os.path.join(
        script_dir,
        "模型分析",
        "Bagging_fold_testing_accuracies.xlsx"
    )

    csv_to_vertical_xlsx(input_file, output_file)