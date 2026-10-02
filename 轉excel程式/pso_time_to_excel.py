import os

import pandas as pd


def pso_time_to_excel(pso_csv_path, output_xlsx):
    """
    讀取 PSO Accuracy Log 中的運行時間，
    並整理成 Excel。

    輸出欄位：
    - 資料集類型
    - 資料集名稱
    - PSO運行時間
    """

    # 1. 確認 PSO CSV 是否存在
    if not os.path.exists(pso_csv_path):
        print("錯誤：找不到 PSO CSV 檔案。")
        print(f"缺少：{pso_csv_path}")
        return

    # 2. 讀取 CSV
    try:
        df_pso = pd.read_csv(pso_csv_path)
    except Exception as error:
        print("PSO CSV 讀取失敗。")
        print(f"錯誤訊息：{error}")
        return

    # 3. 確認必要欄位存在
    required_columns = {"Dataset", "Total_Time"}
    missing_columns = required_columns - set(df_pso.columns)

    if missing_columns:
        print("CSV 缺少必要欄位。")
        print(f"缺少欄位：{sorted(missing_columns)}")
        print(f"目前欄位：{list(df_pso.columns)}")
        return

    rows = []

    # 4. 整理資料
    for _, row in df_pso.iterrows():
        dataset_full = str(row["Dataset"])

        # 解析名稱：
        # 二類別_Algerian -> 二類別、Algerian
        if "_" in dataset_full:
            category_type, dataset_name = dataset_full.split("_", 1)
        else:
            category_type = "未知"
            dataset_name = dataset_full

        rows.append({
            "資料集類型": category_type,
            "資料集名稱": dataset_name,
            "PSO運行時間": row["Total_Time"]
        })

    # 5. 建立 DataFrame
    df_final = pd.DataFrame(rows)

    if df_final.empty:
        print("沒有 PSO 運行時間資料可供處理。")
        return

    # 將運行時間轉成 Excel 可計算的數值格式
    df_final["PSO運行時間"] = pd.to_numeric(
        df_final["PSO運行時間"],
        errors="coerce"
    )

    # 定義排序順序：
    # 二類別 -> 多類別 -> 其他
    category_order = {
        "二類別": 0,
        "多類別": 1
    }

    df_final["_sort"] = (
        df_final["資料集類型"]
        .map(category_order)
        .fillna(2)
    )

    df_final = (
        df_final
        .sort_values(
            by=["_sort", "資料集名稱"],
            kind="stable"
        )
        .drop(columns=["_sort"])
        .reset_index(drop=True)
    )

    # 6. 建立輸出資料夾
    output_dir = os.path.dirname(output_xlsx)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # 7. 輸出 Excel
    try:
        df_final.to_excel(
            output_xlsx,
            index=False,
            engine="openpyxl"
        )
    except Exception as error:
        print("Excel 輸出失敗。")
        print(f"錯誤訊息：{error}")
        return

    print("PSO 運行時間轉換完成！")
    print(f"分析檔案已存至：{output_xlsx}")


if __name__ == "__main__":
    # 目前腳本所在資料夾：
    # PSOandBagging_PROJECT/轉excel程式
    current_script_dir = os.path.dirname(
        os.path.abspath(__file__)
    )

    # 專案根目錄：
    # PSOandBagging_PROJECT
    project_root = os.path.dirname(current_script_dir)

    # PSO Accuracy Log CSV
    pso_csv_path = os.path.join(
        project_root,
        "accuracy_result",
        "PSO_Accuracy_Log.csv"
    )

    # 輸出的 Excel
    output_xlsx_path = os.path.join(
        project_root,
        "模型分析",
        "pso模型運行時間.xlsx"
    )

    pso_time_to_excel(
        pso_csv_path,
        output_xlsx_path
    )
