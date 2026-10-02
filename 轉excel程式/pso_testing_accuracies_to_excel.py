import os
import pandas as pd


def pso_accuracies_to_excel(pso_csv, output_xlsx):
    """
    讀取 PSO Accuracy Log，將各資料集的 5-Fold Testing Accuracy
    由「正確率:筆數」格式拆分後輸出成 Excel。

    輸出欄位：
    - 資料集類型
    - 資料集名稱
    - fold
    - 每折資料筆數
    - PSO平均正確率
    """

    # 1. 確認 PSO CSV 是否存在
    if not os.path.exists(pso_csv):
        print("錯誤：找不到 PSO CSV 檔案。")
        print(f"缺少：{pso_csv}")
        return

    # 2. 讀取 PSO 原始資料
    try:
        df_pso = pd.read_csv(pso_csv)
    except Exception as error:
        print("PSO CSV 讀取失敗。")
        print(f"錯誤訊息：{error}")
        return

    # 3. 確認必要欄位存在
    if "Dataset" not in df_pso.columns:
        print("CSV 中找不到必要欄位：Dataset")
        print(f"目前欄位：{list(df_pso.columns)}")
        return

    df_pso = df_pso.set_index("Dataset")

    all_rows = []

    # 4. 逐一處理資料集
    for dataset_full in df_pso.index:
        dataset_full = str(dataset_full)

        # 拆分資料集類型與資料集名稱
        if "_" in dataset_full:
            category_type, dataset_name = dataset_full.split("_", 1)
        else:
            category_type = "未知"
            dataset_name = dataset_full

        # 處理 5 折資料
        for fold_index in range(1, 6):
            column_name = f"Fold_{fold_index}:num"

            pso_accuracy = ""
            sample_num = ""

            if column_name in df_pso.columns:
                value = df_pso.loc[dataset_full, column_name]

                # 避免 Dataset 重複時回傳 Series
                if isinstance(value, pd.Series):
                    value = value.iloc[0]

                if pd.notna(value):
                    value_text = str(value).strip()

                    if ":" in value_text:
                        try:
                            accuracy_text, sample_text = value_text.split(":", 1)
                            pso_accuracy = accuracy_text.strip()
                            sample_num = sample_text.strip()
                        except ValueError:
                            pass

            all_rows.append({
                "資料集類型": category_type,
                "資料集名稱": dataset_name,
                "fold": fold_index,
                "每折資料筆數": sample_num,
                "PSO平均正確率": pso_accuracy
            })

    # 5. 建立並排序 DataFrame
    df_final = pd.DataFrame(all_rows)

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
            by=["_sort", "資料集名稱", "fold"],
            kind="stable"
        )
        .drop(columns=["_sort"])
        .reset_index(drop=True)
    )

    # 將可轉換的欄位改為數值，空白仍保留
    df_final["每折資料筆數"] = pd.to_numeric(
        df_final["每折資料筆數"],
        errors="coerce"
    )

    df_final["PSO平均正確率"] = pd.to_numeric(
        df_final["PSO平均正確率"],
        errors="coerce"
    )

    # 6. 建立輸出資料夾並輸出 Excel
    output_dir = os.path.dirname(output_xlsx)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

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

    print("PSO Testing Accuracy 轉換完成！")
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

    # PSO Testing Accuracy CSV
    pso_path = os.path.join(
        project_root,
        "accuracy_result",
        "PSO_Accuracy_Log.csv"
    )

    # 輸出的 Excel
    output_xlsx_path = os.path.join(
        project_root,
        "模型分析",
        "pso_testing正確率.xlsx"
    )

    pso_accuracies_to_excel(
        pso_path,
        output_xlsx_path
    )
