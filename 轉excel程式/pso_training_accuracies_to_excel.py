import json
import os

import pandas as pd


def pso_training_to_excel(pso_json_path, output_xlsx):
    """
    讀取 PSO 的 Training Accuracy JSON，
    將各資料集的正確率轉為縱向 Excel。

    輸出欄位：
    - 資料集類型
    - 資料集名稱
    - 模型編號
    - PSO正確率
    """

    # 1. 確認 PSO JSON 是否存在
    if not os.path.exists(pso_json_path):
        print("❌ 錯誤：找不到 PSO Training JSON。")
        print(f"缺少：{pso_json_path}")
        return

    # 2. 讀取 JSON
    try:
        with open(pso_json_path, "r", encoding="utf-8") as file:
            json_pso = json.load(file)
    except json.JSONDecodeError as error:
        print("❌ PSO JSON 格式錯誤，無法讀取。")
        print(f"錯誤訊息：{error}")
        return
    except Exception as error:
        print("❌ PSO JSON 讀取失敗。")
        print(f"錯誤訊息：{error}")
        return

    if not isinstance(json_pso, dict):
        print("❌ JSON 最外層格式應為物件（dict）。")
        return

    rows = []

    # 3. 整理各資料集的 Training Accuracy
    for full_name, pso_list in json_pso.items():
        full_name = str(full_name)

        # 解析名稱，例如：
        # 二類別_Algerian -> 二類別、Algerian
        if "_" in full_name:
            category_type, dataset_name = full_name.split("_", 1)
        else:
            category_type = "未知"
            dataset_name = full_name

        # 確認正確率資料是列表
        if not isinstance(pso_list, list):
            print(f"⚠️ 跳過 {full_name}：資料不是列表。")
            continue

        # 每個資料集通常會有 125 筆 Training Accuracy
        for index, accuracy in enumerate(pso_list, start=1):
            rows.append({
                "資料集類型": category_type,
                "資料集名稱": dataset_name,
                "PSO正確率": accuracy
            })

    # 4. 建立 DataFrame
    df_final = pd.DataFrame(rows)

    if df_final.empty:
        print("❌ 沒有 PSO Training Accuracy 可供處理。")
        return

    # 將正確率轉成可供 Excel 計算的數值格式
    df_final["PSO正確率"] = pd.to_numeric(
        df_final["PSO正確率"],
        errors="coerce"
    )

    # 定義資料集類型排序：
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

    # 5. 建立輸出資料夾
    output_dir = os.path.dirname(output_xlsx)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # 6. 輸出 Excel
    try:
        df_final.to_excel(
            output_xlsx,
            index=False,
            engine="openpyxl"
        )
    except Exception as error:
        print("❌ Excel 輸出失敗。")
        print(f"錯誤訊息：{error}")
        return

    print("✅ PSO Training Accuracy 轉換完成！")
    print(f"📊 分析檔案已存至：{output_xlsx}")
    print(f"📌 共輸出 {len(df_final)} 筆資料。")


if __name__ == "__main__":
    # 目前腳本所在資料夾：
    # PSOandBagging_PROJECT/轉excel程式
    current_script_dir = os.path.dirname(
        os.path.abspath(__file__)
    )

    # 專案根目錄：
    # PSOandBagging_PROJECT
    project_root = os.path.dirname(current_script_dir)

    # PSO Training Accuracy JSON
    pso_json_path = os.path.join(
        project_root,
        "training_accuracies",
        "PSO_TRENB.json"
    )

    # 輸出的 Excel
    output_xlsx_path = os.path.join(
        project_root,
        "模型分析",
        "pso_training正確率.xlsx"
    )

    pso_training_to_excel(
        pso_json_path,
        output_xlsx_path
    )
