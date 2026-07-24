import pandas as pd
import json
import os

def combine_training_to_excel(bagging_json_path, pso_json_path, output_xlsx):
    """
    讀取 Bagging 與 PSO 的 Training JSON，將 125 個正確率對齊後轉為縱向 Excel。
    """
    # 1. 讀取 JSON 資料
    json_bag = {}
    json_pso = {}

    if os.path.exists(bagging_json_path):
        with open(bagging_json_path, 'r', encoding='utf-8') as f:
            json_bag = json.load(f)
    else:
        print(f"⚠️ 找不到 Bagging JSON: {bagging_json_path}")

    if os.path.exists(pso_json_path):
        with open(pso_json_path, 'r', encoding='utf-8') as f:
            json_pso = json.load(f)
    else:
        print(f"⚠️ 找不到 PSO JSON: {pso_json_path}")

    # 取得兩者資料集的聯集並排序
    all_keys = sorted(list(set(json_bag.keys()) | set(json_pso.keys())))
    
    rows = []

    # 2. 整理數據
    for full_name in all_keys:
        # 解析名稱：二類別_Algerian -> 類型, 名稱
        if "_" in str(full_name):
            category_type, dataset_name = str(full_name).split("_", 1)
        else:
            category_type, dataset_name = "未知", str(full_name)

        # 取得正確率列表（預期各 125 個）
        bag_list = json_bag.get(full_name, [])
        pso_list = json_pso.get(full_name, [])

        # 以最長的列表長度為準進行對齊 (通常是 125)
        max_len = max(len(bag_list), len(pso_list))

        for i in range(max_len):
            b_acc = bag_list[i] if i < len(bag_list) else ""
            p_acc = pso_list[i] if i < len(pso_list) else ""

            rows.append({
                "資料集類型": category_type,
                "資料集名稱": dataset_name,
                "Bagging正確率": b_acc,
                "PSO正確率": p_acc
            })

    # 3. 建立 DataFrame 並排序
    df_final = pd.DataFrame(rows)
    
    if not df_final.empty:
        # 定義排序順序：二類別 -> 多類別
        category_order = {"二類別": 0, "多類別": 1}
        df_final["_sort"] = df_final["資料集類型"].map(category_order).fillna(2)
        df_final = df_final.sort_values(by=["_sort", "資料集名稱"])
        df_final = df_final.drop(columns=["_sort"])

        # 4. 輸出 Excel
        os.makedirs(os.path.dirname(output_xlsx), exist_ok=True)
        df_final.to_excel(output_xlsx, index=False, engine="openpyxl")
        print(f"✅ Training 正確率合併完成！")
        print(f"📊 分析檔案已存至：{output_xlsx}")
    else:
        print("❌ 沒有數據可供處理。")

if __name__ == "__main__":
    # --- 自動路徑處理 ---
    # 獲取目前腳本所在資料夾 (轉excel程式)
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    # 獲取專案根目錄
    project_root = os.path.dirname(current_script_dir)

    # JSON 原始檔案路徑
    bagging_json = os.path.join(project_root, "training_accuracies", "Bagging.json")
    pso_json = os.path.join(project_root, "training_accuracies", "PSO_TRENB.json")
    
    # 輸出的 Excel 路徑
    output_xlsx = os.path.join(project_root, "模型分析", "training正確率.xlsx")

    combine_training_to_excel(bagging_json, pso_json, output_xlsx)