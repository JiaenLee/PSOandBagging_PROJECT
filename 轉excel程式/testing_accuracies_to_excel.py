import pandas as pd
import numpy as np
import os

def combine_accuracies_to_excel(bagging_csv, pso_csv, output_xlsx):
    """
    讀取 Bagging 與 PSO 的 Accuracy Log，合併並拆分 "正確率:筆數" 格式。
    若該折無資訊，則在 Excel 中留白。
    """
    if not os.path.exists(bagging_csv) or not os.path.exists(pso_csv):
        print(f"錯誤：找不到必要的 CSV 檔案。")
        if not os.path.exists(bagging_csv): print(f"缺少：{bagging_csv}")
        if not os.path.exists(pso_csv): print(f"缺少：{pso_csv}")
        return

    # 1. 讀取兩份原始資料
    df_bag = pd.read_csv(bagging_csv).set_index("Dataset")
    df_pso = pd.read_csv(pso_csv).set_index("Dataset")

    # 2. 取得所有出現在兩份檔案中的資料集聯集 (Union)
    all_datasets = sorted(list(set(df_bag.index) | set(df_pso.index)))

    all_rows = []

    # 3. 解析與合併資料
    for dataset_full in all_datasets:
        # 解析名稱標籤
        if "_" in str(dataset_full):
            category_type, dataset_name = str(dataset_full).split("_", 1)
        else:
            category_type, dataset_name = "未知", dataset_full

        # 遍歷 5 折
        for f_idx in range(1, 6):
            col_name = f"Fold_{f_idx}:num"
            
            bag_acc = ""
            pso_acc = ""
            sample_num = ""

            # --- 嘗試提取 Bagging 資訊 ---
            if dataset_full in df_bag.index and col_name in df_bag.columns:
                val = df_bag.loc[dataset_full, col_name]
                if pd.notna(val) and ":" in str(val):
                    try:
                        bag_acc, s_num = str(val).split(":")
                        sample_num = s_num # 優先使用 Bagging 提供的樣本數
                    except:
                        pass

            # --- 嘗試提取 PSO 資訊 ---
            if dataset_full in df_pso.index and col_name in df_pso.columns:
                val = df_pso.loc[dataset_full, col_name]
                if pd.notna(val) and ":" in str(val):
                    try:
                        p_acc, p_s_num = str(val).split(":")
                        pso_acc = p_acc
                        # 如果前面 Bagging 沒抓到樣本數，改用 PSO 的
                        if not sample_num:
                            sample_num = p_s_num
                    except:
                        pass

            all_rows.append({
                "資料集類型": category_type,
                "資料集名稱": dataset_name,
                "fold": f_idx,
                "每折資料筆數": sample_num,
                "Bagging平均正確率": bag_acc,
                "PSO平均正確率": pso_acc
            })

    # 4. 建立 DataFrame 並排序
    df_final = pd.DataFrame(all_rows)
    
    # 定義排序順序
    category_order = {"二類別": 0, "多類別": 1}
    df_final["_sort"] = df_final["資料集類型"].map(category_order).fillna(2)
    df_final = df_final.sort_values(by=["_sort", "資料集名稱", "fold"])
    df_final = df_final.drop(columns=["_sort"])

    # 5. 輸出 Excel
    os.makedirs(os.path.dirname(output_xlsx), exist_ok=True)
    df_final.to_excel(output_xlsx, index=False, engine="openpyxl")

    print(f"合併轉換完成！")
    print(f"分析檔案已存至：{output_xlsx}")

if __name__ == "__main__":
    # 獲取目前腳本所在路徑 (轉excel程式)
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    # 專案根目錄
    project_root = os.path.dirname(current_script_dir)

    # 輸入輸出路徑
    bagging_path = os.path.join(project_root, "accuracy_result", "Bagging_Accuracy_Log.csv")
    pso_path = os.path.join(project_root, "accuracy_result", "PSO_Accuracy_Log.csv")
    output_xlsx_path = os.path.join(project_root, "模型分析", "testing正確率.xlsx")

    combine_accuracies_to_excel(bagging_path, pso_path, output_xlsx_path)