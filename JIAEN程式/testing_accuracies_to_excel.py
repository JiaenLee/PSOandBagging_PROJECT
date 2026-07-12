import pandas as pd
import os

# 設定資料來源根目錄
DATASET_BASE_DIR = "../datasets/離散化資料集"

def get_ordered_categories():
    """
    依照 PSO_TRENB.py 的掃描邏輯，生成一個正確的『名稱 -> 類別』對照清單。
    這考慮到了同名檔案出現在不同資料夾的情況。
    """
    ordered_labels = []
    folders = ["二類別", "多類別"]
    
    for folder in folders:
        path = os.path.join(DATASET_BASE_DIR, folder)
        if os.path.exists(path):
            # 取得該資料夾下所有 CSV 並排序 (與主程式邏輯一致)
            files = sorted([f.replace(".csv", "") for f in os.listdir(path) if f.endswith(".csv")])
            for filename in files:
                ordered_labels.append({"Dataset": filename, "類別類型": folder})
                
    return pd.DataFrame(ordered_labels)

def csv_to_vertical_xlsx(input_csv, output_xlsx):
    if not os.path.exists(input_csv):
        print(f"找不到檔案: {input_csv}")
        return

    # 1. 讀取原始橫向 CSV (這裡的 Dataset 名稱順序是關鍵)
    df_raw = pd.read_csv(input_csv)

    # 2. 取得正確的類別標籤順序
    # 因為主程式是按順序 append 進 CSV 的，所以我們按順序給標籤
    df_labels = get_ordered_categories()

    # 3. 處理重複名稱：將標籤與原始數據進行「順序對接」
    # 我們不使用 merge (因為名稱會重複)，我們直接把標籤併進去
    if len(df_raw) == len(df_labels):
        df_raw['類別類型'] = df_labels['類別類型']
    else:
        # 如果行數對不上(可能中途停止)，退而求其次使用檔案檢查
        print("[提示] 數據行數與資料夾檔案數不符，改用內容檢查法...")
        def fallback_check(row):
            # 優先檢查多類別是否存在
            multi_path = os.path.join(DATASET_BASE_DIR, "多類別", f"{row['Dataset']}.csv")
            if os.path.exists(multi_path):
                # 讀取檔案確認類別數
                temp = pd.read_csv(multi_path, usecols=['class'])
                if temp['class'].nunique() > 2:
                    return "多類別"
            return "二類別"
        df_raw['類別類型'] = df_raw.apply(fallback_check, axis=1)

    # 4. 使用 melt 功能將 Fold_1~Fold_5 轉為縱向
    df_vertical = df_raw.melt(
        id_vars=['Dataset', '類別類型'], 
        value_vars=['Fold_1', 'Fold_2', 'Fold_3', 'Fold_4', 'Fold_5'],
        var_name='Fold_Index', 
        value_name='Testing正確率'
    )

    # 5. 整理與排序
    # 順序：二類別 A-Z -> 多類別 A-Z
    df_final = df_vertical[['類別類型', 'Dataset', 'Testing正確率']].copy()
    df_final.columns = ['類別類型', '資料集名稱', 'Testing正確率']
    
    # 這裡的排序要小心，為了維持同一個資料集的 5 折在一起
    df_final = df_final.sort_values(by=['類別類型', '資料集名稱']).reset_index(drop=True)

    # 6. 存成 Excel 檔
    df_final.to_excel(output_xlsx, index=False)
    print("-" * 30)
    print(f"Excel 轉換完成！檔案路徑: {output_xlsx}")

if __name__ == "__main__":
    input_file = "../accuracy_result/PSO_fold_testing_accuracies.csv" 
    folder_name = "模型分析"
    if not os.path.exists(folder_name): os.makedirs(folder_name)
    output_file = os.path.join(folder_name, "PSO_testing_accuracie分析s.xlsx") 

    csv_to_vertical_xlsx(input_file, output_file)