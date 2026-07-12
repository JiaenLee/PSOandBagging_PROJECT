import pandas as pd
import os

# 設定資料來源根目錄 (用來找 CSV 檔案判斷類別)
DATASET_BASE_DIR = "../datasets/離散化資料集"

def determine_real_category(dataset_name):
    """
    根據資料集名稱，去硬碟讀取原始 CSV 並判斷是二類別還是多類別
    """
    check_paths = [
        os.path.join(DATASET_BASE_DIR, "二類別", f"{dataset_name}.csv"),
        os.path.join(DATASET_BASE_DIR, "多類別", f"{dataset_name}.csv")
    ]
    
    for path in check_paths:
        if os.path.exists(path):
            try:
                # 只讀取 class 欄位來節省時間
                temp_df = pd.read_csv(path, usecols=['class'])
                class_count = temp_df['class'].nunique()
                return "二類別" if class_count == 2 else "多類別"
            except Exception:
                continue
    return "未知"

def csv_to_vertical_xlsx(input_csv, output_xlsx):
    """
    讀取橫向 5-Fold CSV，旋轉為縱向，增加類別判斷後存為 XLSX
    """
    if not os.path.exists(input_csv):
        print(f"找不到檔案: {input_csv}")
        return

    # 1. 讀取原始橫向 CSV
    df = pd.read_csv(input_csv)

    # 2. 使用 melt 功能將 Fold_1~Fold_5 轉為縱向
    df_vertical = df.melt(
        id_vars=['Dataset'], 
        value_vars=['Fold_1', 'Fold_2', 'Fold_3', 'Fold_4', 'Fold_5'],
        var_name='Fold_Index', 
        value_name='Testing正確率'
    )

    # 3. 排序，讓同一個資料集的五個正確率排在一起
    df_vertical = df_vertical.sort_values(by=['Dataset', 'Fold_Index'])

    # 4. 增加「類別類型」欄位 (精準判斷)
    print("正在根據資料集內容精準判斷類別類型...")
    df_vertical['類別類型'] = df_vertical['Dataset'].apply(determine_real_category)

    # 5. 整理欄位順序與名稱
    # 格式：類別類型 | 資料集名稱 | Testing正確率
    df_final = df_vertical[['類別類型', 'Dataset', 'Testing正確率']].copy()
    df_final.columns = ['類別類型', '資料集名稱', 'Testing正確率']

    # 6. 最後根據類別和名稱排序，讓 Excel 看起來更整齊
    df_final = df_final.sort_values(by=['類別類型', '資料集名稱']).reset_index(drop=True)

    # 7. 存成 Excel 檔
    df_final.to_excel(output_xlsx, index=False)
    print("-" * 30)
    print(f"Excel 轉換完成！檔案路徑: {output_xlsx}")
    print(f"目前分類統計：\n{df_final['類別類型'].value_counts() // 5} (每個資料集有5折)")

if __name__ == "__main__":
    # 使用 ../ 代表回到上一層資料夾，再進入 accuracy_result
    input_file = "../accuracy_result/PSO_fold_testing_accuracies.csv" 
    
    # 輸出路徑
    folder_name = "模型分析"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        
    output_file = os.path.join(folder_name, "PSO_fold_testing_accuracies.xlsx") 

    csv_to_vertical_xlsx(input_file, output_file)