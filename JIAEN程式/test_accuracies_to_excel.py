import pandas as pd
import os

def csv_to_vertical_xlsx(input_csv, output_xlsx):
    """
    讀取橫向 5-Fold CSV，旋轉為縱向後存為 XLSX
    """
    if not os.path.exists(input_csv):
        print(f"找不到檔案: {input_csv}")
        return

    # 1. 讀取 CSV
    df = pd.read_csv(input_csv)

    # 2. 使用 melt 功能將 Fold_1~Fold_5 轉為縱向 (正確率一欄)
    # id_vars 是固定不動的欄位，value_vars 是要轉成縱向的欄位
    df_vertical = df.melt(
        id_vars=['Dataset'], 
        value_vars=['Fold_1', 'Fold_2', 'Fold_3', 'Fold_4', 'Fold_5'],
        var_name='Fold_Index', 
        value_name='Testing正確率'
    )

    # 3. 排序，讓同一個資料集的五個正確率排在一起
    df_vertical = df_vertical.sort_values(by=['Dataset', 'Fold_Index'])

    # 4. 只保留「資料集」與「正確率」兩欄 (去掉 Fold_Index)
    df_final = df_vertical[['Dataset', '正確率']]

    # 5. 存成 Excel 檔 (需要 openpyxl 套件)
    df_final.to_excel(output_xlsx, index=False)
    print(f"Excel 轉換完成！檔案路徑: {output_xlsx}")

if __name__ == "__main__":
    # 使用 ../ 代表回到上一層資料夾，再進入 accuracy_result
    input_file = "../accuracy_result/PSO_fold_testing_accuracies.csv" 
    
    # 輸出檔案：因為你現在就在 JIAEN程式 資料夾裡，直接寫子資料夾名稱即可
    output_file = "模型分析/PSO_fold_testing_vertical.xlsx" 

    csv_to_vertical_xlsx(input_file, output_file)