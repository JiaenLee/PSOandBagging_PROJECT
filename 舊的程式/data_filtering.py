import numpy as np
import pandas as pd
import os

def process_filtering(y_train, data_filter_table, m, b, fold):
    """
    執行多類別資料過濾，並將過濾結果寫入 Excel
    Input:
        y_train: 訓練集 y
        data_filter_table: 資料過濾表，shape = (樣本數, 基本模型數, 類別數) (N, B, C)
        m: 集成挑選選擇模型數量
        b: 集成挑選選擇模型數量
    Return:
        指定資料集保留下來的訓練樣本
    """
    # 計算訓練集中總共有多少個類別
    class_nums = len(np.unique(y_train))

    # 計算每個樣本在每個類別得到的總票數
    # data_filter_table 原始形狀 (N, B, C)，對 B (模型維度) 加總
    # 得到的 all_samples_votes 形狀為 (N, C)，代表 N 個樣本在 C 個類別中各得幾票
    all_samples_votes = np.sum(data_filter_table, axis = 1)  # shape(N, C)

    # 用來存儲決定「保留」的樣本索引
    X_reserved = []

    # 用來儲存準備寫入 Excel 的詳細日誌資料
    excel_data_rows = [] 

    # 檔案路徑
    excel_file = 'data_filter_var.xlsx'

    for j in range(len(y_train)):
        true_class = y_train[j]          # 樣本 j 的正確答案（類別索引）
        temp_votes = all_samples_votes[j].copy()    # 複製該樣本得到的票數分佈
        
        # 計算樣本 j 的真實類別票數
        Vyj = temp_votes[true_class]
        
        # 找出「最強的錯誤答案」得票數 (max_Vcf)
        temp_votes[true_class] = -1         # 先把正確類別的票數遮蓋掉（設為 -1）
        max_Vcf = np.max(temp_votes)        # 此時剩餘的最大值就是最強錯誤類別的得票數
        
        # 過濾條件
        is_reserved = True
        
        # 條件 A: Vyj - (m - b) > max_Vcf (正確票數大幅領先，視為冗餘或太簡單的資料)
        # 條件 B: Vyj < b / class_nums (正確票數過低，視為標籤錯誤的雜訊)
        # 條件 C: Vyj + (m - b) < max_Vcf (正確票數即使加了寬限值也輸給錯誤類別，視為重度雜訊)
        if ((Vyj - (m - b) > max_Vcf) or (Vyj < b / class_nums) or (Vyj + (m - b) < max_Vcf)):
            is_reserved = False
        else:
            X_reserved.append(j)
            is_reserved = True

        # 收集每一列的數據
        row_data = [j, Vyj, m, b, max_Vcf, class_nums, true_class, 1 if is_reserved else 0]
        excel_data_rows.append(row_data)

    """
    將資料過濾結果寫入 Excel
    """
    columns = ['Sample', 'Vyj', 'm', 'b', 'Max Vcf', 'class_nums', 'true_class', 'reserved']
    
    # 建立 DataFrame
    result_df = pd.DataFrame(excel_data_rows, columns=columns)
    result_df.set_index('Sample', inplace=True)

    # 如果檔案存在就用 'a'，不存在就用 'w'
    mode = 'a' if os.path.exists(excel_file) else 'w'
    if_sheet_exists = 'replace' if mode == 'a' else None
        
    try:   
        # 寫入檔案
        # 使用 ExcelWriter 寫入指定的折數 (Sheet Name 為 fold_1, fold_2 等)
        with pd.ExcelWriter(excel_file, engine='openpyxl', mode=mode, if_sheet_exists=if_sheet_exists) as writer:
            result_df.to_excel(writer, sheet_name=f"fold_{fold + 1}")
    except Exception as e:
        print(f"Error writing to Excel: {e}")
    
    # 回傳保留下來的樣本索引列表
    return X_reserved