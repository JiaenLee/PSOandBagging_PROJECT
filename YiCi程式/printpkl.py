import pickle
import pandas as pd
import numpy as np
import os

# 1. 設定路徑
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
pkl_path = os.path.join(project_root, 'temp_models', 'PSO_TRENB', 'Abalone_3class_models.pkl')

folder_name = os.path.join(current_dir, "模型分析")
output_excel = os.path.join(folder_name, "PSO_Abalone_3class_models_模型參數分析.xlsx")

# 檢查資料夾
if not os.path.exists(folder_name):
    os.makedirs(folder_name)

# 2. 載入 pkl 檔案
try:
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
except FileNotFoundError:
    print(f"錯誤：找不到檔案 {pkl_path}")
    exit()

# 3. 解析資料並整理成表格
rows = []

for fold_name, fold_content in data.items():
    # 取得該折的所有模型 (通常是 50 個)
    models_list = fold_content.get('PSO_TRENB', [])
    
    for m_idx, (prior, likelihood) in enumerate(models_list):
        # 基礎資訊
        model_info = {
            "Fold": fold_name,
            "Model_ID": m_idx + 1
        }
        
        # --- 處理先驗機率 (Prior) ---
        # 例如 [1.e-10, 1.e-10, 1.e+00]
        for class_idx, prob in enumerate(prior):
            model_info[f"Prior_類別_{class_idx}"] = prob
            
        # --- 處理似然機率 (Likelihood) ---
        # 因為似然機率是 8 個矩陣，我們存入「特徵 0」的摘要作為參考
        # 或者你可以選擇將所有特徵的矩陣轉為字串存入一個欄位
        for feat_idx, l_matrix in enumerate(likelihood):
            # 將矩陣轉為字串，方便在 Excel 儲存格內查看內容
            matrix_str = str(np.round(l_matrix, 4)) 
            model_info[f"特徵_{feat_idx}_似然矩陣"] = matrix_str
            
        rows.append(model_info)

# 4. 建立 DataFrame 並存入 Excel
df = pd.DataFrame(rows)

# 重新排列欄位順序 (讓 Fold 和 Model_ID 在最前面)
cols = ['Fold', 'Model_ID'] + [c for c in df.columns if c not in ['Fold', 'Model_ID']]
df = df[cols]

df.to_excel(output_excel, index=False)

print("-" * 30)
print(f"模型參數分析完成！")
print(f"檔案已儲存：{output_excel}")