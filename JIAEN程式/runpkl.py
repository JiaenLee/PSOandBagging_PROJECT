#我只是在看PKL裡面到底是什麼

import pickle
import numpy as np

# 1. 載入數據
with open('temp_models/PSO_TRENB/Abalone_3class_models.pkl', 'rb') as f:
    data = pickle.load(f)

# 2. 檢查第一層：有哪些折數 (Folds)
print("檔案中的折數:", data.keys()) # 預期輸出: dict_keys(['fold_1', 'fold_2', 'fold_3', 'fold_4', 'fold_5'])

# 3. 檢查某一折的模型數量
fold1_models = data['fold_2']['PSO_TRENB']
print(f"Fold 2 中的模型數量: {len(fold1_models)}") # 預期輸出: 25

# 4. 檢查第一個基本模型的參數 (Prior 和 Likelihood)
first_model = fold1_models[0]
prior, likelihood = first_model

print("\n--- 第一個模型的先驗機率 (Prior) ---")
print(prior) 

print("\n--- 第一個模型的第二個特徵的似然機率 (Likelihood of Feature 0) ---")
print("\n--- 每一列代表不同類別, 機率是該類別在該特徵值為0,1,2...的機率")
print(likelihood[1])

# 1. 檢查 likelihood 列表的長度（這代表特徵的總數）
print(f"模型包含的特徵總數: {len(likelihood)}") 

# 2. 依序印出每個特徵矩陣的形狀
for i, l_matrix in enumerate(likelihood):
    print(f"特徵 {i} 的矩陣形狀: {l_matrix.shape} (類別數 x 該特徵的值數量)")