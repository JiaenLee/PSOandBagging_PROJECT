import pickle

# 1. 打開 pkl 檔案
with open('../temp_models/PSO_TRENB/Abalone_3class_models.pkl', 'rb') as f:
    # 2. 載入內容
    data = pickle.load(f)

# 3. 顯示資料
print(data)