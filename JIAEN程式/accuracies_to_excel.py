import pandas as pd
import json
import os

# 1. 設定路徑與檔名
folder_name = "模型分析"
output_filename = "PS0_模型正確率報告.xlsx"
output_path = os.path.join(folder_name, output_filename)
json_file_path = 'training_accuracies/PSO_TRENB.json' # 假設您的原始檔案叫這個名字

# 2. 檢查資料夾是否存在，若否則建立
if not os.path.exists(folder_name):
    os.makedirs(folder_name)
    print(f"已建立資料夾：{folder_name}")
else:
    print(f"資料夾「{folder_name}」已存在，直接準備儲存。")

# 3. 讀取數據 (這裡使用 try-except 確保檔案存在)
try:
    with open(json_file_path, 'r', encoding='utf-8') as f:
        json_data = json.load(f)
except FileNotFoundError:
    print(f"錯誤：找不到 {json_file_path}，請確認檔案放在相同路徑下。")
    # 如果找不到檔案，這裡放一個範例數據方便測試
    json_data = {"Abalone_3class": [0.6064, 0.6096, 0.6004]} 

# 4. 整理數據格式
rows = []
for dataset_name, accuracies in json_data.items():
    for acc in accuracies:
        rows.append({
            "資料集名稱": dataset_name,  # 第一欄
            "正確率": acc               # 第二欄
        })

# 5. 轉換為 DataFrame
df = pd.DataFrame(rows)

# 6. 輸出成 Excel 檔
# 使用 index=False 確保不會輸出左側的 0, 1, 2 序號欄位
df.to_excel(output_path, index=False)

print("-" * 30)
print(f"處理完成！")
print(f"檔案路徑：{os.path.abspath(output_path)}")
print(f"共寫入 {len(df)} 筆模型數據。")