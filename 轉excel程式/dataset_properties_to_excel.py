import os
import pandas as pd

def main():
    # 1. 互動式選單：讓使用者選擇要分析的類別
    print("=" * 40)
    print("請選擇要統計的資料集類型：")
    print(" [1] 二類別 (datasets/離散化資料集/二類別)")
    print(" [2] 多類別 (datasets/離散化資料集/多類別)")
    print("=" * 40)
    
    choice = input("請輸入選項 (1 或 2): ").strip()
    
    if choice == "1":
        category_name = "二類別"
    elif choice == "2":
        category_name = "多類別"
    else:
        print("輸入無效，程式結束。")
        return

    # 2. 定位相對路徑 (腳本在 轉excel程式/ 目錄下)
    script_dir = os.path.dirname(os.path.abspath(__file__))      # 轉excel程式 目錄
    root_dir = os.path.dirname(script_dir)                       # 專案根目錄
    
    # 資料集路徑與輸出檔案路徑
    dataset_dir = os.path.join(root_dir, "datasets", "離散化資料集", category_name)
    output_dir = os.path.join(root_dir, "模型分析")
    output_file = os.path.join(output_dir, f"{category_name}_資料集特徵.xlsx")

    # 確保輸出目錄存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if not os.path.exists(dataset_dir):
        print(f"找不到目標資料夾：{dataset_dir}")
        return

    # 3. 讀取該目錄下所有 .csv 檔案並依照名稱排序
    csv_files = sorted([f for f in os.listdir(dataset_dir) if f.endswith(".csv")])
    
    if not csv_files:
        print(f"在 {dataset_dir} 找不到任何 .csv 檔案！")
        return

    print(f"\n正在處理 [{category_name}]，共找到 {len(csv_files)} 個資料集...\n")

    summary_list = []

    # 4. 逐一讀取並計算各資料集特性
    for file in csv_files:
        file_path = os.path.join(dataset_dir, file)
        dataset_name = file.replace(".csv", "")
        
        try:
            df = pd.read_csv(file_path)
            
            # 樣本數（資料筆數）
            sample_count = len(df)
            
            # 檢查是否存在 class 欄位
            if "class" in df.columns:
                # 特徵數 = 總欄位數 - class 欄位
                feature_count = len(df.columns) - 1
                # 類別數 = class 欄位中不重複的數值數量
                class_count = df["class"].nunique()
            else:
                # 若無 class 欄位，預設最後一欄為標籤
                feature_count = len(df.columns) - 1
                class_count = df.iloc[:, -1].nunique()
            
            summary_list.append({
                "資料集名稱": dataset_name,
                "樣本數": sample_count,
                "特徵數": feature_count,
                "類別數": class_count
            })
            print(f"已讀取: {dataset_name:<20} | 樣本數: {sample_count:<6} | 特徵數: {feature_count:<4} | 類別數: {class_count}")

        except Exception as e:
            print(f"讀取 {file} 時發生錯誤: {e}")

    # 5. 轉成 DataFrame 並輸出為 Excel
    df_summary = pd.DataFrame(summary_list)
    
    df_summary.to_excel(output_file, index=False)
    print("-" * 50)
    print(f"處理完成！結果已成功儲存至：\n {output_file}\n")

if __name__ == "__main__":
    main()