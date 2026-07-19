import pandas as pd
from scipy import stats

def analyze_matched_samples(file_path):
    # 1. 讀取 Excel 檔案
    df = pd.read_excel(file_path)
    
    # 檢查必要的欄位是否存在
    required_cols = ['類別類型', 'Bagging正確率', 'PSO正確率']
    if not all(col in df.columns for col in required_cols):
        print(f"錯誤：Excel 檔案必須包含以下欄位：{required_cols}")
        return

    # 2. 找出所有不同的類別類型（例如：二類別、多類別）
    categories = df['類別類型'].unique()
    
    print("==================================================")
    print("       Matched Sample Design 統計分析結果")
    print("==================================================")
    
    for cat in categories:
        print(f"\n▶ 正在分析【{cat}】資料...")
        
        # 篩選出該類別的資料
        sub_df = df[df['類別類型'] == cat].dropna(subset=['Bagging正確率', 'PSO正確率'])
        
        n = len(sub_df)
        if n < 2:
            print(f" ⚠️ 樣本數不足（N={n}），無法進行統計檢定。")
            continue
            
        bagging_scores = sub_df['Bagging正確率']
        pso_scores = sub_df['PSO正確率']
        
        # 計算基本敘述統計
        mean_bagging = bagging_scores.mean()
        mean_pso = pso_scores.mean()
        mean_diff = (pso_scores - bagging_scores).mean()
        
        print(f"  - 樣本數 (N): {n}")
        print(f"  - Bagging 平均正確率: {mean_bagging:.4f}")
        print(f"  - PSO 平均正確率: {mean_pso:.4f}")
        print(f"  - 平均差異 (PSO - Bagging): {mean_diff:.4f}")
        
        # 3. 常態性檢定 (Shapiro-Wilk Test) - 檢定兩者差異是否符合常態分佈
        diff = pso_scores - bagging_scores
        # Shapiro-Wilk 限制樣本數在 3 到 5000 之間
        if 3 <= n <= 5000:
            _, shapiro_p = stats.shapiro(diff)
            is_normal = shapiro_p > 0.05
        else:
            is_normal = True # 樣本數過大或過小時的預設彈性處理
            shapiro_p = float('nan')

        # 4. 根據常態性選擇檢定方法
        if is_normal:
            # 符合常態分佈：使用 Paired t-test
            t_stat, p_val = stats.ttest_rel(pso_scores, bagging_scores)
            test_method = "配對樣本 t 檢定 (Paired t-test) [符合常態分佈]"
        else:
            # 不符合常態分佈：使用 Wilcoxon Signed-Rank Test
            t_stat, p_val = stats.wilcoxon(pso_scores, bagging_scores)
            test_method = "威爾考克森符號秩檢定 (Wilcoxon Signed-Rank Test) [不符合常態分佈]"
            
        print(f"  - 採用的統計方法: {test_method}")
        print(f"  - 統計量 (Statistic): {t_stat:.4f}")
        print(f"  - p 值 (p-value): {p_val:.6e}")
        
        # 5. 結論判斷
        if p_val < 0.05:
            print("  - 🎉 結論: 在顯著水準 α = 0.05 下，兩種演算法的正確率有**顯著差異**。")
            if mean_pso > mean_bagging:
                print("    👉 PSO 的表現顯著優於 Bagging。")
            else:
                print("    👉 Bagging 的表現顯著優於 PSO。")
        else:
            print("  - ⚖️ 結論: 在顯著水準 α = 0.05 下，兩種演算法的正確率**沒有顯著差異**。")
            
    print("\n==================================================")

analyze_matched_samples('PSO&Bagging_training_accuracies_mixed.xlsx')