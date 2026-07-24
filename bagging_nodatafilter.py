import pickle
import random, re
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
import os, time, json, csv
from pathlib import Path
from tqdm import tqdm
import config

def naive_bayes_classifier(df_train, attributes, class_nums):
    """
    使用訓練資料集建構簡易貝氏分類器，attributes: 特徵集合
    Return:
        prior: 先驗機率
        p_Xi_Cj_dict: 條件機率
    """
    all_classes = np.arange(class_nums)  # 將類別數轉為列表
    nj_counts = df_train['class'].value_counts()
    # 缺失類別用 0 補齊
    nj = nj_counts.reindex(all_classes, fill_value=0) #統計訓練集中每一個類別出現的次數。

    prior = nj / len(df_train)    # prior：計算 P(Cj)，即「類別 j 出現的機率 = 類別 j 的次數 / 總樣本數」。
    att_value_counts = {x: df_train[x].nunique() for x in attributes}  # 計算每個特徵（屬性）有多少個不同的值（例如：特徵「天氣」有 3 種值：晴天、陰天、雨天)
    
    p_Xi_Cj_dict = {}
    # 針對每個特徵 x
    for att in attributes:
        # 依照 (att, class) 分組，並計算每個組合的出現次數，最後轉置成 (att, class) 的 DataFrame
        '''
        nij：產出一個表格。
            橫軸（Index）：特徵的所有可能值。
            縱軸（Columns）：各個類別。
            內容：該類別下，該特徵值出現了幾次。
        '''
        nij = df_train.groupby([att, 'class']).size().unstack(fill_value=0)  # p(xi | cj)
        # 補上這行：強制補齊所有類別欄位 (0 到 class_nums-1)，確保結構完整不缺漏
        nij = nij.reindex(columns=np.arange(class_nums), fill_value=0)
        #P(Xi​∣Cj​)=(出現次數+1​)/(該類別總數+該特徵總值數)
        p_Xi_Cj_dict[att] = (nij + 1) / (nj + att_value_counts[att])  # Laplace estimator (拉普拉斯平滑)
    # p_Xi_Cj_dict 的格式為 {特徵名稱: 儲存特徵類別組合機率的 DataFrame}
    """
        prior：先驗機率（每個類別的基本勝率）。
        p_Xi_Cj_dict：似然機率字典。Key 是特徵名，Value 是一個 DataFrame，存著「在某類別下出現某特徵值」的平滑機率。
        att_value_counts：每個特徵的獨特值數量（預測時若遇到新值會用到）
    """
    return prior, p_Xi_Cj_dict, att_value_counts
    
#利用訓練階段算好的機率資料，去判斷新樣本最可能屬於哪一個類別
def predict(nj, df_predict, prior, p_Xi_Cj_dict, att_value_counts):

    """
    Parmas:
        nj : 類別數量 (指的是被抽樣後的子訓練集的類別數量)
    Return:
        1. 樣本的預測向量，shape = (N, C)，裡面每個元素為 1d array，裡面有 C 個元素，代表 C 個類別的預測與否，1/0
        2. 所有樣本實際預測類別，shape = (N, )
    """ 
    pred_class = []
    pred_vector = []
    # 預測每個樣本:一個一個處理要預測的樣本。
    for i in range(len(df_predict)):
        attributes = df_predict.iloc[i, :-1]   # ：取出目前這個樣本的所有特徵值
        max_prob = -np.inf
        pred_class_i = None   # 樣本 i 的預測類別
        vector_i = np.zeros_like(prior, dtype=int)   # 樣本 i 的預測向量

        # 針對每個樣本，程式會測試所有可能的類別 Cj，看哪一個類別算出來的分數（後驗機率）最高
        for Cj in nj.index:
            class_condition = 1  # class condition 就是類別下特徵組合機率
            # 計算每個特徵值在樣本 i 的出現機率
            for att in attributes.index:
                if pd.notna(df_predict[att][i]):  # 忽略NaN機率值
                    try:
                        # 正常查表：取出該類別下，該特徵值的機率
                        # 取出特徵 att 的 likelihood DataFrame，再取出 Cj 欄位下，att 值的對應機率
                        class_condition *= p_Xi_Cj_dict[att][Cj][attributes[att]]
                    except KeyError:
                         # 沒看過的值：如果在訓練集中沒出現過這個特徵值，使用拉普拉斯平滑
                        class_condition *= 1 / (nj[Cj] + att_value_counts[att]) 
            # 後驗機率
            posterior_prob = prior[Cj] * class_condition
           
            if posterior_prob > max_prob:
                pred_class_i = Cj
                max_prob = posterior_prob
        
        vector_i[pred_class_i] = 1
        pred_vector.append(vector_i)
        pred_class.append(pred_class_i)
    
    return np.array(pred_vector), np.array(pred_class)

def cross_validation_with_ensemble(file_path, target_column, model_config, dataset_name):
    """
    進行五折交叉驗證訓練，以下每個步驟都是在每一折內各自進行
    1. 每一折訓練產生 25 個基本模型 (Bagging)
    2. 每個基本模型預測每個訓練樣本建構資料過濾表
    3. 進行資料過濾
    4. 使用過濾後資料集進行最佳化集成挑選
    5. 進行集成，預測測試集樣本
    """
    # 從 model_config 中解析模型參數
    k = model_config['k_folds']
    num_base_models = model_config['num_base_models']

    data = pd.read_csv(file_path)
    attributes = data.columns[:-1]  # 特徵名稱集合
    X = data.drop(columns = [target_column]).values
    y_raw = data[target_column].values  # 取得原始標籤

    # === 對類別標籤 y 做 LabelEncoder，解決索引問題 ===
    le_y = LabelEncoder()
    y = le_y.fit_transform(y_raw)
    class_nums = len(le_y.classes_)

   # 對 X 別做 LabelEncode，防止空箱問題產生
    for i in range(X.shape[1]): # 針對每個特徵
        le = LabelEncoder()
        X[:, i] = le.fit_transform(X[:, i]) 

    start_time = time.time()  # 計時開始

    # 分割訓練與測試
    kf = KFold(n_splits=k, shuffle=True, random_state=42) 

    training_accuracies = []    # 儲存每個 fold 基本模型對 training set 的預測準確率
    test_results_with_counts = []  # 儲存每個折數的測試結果資訊

    fold_models = {}  # 儲存每個折數的模型 

    # 進行五折交叉驗證訓練
    for fold, (train_index, test_index) in enumerate(kf.split(X)):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        train_data = pd.DataFrame(X_train, columns=attributes)
        train_data['class'] = y_train
        test_data = pd.DataFrame(X_test, columns=attributes)
        test_data['class'] = y_test

        # fold_data 儲存模型資訊
        fold_data = {
            "test_index": test_index.tolist(),
            "train_class": y_train.tolist(),
            "class_nums": class_nums,
        }
        bagging_models = []  # 記錄訓練出的模型

        # 計算訓練集樣本數
        N = len(X_train)
        # 每個基本模型對測試集樣本的預測，單一折數中基本模型的預測結果，(樣本數, 模型數)
        model_test_predictions = np.zeros((len(test_data), num_base_models), dtype=int)   

        # 集成 num_base_models 個 base models 的預測結果  
        nums = 0
        with tqdm(total = num_base_models, desc = f"Fold {fold + 1} - Building Models") as pbar:
            while nums < num_base_models:
                # 生成 0 ~ N-1 範圍內的隨機亂數，總共生成 N 個
                # 這步驟代表 bagging 的取後放回抽樣，陣列裡的每個元素即為抽到的訓練集樣本索引
                sampled_indices = random.choices(range(N), k = N)
                bag_train_data = train_data.iloc[sampled_indices]
                nj = bag_train_data['class'].value_counts()

                # 計算訓練資料集的先驗機率、似然機率
                prior, p_Xi_Cj_dict, att_value_counts = naive_bayes_classifier(bag_train_data, attributes, class_nums)
                bagging_models.append((prior, p_Xi_Cj_dict, att_value_counts, nj))  # nj 為子訓練集的類別數量

                # 對原始訓練集的預測
                pred_vector, pred_class = predict(nj, train_data, prior, p_Xi_Cj_dict, att_value_counts)
                training_accuracies.append(np.mean(y_train == pred_class))  # 記錄訓練集準確率
                
                # 測試集預測
                pred_vector, pred_class = predict(nj, test_data, prior, p_Xi_Cj_dict, att_value_counts)
                model_test_predictions[:,nums] = pred_class  # 基本模型 i 對所有測試樣本的預測結果

                nums += 1
                pbar.update(1)

        fold_data["Bagging"] = bagging_models  # 記錄所有 Bagging 基本模型資訊

        # 保存該折模型與樣本資訊
        fold_models[f"fold_{fold + 1}"] = fold_data
                
        # 集成所有基本模型的預測成為最終預測 (投票)
        ensemble_test_predictions = np.apply_along_axis(
            lambda x : np.bincount(x.astype(int), minlength = class_nums).argmax(),
            axis = 1, arr = model_test_predictions
        )

        test_accuracy = np.mean(y_test == ensemble_test_predictions)
        # 格式化為 "準確率:資料筆數"
        test_results_with_counts.append(f"{test_accuracy:.4f}:{len(test_data)}")

    end_time = time.time()
    exec_time = end_time - start_time  # 訓練結束，計算總時間

    path = config.PATH.get("Bagging")

    # 將所有預測資訊寫入
    write_json_data(path["training_accuracy_path"], dataset_name, training_accuracies)  # 將五折交叉驗證中的訓練集樣本預測正確率寫入
   
    # 保存模型至文件
    output_path = os.path.join(path["model_path"], f"{dataset_name}_models.pkl")
    # 確保 model 檔案的父資料夾存在
    model_folder = os.path.dirname(output_path)
    if model_folder and not os.path.exists(model_folder):
        os.makedirs(model_folder, exist_ok=True)

    with open(output_path, 'wb') as f:
        pickle.dump(fold_models, f)

    print(f"模型已保存至 {output_path}")

    return training_accuracies, test_results_with_counts, exec_time

# 寫入 json 檔案，並壓縮內層
def write_json_data(path, dataset_name, content):

    # ===== 1️⃣ 建立資料夾（如果不存在）=====
    folder = os.path.dirname(path)
    if folder and not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)

    # ===== 2️⃣ 如果檔案不存在，先建立空 JSON =====
    if not os.path.exists(path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump({}, f)

    # ===== 3️⃣ 讀取 JSON（避免空檔壞掉）=====
    with open(path, "r", encoding="utf-8") as r:
        try:
            json_data = json.load(r)
        except json.JSONDecodeError:
            json_data = {}

    # ===== 4️⃣ 更新內容 =====
    json_data[dataset_name] = content

    # ===== 5️⃣ 轉成字串並壓縮內層格式 =====
    json_str = json.dumps(json_data, ensure_ascii=False, indent=3)

    json_str = re.sub(
        r'\[\s*([0-9\.\,\s\-]+?)\s*\]',
        lambda m: '[' + ', '.join([x.strip() for x in m.group(1).split(',')]) + ']',
        json_str
    )

    json_str = re.sub(
        r'(\[\s*(?:\[[0-9\.\,\s\-]+\]\s*,?\s*)+\])',
        lambda m: re.sub(r'\s+', ' ', m.group(1))
        .replace(' [', '[')
        .replace('] ]', ']]'),
        json_str
    )

    # ===== 6️⃣ 寫回檔案 =====
    with open(path, "w", encoding="utf-8") as f:
        f.write(json_str)

if __name__ == "__main__":
    random.seed(42)
    model_config = config.MODEL_CONFIG
    path = config.PATH.get("Bagging")
    
    # 設定輸出資料夾為 accuracy_result 
    output_dir = "accuracy_result"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # 重新定義主要 Log 檔案路徑與標頭
    main_log_csv = os.path.join(output_dir, "Bagging_Accuracy_Log.csv")
    with open(main_log_csv, mode='w', encoding='utf-8', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Dataset", "Fold_1:num", "Fold_2:num", "Fold_3:num", "Fold_4:num", "Fold_5:num", "Total_Time"])

    data_folders = [
        "datasets/離散化資料集/二類別",  
        "datasets/離散化資料集/多類別"   
    ]
    
    for data_folder in data_folders:
        folder_label = os.path.basename(data_folder)
        print(f"\n================ 正在掃描資料夾: {folder_label} ================")
        
        dataset_list = sorted([
            f.replace(".csv", "")
            for f in os.listdir(data_folder)
            if f.endswith(".csv")
        ])

        for filename in dataset_list: 
            unique_name = f"{folder_label}_{filename}"
            print(f"處理資料集: {unique_name}")
            file_path = os.path.join(data_folder, filename + '.csv')
            target_column = "class" 

            res = cross_validation_with_ensemble(file_path, target_column, model_config, unique_name)
            training_acc_list, test_info_list, exec_time = res

            # 將結果寫入 accuracy_result 下的 CSV
            with open(main_log_csv, mode='a', encoding='utf-8', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([unique_name] + test_info_list + [exec_time])
