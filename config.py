import os
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
import numpy as np

# 1. ---模型參數設定 ---
MODEL_CONFIG = {
    'k_folds': 5,                  # K 折交叉驗證
    'num_base_models': 25,         # 基本模型總數
    'selection_model_nums': 25,    # 最後挑選的模型數
    'time_limit_per_fold': 600     # 每個 fold 的最大求解時間（秒），超過則停止並輸出當前最佳解
}


# 2. ---PSO 超參數設定 (Hyperparameters)---
PSO_CONFIG = {
    'num_particles': 200,    # 粒子數量
    'max_iter': 400,        # 最大迭代次數
    'w': 0.9,               # 慣性權重 (Inertia Weight)
    'c1': 2.0,              # 個體學習因子 (Cognitive)
    'c2': 2.0,              # 群體學習因子 (Social)
    'bounds': (1e-10, 1.0)  # 參數邊界 (機率值)
}

# 3. ---實驗環境與路徑設定 ---
PARENT_DIR = os.path.dirname(os.path.abspath(__file__))
TRAING_ACCURACY_RESULT_DIR = os.path.join(PARENT_DIR, "training_accuracies")
MODEL_DIR = os.path.join(PARENT_DIR, "temp_models")

PATH = {
    "PSO_TRENB":{
        # 基本模型訓練集正確率儲存路徑
        "training_accuracy_path": os.path.join(TRAING_ACCURACY_RESULT_DIR, "PSO_TRENB.json"),
        # 模型儲存路徑
        "model_path": os.path.join(MODEL_DIR, "PSO_TRENB"),
    },
    "Bagging":{
        # 基本模型訓練集正確率儲存路徑
        "training_accuracy_path": os.path.join(TRAING_ACCURACY_RESULT_DIR, "Bagging.json"),
        # 模型儲存路徑
        "model_path": os.path.join(MODEL_DIR, "Bagging"),
    },
}

# 4. ---資料集設定 ---
DATASET_LIST = ["Abalone_3class",
                "Acoustic",
                "Balance Scale",
                "Car",
                "Choice",
                "DryBean",
                "Ecoli",
                "Glass",
                "Iris",
                "Landsat",
                "Leaf",
                "Maternal Health Risk Data Set",
                # "Modeling",
                "nursery",
                "Performance",
                "Seeds",
                "Student",
                "tae",
                "Vertebral",
                "Wholesale customers data",
                "Winequality-red",
                "Winequality-white",
                "Yeast",
                "Zoo"]

# DATASET_LIST = ["Ecoli"]

# 5. fitness function 設定 ---
class FitnessFunction:
    def cal_micro_accuracy(self, y, pred_y):
        return accuracy_score(y, pred_y)

    def cal_balanced_accuracy(self, y, pred_y):
        return balanced_accuracy_score(y, pred_y)

    def cal_penalty_accuracy(self, y, prior, likelihood, pred_y, penalty_weight=0.01, eps=1e-12):
        """
        加入 entropy penalty 來避免模型過擬合
        """
        acc = accuracy_score(y, pred_y)
        
        # 計算 prior 的 entropy
        prior_safe = np.clip(prior, eps, 1.0)
        prior_entropy = -np.sum(prior_safe * np.log(prior_safe))

        # 計算 likelihood 的 entropy  
        likelihood_entropy_sum = 0.0
        likelihood_rows = 0
        # likelihood 的 entropy 就是把每個特徵的機率 entropy 加總
        for f_prob in likelihood:
            f_prob_safe = np.clip(f_prob, eps, 1.0)
            likelihood_entropy_sum += -np.sum(f_prob_safe * np.log(f_prob_safe))
            likelihood_rows += f_prob_safe.shape[0]  # 要記錄有幾筆特徵值

        # 最後再取平均
        avg_likelihood_entropy = (
            likelihood_entropy_sum / likelihood_rows if likelihood_rows > 0 else 0.0
        )
        penalty = penalty_weight * (prior_entropy + avg_likelihood_entropy)
        # 期望能得到高正確率且低 entropy 的模型
        return acc - penalty

    def cal_macro_avg_accuracy(self, y, pred_y):
        """
        Macro-average accuracy fitness
        """
        # 取得混淆矩陣
        # cm[i][j] 代表：真實是 i 類，預測成 j 類
        cm = confusion_matrix(y, pred_y)
        total_samples = np.sum(cm)
        
        # 計算每個類別的 TP, TN, FP, FN
        TP = np.diag(cm)
        FP = np.sum(cm, axis=0) - TP
        FN = np.sum(cm, axis=1) - TP
        TN = total_samples - (TP + FP + FN)
        
        # 計算每個類別個別的 accuracy_i
        # accuracy_i = (TP_i + TN_i) / (TP_i + TN_i + FP_i + FN_i)
        class_accuracies = (TP + TN) / (TP + TN + FP + FN)
        
        # 計算 Macro-avg Accuracy，直接取平均
        macro_avg_acc = np.mean(class_accuracies)
    
        return macro_avg_acc
    

    def cal_weighted_avg_accuracy_fitness(y, prior, likelihood, pred_y):
        """
        Weighted-average accuracy
        """
        # 取得混淆矩陣
        # cm[i][j] 代表：真實是 i 類，預測成 j 類
        cm = confusion_matrix(y, pred_y)
        total_samples = np.sum(cm)
        
        # 計算每個類別的 TP, TN, FP, FN
        TP = np.diag(cm)
        FP = np.sum(cm, axis=0) - TP
        FN = np.sum(cm, axis=1) - TP
        TN = total_samples - (TP + FP + FN)
        
        # 計算每個類別個別的 accuracy_i
        # accuracy_i = (TP_i + TN_i) / (TP_i + TN_i + FP_i + FN_i)
        class_accuracies = (TP + TN) / (TP + TN + FP + FN)
        
        # 計算 Weighted-avg Accuracy
        weights = TP + FN   # 權重是該類別樣本數
        class_accuracies = (TP + TN) / (TP + TN + FP + FN)
        
        # 使用 np.average 進行加權平均
        weighted_avg_acc = np.average(class_accuracies, weights=weights)

        return weighted_avg_acc
