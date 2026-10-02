# PSO + Bagging for binary and multi-class classification

以 Naive Bayes 作為基礎分類器，結合 Particle Swarm Optimization (PSO) 與 Bootstrap Aggregating (Bagging) 進行二類別與多類別資料集的實驗評估。目的是比較PSO 優化的簡易貝氏模型與 Bagging 的分類效果，並輸出各資料集的交叉驗證結果與模型資訊。

## 專案簡介

- 資料前處理：對原始資料進行等寬離散化，並將目標變數重新編碼成從 0 開始的整數標籤。
- PSO：使用隨機生成的 Naive Bayes 先驗機率與似然機率，透過粒子群優化更新模型參數。
- Bagging：對訓練資料進行重抽樣，訓練多個 Naive Bayes 基本模型，最後以投票方式組成集成預測。
- 多線程處理：使用多線程方式平行處理不同資料集，使多個資料集可以同時進行訓練與實驗，以縮短整體實驗執行時間。
- 實驗輸出：記錄每個 fold 的訓練準確率、測試結果與時間，並將模型與結果儲存到檔案中。

## 目前專案內容

- `config.py`：集中管理實驗參數，例如交叉驗證 fold 數、基本模型數、PSO 超參數與資料集清單。
- `data_preprocessing.py`：將 `datasets/原始資料集` 中的 CSV 檔做等寬離散化，輸出到 `datasets/離散化資料集`。
- `PSO.py`：訓練 PSO 優化的 Naive Bayes 基本模型，並記錄準確率與模型結果；不同資料集可透過多線程平行處理。
- `bagging.py`：訓練 Bagging 的多個 Naive Bayes 基本模型，並輸出交叉驗證結果；不同資料集可透過多線程平行處理。
- `requirements.txt`：Python 套件依賴清單。
- `轉excel程式/`：將產生的結果資料轉成 Excel，方便分析比較。

## 專案結構

```text
PSOandBagging_PROJECT/
├── PSO.py                       # PSO 優化的簡易貝氏模型訓練
├── bagging.py                   # Bagging 集成模型訓練
├── config.py                    # 全域參數設定
├── data_preprocessing.py        # 等寬離散化處理
├── requirements.txt             # Python 套件需求
├── README.md                    # 專案說明
├── 轉excel程式/
└──.gitignore
```

## 執行方式

### 1. 建立環境

建議使用 Python 3.10+，並安裝依賴：

```bash
pip install -r requirements.txt
```

### 2. 執行資料前處理

```bash
python data_preprocessing.py
```

這會將原始資料轉成離散化資料，並存放在 `datasets/離散化資料集/`。

### 3. 執行 PSO 訓練

```bash
python PSO.py
```
程式會使用多線程方式平行處理不同資料集，以提升整體實驗執行效率。

### 4. 執行 Bagging 訓練

```bash
python bagging.py
```
程式會使用多線程方式平行處理不同資料集，以提升整體實驗執行效率。

## 輸出說明
- `accuracy_result/`
  - 主要實驗結果 CSV
  - 每一列對應一個資料集與其各 fold 的測試表現
- `temp_models/`
  - 每個 fold 訓練出的基本模型資訊
- `training_accuracies/`
  - 每個資料集在各模型訓練過程中的 accuracy 記錄
- `output_PSO.txt` / `output_bagging.txt`
  - 程式執行進度與狀態

