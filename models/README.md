# Models Directory

此目錄儲存訓練完成的手術時間預測模型檔案。

## 目錄結構

```
models/
├── README.md                       # 本說明檔案
├── ENT_random_forest_model.pkl     # ENT(耳鼻喉科)模型
├── GS_random_forest_model.pkl      # GS(一般外科)模型  
├── GU_random_forest_model.pkl      # GU(泌尿外科)模型
├── OPH_random_forest_model.pkl     # OPH(眼科)模型
└── ORTH_random_forest_model.pkl    # ORTH(骨科)模型
```

## 模型規格

### 演算法
- **模型類型**: Random Forest Regressor
- **實作框架**: scikit-learn
- **前處理**: StandardScaler (數值特徵) + OneHotEncoder (類別特徵)
- **序列化**: joblib pickle format

### 最佳化參數

基於 GridSearchCV 5-fold 交叉驗證結果：

| 科別 | 樹數量 | 最大深度 | 最小分裂樣本 | 最小葉節點 | 特徵選擇 |
|------|--------|----------|--------------|------------|----------|
| ENT | 300 | None | 10 | 1 | 1.0 |
| GS | 500 | 20 | 15 | 1 | log2 |
| GU | 288 | None | 17 | 1 | sqrt |
| OPH | 200 | 15 | 12 | 3 | sqrt |
| ORTH | 350 | None | 20 | 1 | sqrt |

### 效能指標

| 科別 | 訓練集 MSE | 測試集 MAE | R² Score | 模型狀態 |
|------|------------|------------|----------|----------|
| ENT | 3,063 | 34.8 | 0.84 | Production Ready |
| GS | 2,580 | 32.8 | 0.73 | Production Ready |
| GU | 1,923 | 40.5 | 0.87 | Production Ready |
| OPH | 441 | 12.8 | 0.68 | Production Ready |
| ORTH | 3,232 | 31.5 | 0.72 | Production Ready |

## 使用方式

### 載入模型進行預測

```python
from src.models.predictor import SurgeryPredictor

# 方法一: 直接指定模型路徑
predictor = SurgeryPredictor('ENT', 'models/ENT_random_forest_model.pkl')

# 方法二: 使用便捷方法
predictor = SurgeryPredictor.from_department('ENT', 'models')

# 單筆預測
patient_data = {
    '年齡(G)': 45,
    '性別(H)': 'M',
    '身份(I)': 'A',
    '麻醉(K)': 'General',
    # ... 其他必要特徵
}
result = predictor.predict_single(patient_data)
```

### 批次載入所有科別模型

```python
from src.config.departments import DEPARTMENT_CONFIGS

predictors = {}
for dept in DEPARTMENT_CONFIGS.keys():
    try:
        predictors[dept] = SurgeryPredictor.from_department(dept, 'models')
        print(f"{dept} 模型載入成功")
    except FileNotFoundError:
        print(f"{dept} 模型檔案不存在")
```

## 重要事項

### 資料保密
- 模型檔案包含從訓練資料學習的參數
- 基於保密協定，模型檔案**不會**上傳至 Git repository
- 使用 `.gitignore` 確保模型檔案僅存在於本地環境

### 模型訓練
重新訓練模型：

```bash
# 訓練所有科別模型
python scripts/train_all_departments.py

# 訓練特定科別（含超參數調整）
python scripts/tune_hyperparameters.py --department ENT
```

### 模型評估
驗證模型效能：

```bash
python scripts/evaluate_models.py
```

## 檔案格式說明

### 模型檔案內容
每個 `.pkl` 檔案包含完整的 scikit-learn Pipeline：
```
Pipeline([
    ('preprocessor', ColumnTransformer([
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(), categorical_features)
    ])),
    ('regressor', RandomForestRegressor(**optimized_params))
])
```

### 相依性要求
- Python >= 3.8
- scikit-learn >= 1.2.0
- pandas >= 1.5.0
- joblib >= 1.2.0

## 模型維護

### 版本管理
- 模型版本透過檔案修改時間追蹤
- 建議定期重新訓練以維持預測準確度
- 重大更新時保留舊版本作為備份

### 效能監控
建議監控指標：
- 預測準確度是否下降
- 新資料分布是否偏移
- 特徵重要性變化

### 更新策略
1. 定期評估模型效能
2. 當 R² 下降超過 0.05 時考慮重訓
3. 新增資料時進行增量學習評估

## 故障排除

### 常見問題

**模型載入失敗**
- 檢查檔案路徑是否正確
- 確認 Python 環境與訓練環境一致
- 驗證 scikit-learn 版本相容性

**預測結果異常**
- 確認輸入特徵格式與訓練時一致
- 檢查缺失值處理是否正確
- 驗證類別特徵值是否在訓練集範圍內

**記憶體使用過高**
- 考慮減少 `n_estimators` 參數
- 使用 `n_jobs=1` 限制並行處理
- 批次處理大量預測請求

## 聯絡資訊

如有模型相關技術問題，請參考：
- 專案文檔: `docs/api_reference.md`
- 程式碼範例: `notebooks/` 目錄
- 單元測試: `tests/test_models.py`
