"""專案設定檔"""

# 資料設定
DATA_CONFIG = {
    'encoding': 'big5',
    'target_column': '手術時間（分）(BQ)',
    'random_state': 42
}

# 特徵設定
FEATURE_CONFIG = {
    'numeric_features': [
        '年齡(G)', 
        '醫師年資（月）(AG)', 
        '手術數量(AQ)', 
        '醫師人數(BG)'
    ],
    'categorical_features': [
        '性別(H)', '身份(I)', '分類(J)', '麻醉(K)', 
        '手術名稱(L)', '主治醫師(AF)', '分類(AZ)'
    ]
}

# 模型設定
MODEL_CONFIG = {
    'random_forest': {
        'default_params': {
            'random_state': 42,
            'n_jobs': -1
        },
        'param_grid': {
            'n_estimators': [100, 200, 300, 400, 500],
            'max_depth': [10, 20, 30, None],
            'min_samples_split': [2, 5, 10, 15, 20],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', 1.0],
            'bootstrap': [True, False]
        }
    }
}

# 交叉驗證設定
CV_CONFIG = {
    'cv_folds': 5,
    'scoring': 'neg_mean_squared_error',
    'n_iter': 100,
    'verbose': 2
}
