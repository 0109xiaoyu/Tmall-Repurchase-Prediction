import pandas as pd
import numpy as np
import lightgbm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, f1_score, precision_score, confusion_matrix, classification_report
import joblib

train = pd.read_csv('train_clean.csv')
test = pd.read_csv('test_clean.csv')

features = [col for col in train.columns if col not in ['user_id', 'merchant_id', 'label']]
X = train[features]
y = train['label']
groups = train['user_id']

# 下采样处理不平衡
n = sum(y == 1)
pos_idx = y[y == 0].sample(n, random_state=123).index
neg_idx = y[y == 1].index
X_balanced = X.loc[pos_idx.union(neg_idx)]
y_balanced = y.loc[pos_idx.union(neg_idx)]
groups_balanced = groups.loc[pos_idx.union(neg_idx)]

gkf = GroupKFold(n_splits=5)
auc_scores = []
f1_scores = []
best_model = None

print("【开始 GroupKFold 交叉验证...】")
for fold, (train_idx, val_idx) in enumerate(gkf.split(X_balanced, y_balanced, groups_balanced)):
    X_train, X_val = X_balanced.iloc[train_idx], X_balanced.iloc[val_idx]
    y_train, y_val = y_balanced.iloc[train_idx], y_balanced.iloc[val_idx]

    sc = StandardScaler()
    X_train_scaled = sc.fit_transform(X_train)
    X_val_scaled = sc.transform(X_val)

    model = lightgbm.LGBMClassifier(
        n_estimators=1000, max_depth=8, num_leaves=25,
        colsample_bytree=0.5, learning_rate=0.1, metric='auc',
        random_state=42
    )
    model.fit(
        X_train_scaled, y_train,
        eval_metric='auc',
        eval_set=[(X_train_scaled, y_train), (X_val_scaled, y_val)],
        callbacks=[lightgbm.early_stopping(stopping_rounds=100), lightgbm.log_evaluation(period=0)]
    )

    y_pred_prob = model.predict_proba(X_val_scaled)[:, 1]
    y_pred = model.predict(X_val_scaled)
    auc = roc_auc_score(y_val, y_pred_prob)
    f1 = f1_score(y_val, y_pred)
    auc_scores.append(auc)
    f1_scores.append(f1)
    print(f"Fold {fold + 1} | AUC: {auc:.4f} | F1: {f1:.4f}")

    if fold == 4:
        best_model = model
        final_sc = sc

print(f"\n【模型评估结果】")
print(f"交叉验证平均 AUC: {np.mean(auc_scores):.4f} (+/- {np.std(auc_scores):.4f})")
print(f"交叉验证平均 F1: {np.mean(f1_scores):.4f} (+/- {np.std(f1_scores):.4f})")

joblib.dump(best_model, 'lgb_model_no_leak.pkl')
joblib.dump(final_sc, 'scaler_no_leak.pkl')

# 测试集预测
test_features = test[features]
test_scaled = final_sc.transform(test_features)
test['prob'] = best_model.predict_proba(test_scaled)[:, 1]
test[['user_id', 'merchant_id', 'prob']].to_csv('prediction_no_leak.csv', index=None)
print("模型训练完成，预测结果已保存 prediction_no_leak.csv\n")