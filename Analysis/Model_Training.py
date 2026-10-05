import pandas as pd
import numpy as np
import shap
import joblib
import warnings
warnings.filterwarnings('ignore')

# 加载模型
model = joblib.load('lgb_model_no_leak.pkl')
sc = joblib.load('scaler_no_leak.pkl')
train = pd.read_csv('train_clean.csv')

features = [col for col in train.columns if col not in ['user_id', 'merchant_id', 'label']]
X_sample = train[features].sample(1000, random_state=42)
X_sample_scaled = sc.transform(X_sample)

print("="*50)
print("【1. SHAP 特征重要性分析】")
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_sample_scaled)
if isinstance(shap_values, list): shap_values = shap_values[1]

# 计算每个特征的平均绝对SHAP值（即全局重要性）
mean_shap = np.abs(shap_values).mean(axis=0)
feature_importance = pd.DataFrame({'feature': features, 'shap_importance': mean_shap})
feature_importance = feature_importance.sort_values('shap_importance', ascending=False).reset_index(drop=True)

print("Top 10 最重要特征（复购核心驱动因子）：")
for i in range(min(10, len(feature_importance))):
    row = feature_importance.iloc[i]
    print(f"  {i+1}. {row['feature']} (SHAP值: {row['shap_importance']:.4f})")

print("\n" + "="*50)
print("【2. Lift 提升曲线与业务评估】")
# 使用训练集评估业务指标（实际项目中应使用独立验证集）
train_probs = model.predict_proba(sc.transform(train[features]))[:, 1]
df_eval = train[['label']].copy()
df_eval['prob'] = train_probs
df_eval = df_eval.sort_values('prob', ascending=False).reset_index(drop=True)

df_eval['cum_pos'] = df_eval['label'].cumsum()
df_eval['total_pos'] = df_eval['label'].sum()
df_eval['gain'] = df_eval['cum_pos'] / df_eval['total_pos']
overall_rate = df_eval['label'].mean()
df_eval['lift'] = (df_eval['cum_pos'] / (df_eval.index + 1)) / overall_rate

print(f"总体复购率基准：{overall_rate*100:.2f}%")
print("Top K% 用户业务价值：")
for p in [0.05, 0.10, 0.20, 0.30]:
    idx = int(len(df_eval) * p)
    gain_val = df_eval.loc[idx, 'gain'] * 100
    lift_val = df_eval.loc[idx, 'lift']
    print(f"  Top {int(p*100)}% 用户 -> 覆盖 {gain_val:.2f}% 的复购用户，营销效率提升 {lift_val:.2f} 倍")

print("\n" + "="*50)
print("【3. 高潜用户名单生成】")
test = pd.read_csv('test_clean.csv')
test_scaled = sc.transform(test[features])
test['prob'] = model.predict_proba(test_scaled)[:, 1]
high_potential = test[test['prob'] > 0.7][['user_id', 'merchant_id', 'prob']].sort_values('prob', ascending=False)
high_potential.to_csv('high_potential_no_leak.csv', index=False)
print(f"筛选出预测概率 > 0.7 的高潜用户，共 {len(high_potential)} 人")
print("名单已保存至 high_potential_no_leak.csv\n")