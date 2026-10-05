import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import shap
import warnings

# 忽略一些版本警告
warnings.filterwarnings('ignore')

# ================= 1. 环境与数据准备 =================
# 解决 Windows 系统下 Matplotlib 中文显示乱码问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

print("正在加载模型与数据...")
# 加载实训4保存的模型和标准化器
model_lgb = joblib.load('lgb_model.pkl')
sc = joblib.load('scaler.pkl')

# 读取训练集，准备和实训4完全一致的验证集
train = pd.read_csv('train.csv')
n = sum(train['label'] == 1)
posdata = train[train['label'] == 0].sample(n, random_state=123)
negdata = train[train['label'] == 1]
data = pd.concat([posdata, negdata], axis=0)

# 提取特征列（排除 ID 和 label）
features = [col for col in data.columns if col not in ['user_id', 'merchant_id', 'label']]
X = sc.transform(data[features])  # 必须是 transform，复用实训4的标准化器
y = data['label']

# 重新划分验证集（必须和实训4的参数一模一样，否则数据对不上）
from sklearn.model_selection import train_test_split
xtrain, xval, ytrain, yval = train_test_split(
    X, y, test_size=0.2, random_state=2021
)

print(f"验证集大小: {xval.shape}")

# ================= 2. SHAP 可解释性分析 =================
print("\n开始计算 SHAP 值（可能需要1-2分钟，请耐心等待）...")
# 为了防止内存溢出，只抽样 1000 个样本进行 SHAP 解释
sample_idx = np.random.choice(xval.shape[0], size=1000, replace=False)
xval_sample = xval[sample_idx]

# 构建树模型解释器
explainer = shap.TreeExplainer(model_lgb)
shap_values = explainer.shap_values(xval_sample)

# 兼容不同版本的 SHAP：二分类模型有时会返回一个包含两个数组的 list
if isinstance(shap_values, list):
    shap_values = shap_values[1]  # 取类别1（复购）的 SHAP 值

# 绘制 SHAP 摘要图
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, xval_sample, feature_names=features, show=False)
plt.title('特征对复购预测的 SHAP 值影响 (Top特征)', fontsize=16)
plt.tight_layout()
plt.savefig('shap_summary.png', dpi=300, bbox_inches='tight')
plt.show()
print("SHAP 摘要图已保存为 'shap_summary.png'")

# ================= 3. Lift 提升曲线（业务评估） =================
print("\n正在计算 Lift 曲线...")
# 预测验证集概率
y_pred_proba = model_lgb.predict_proba(xval)[:, 1]

# 构建 DataFrame 并按概率降序排序
df_eval = pd.DataFrame({'y_true': yval, 'y_prob': y_pred_proba})
df_eval = df_eval.sort_values(by='y_prob', ascending=False).reset_index(drop=True)

# 计算累计正例和累计增益
df_eval['cumulative_positives'] = df_eval['y_true'].cumsum()
df_eval['total_positives'] = df_eval['y_true'].sum()
df_eval['cumulative_gain'] = df_eval['cumulative_positives'] / df_eval['total_positives']

# 计算 Lift
overall_positive_rate = df_eval['y_true'].mean()
df_eval['percentile'] = df_eval.index / len(df_eval)
df_eval['lift'] = (df_eval['cumulative_positives'] / (df_eval.index + 1)) / overall_positive_rate

# 绘制 Lift 曲线
plt.figure(figsize=(8, 6))
plt.plot(df_eval['percentile'], df_eval['lift'], label='模型 Lift', color='r', linewidth=2)
plt.axhline(y=1, color='b', linestyle='--', label='随机猜测基准线 (Lift=1)')
plt.title('Lift 曲线 (提升曲线) - 业务效果评估', fontsize=14)
plt.xlabel('样本百分比 (按预测概率降序)')
plt.ylabel('Lift (提升倍数)')
plt.legend()
plt.grid(True)
plt.savefig('lift_curve.png', dpi=300, bbox_inches='tight')
plt.show()

# 打印关键业务指标 (Top 10% 和 Top 20%)
for p in [0.1, 0.2]:
    idx = int(len(df_eval) * p)
    lift_val = df_eval.loc[idx, 'lift']
    gain_val = df_eval.loc[idx, 'cumulative_gain']
    print(f"【业务指标】Top {int(p*100)}% 的用户 -> 覆盖了 {gain_val*100:.2f}% 的复购用户，效率提升了 {lift_val:.2f} 倍。")

# ================= 4. 业务落地：高潜用户名单 =================
print("\n正在生成测试集的高潜用户名单...")
# 读取测试集
test = pd.read_csv('test.csv')
# 提取测试集特征并进行标准化（必须用训练集的 sc）
test_features = sc.transform(test[features])
# 预测概率
test['prob'] = model_lgb.predict_proba(test_features)[:, 1]

# 筛选预测概率 > 0.7 的高潜用户（可根据业务需要调整阈值）
high_potential = test[test['prob'] > 0.7][['user_id', 'merchant_id', 'prob']]
high_potential = high_potential.sort_values(by='prob', ascending=False)

# 导出名单
high_potential.to_csv('high_potential_users.csv', index=False)
print(f"已生成高潜用户名单，共 {len(high_potential)} 人，保存为 'high_potential_users.csv'。")
print("\n====== 执行完毕！======")