import pandas as pd
from sklearn.preprocessing import StandardScaler
import lightgbm
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, precision_score, f1_score
import joblib  # 新增：用于保存模型

# 读取数据
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# 样本不平衡处理
train['label'].value_counts()
n = sum(train['label'] == 1)
posdata = train[train['label'] == 0].sample(n, random_state=123)
negdata = train[train['label'] == 1]
data = pd.concat([posdata, negdata], axis=0)

# 标准化 (修复：不用硬编码的 iloc[:, 3:]，而是剔除 ID 列和 label 列)
features = [col for col in data.columns if col not in ['user_id', 'merchant_id', 'label']]
sc = StandardScaler()
df = sc.fit_transform(data[features])
df = pd.DataFrame(df, columns=features)

# 划分训练数据和验证数据
xtrain, xval, ytrain, yval = train_test_split(
    df, data['label'], test_size=0.2, random_state=2021
)

# 模型构建,模型初始化
model_lgb = lightgbm.LGBMClassifier(
    n_estimators=1000,
    max_depth=8,
    num_leaves=25,
    colsample_bytree=0.5,
    learning_rate=0.1,
    metric='auc'
)

# 执行模型训练
model_lgb.fit(
    xtrain, ytrain,
    eval_metric='auc',
    eval_set=[(xtrain, ytrain), (xval, yval)],
    callbacks=[
        lightgbm.early_stopping(stopping_rounds=100),
        lightgbm.log_evaluation(period=1)
    ]
)

# 模型评价
print(model_lgb.best_score_)
pres = model_lgb.predict(xval)
presprob = model_lgb.predict_proba(xval)[:, 1]

print(confusion_matrix(yval, pres))
print(classification_report(yval, pres))
print('准确率：', precision_score(yval, pres))
print('F1值：', f1_score(yval, pres))
print('AUC值：', roc_auc_score(yval, presprob))

# 模型训练的过程
lightgbm.plot_metric(model_lgb.evals_result_, metric='auc')

# 特征重要性
lightgbm.plot_importance(model_lgb)

# 模型预测
# 修复：必须复用训练集的 sc，而不是重新 fit_transform 测试集！
testdf = sc.transform(test[features])
testdf = pd.DataFrame(testdf, columns=features)

# 调用模型预测
test['prob'] = model_lgb.predict_proba(testdf)[:, 1]

# 结果的导出
test[['user_id', 'merchant_id', 'prob']].to_csv('prediction.csv', index=None)

# 保存模型和标准化器
joblib.dump(model_lgb, 'lgb_model.pkl')
joblib.dump(sc, 'scaler.pkl')
print("模型与标准化器已保存！")