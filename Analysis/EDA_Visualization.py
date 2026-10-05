import pandas as pd

# 读取数据
train = pd.read_csv('train_format1.csv')
test = pd.read_csv('test_format1.csv', usecols=['user_id','merchant_id'])
user_info = pd.read_csv('userinfo.csv')
user_log = pd.read_csv('userlog.csv')

print("="*40)
print("【1. 正负样本统计】")
label_counts = train['label'].value_counts()
print(f"标签为0（未复购）的样本数：{label_counts.get(0, 0)}，占比 {label_counts.get(0, 0)/len(train)*100:.2f}%")
print(f"标签为1（复购）的样本数：{label_counts.get(1, 0)}，占比 {label_counts.get(1, 0)/len(train)*100:.2f}%")

print("\n" + "="*40)
print("【2. 性别与复购的关系】")
t_userinfo = train.merge(user_info, on='user_id', how='inner')
gender_label = pd.crosstab(t_userinfo['gender'], t_userinfo['label'], normalize='index') * 100
gender_counts = t_userinfo['gender'].value_counts().sort_index()
for gender in gender_counts.index:
    if gender == 0: g_name = "未知"
    elif gender == 1: g_name = "男性"
    elif gender == 2: g_name = "女性"
    else: g_name = f"性别{gender}"
    total = gender_counts[gender]
    rate_0 = gender_label.loc[gender, 0] if 0 in gender_label.columns else 0
    rate_1 = gender_label.loc[gender, 1] if 1 in gender_label.columns else 0
    print(f"{g_name} (共{total}人)：未复购率 {rate_0:.2f}%，复购率 {rate_1:.2f}%")

print("\n" + "="*40)
print("【3. 年龄与复购的关系】")
age_label = pd.crosstab(t_userinfo['age_range'], t_userinfo['label'], normalize='index') * 100
age_counts = t_userinfo['age_range'].value_counts().sort_index()
for age in age_counts.index:
    total = age_counts[age]
    rate_0 = age_label.loc[age, 0] if 0 in age_label.columns else 0
    rate_1 = age_label.loc[age, 1] if 1 in age_label.columns else 0
    print(f"年龄段 {age} (共{total}人)：未复购率 {rate_0:.2f}%，复购率 {rate_1:.2f}%")
print("\n")