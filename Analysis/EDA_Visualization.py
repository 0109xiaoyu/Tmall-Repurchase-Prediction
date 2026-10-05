import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 读取数据
train = pd.read_csv('train_format1.csv')
test = pd.read_csv('test_format1.csv', usecols=['user_id','merchant_id'])
user_info = pd.read_csv('userinfo.csv')
user_log = pd.read_csv('userlog.csv')

# 正负样本统计
label = train['label'].value_counts()
# 绘制饼图
plt.pie(label, labels=label.index, autopct='%.2f%%', explode=[0, 0.3])
plt.title('0 VS 1')
plt.show()

# 性别和复购的关系
# 表连接
t_userinfo = train.merge(user_info, on='user_id', how='inner')
# 绘图
sns.countplot(x='gender', hue='label', data=t_userinfo)
plt.title('gender & label')
plt.show()

# 年龄和复购的关系
sns.countplot(x='age_range', hue='label', data=t_userinfo)
plt.title('age & label')
plt.show()