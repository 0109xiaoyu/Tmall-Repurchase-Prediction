import pandas as pd
import numpy as np
from sklearn.model_selection import KFold

# 读取数据
train = pd.read_csv('train_format1.csv')
test = pd.read_csv('test_format1.csv', usecols=['user_id', 'merchant_id'])
user_info = pd.read_csv('userinfo.csv')


def read_local(file_name, chunk_size=500000):
    reader = pd.read_csv(file_name, iterator=True, header=0)
    chunks = []
    while True:
        try:
            chunk = reader.get_chunk(chunk_size)
            chunks.append(chunk)
        except StopIteration:
            break
    return pd.concat(chunks, ignore_index=True)


user_log = read_local('userlog.csv')

# 基础类型转换
user_log['item_id'] = user_log['item_id'].astype('int32')
user_log['cat_id'] = user_log['cat_id'].astype('int32')
user_log['brand_id'] = user_log['brand_id'].astype('int32')
user_log['action_type'] = user_log['action_type'].astype('int8')
user_log['time_stamp'] = pd.to_datetime(user_log['time_stamp'], format='%m%d')

# 筛选用户和商家
matrix = pd.concat([train, test], axis=0)
user_log = pd.merge(user_log, matrix[['user_id', 'merchant_id']],
                    on=['user_id', 'merchant_id'], how='inner')


def build_features(df):
    user_feat = df.groupby('user_id').agg(
        user_num=('item_id', 'size'),
        user_days=('time_stamp', lambda x: (x.max() - x.min()).days),
        item_num=('item_id', 'nunique'),
        cat_num=('cat_id', 'nunique'),
        brand_num=('brand_id', 'nunique'),
        merchant_num=('merchant_id', 'nunique')
    ).reset_index()
    action_pivot = pd.pivot_table(df, index='user_id', columns='action_type',
                                  aggfunc='size', fill_value=0).reset_index()
    action_pivot.columns = ['user_id', 'uclick_num', 'uadd_num', 'ubuy_num', 'usave_num']
    user_feat = pd.merge(user_feat, action_pivot, on='user_id', how='left')
    user_feat['user_buy_rate'] = user_feat['ubuy_num'] / (user_feat['uclick_num'] + 1)

    merchant_feat = df.groupby('merchant_id').agg(
        merchant_benum=('item_id', 'size'),
        merchant_user=('user_id', 'nunique'),
        merchant_item=('item_id', 'nunique'),
        merchant_cat=('cat_id', 'nunique'),
        merchant_brand=('brand_id', 'nunique')
    ).reset_index()
    m_action_pivot = pd.pivot_table(df, index='merchant_id', columns='action_type',
                                    aggfunc='size', fill_value=0).reset_index()
    m_action_pivot.columns = ['merchant_id', 'mclick_num', 'mbuy_num', 'madd_num', 'msave_num']
    merchant_feat = pd.merge(merchant_feat, m_action_pivot, on='merchant_id', how='left')
    merchant_feat['merchant_buy_rate'] = merchant_feat['mbuy_num'] / (merchant_feat['mclick_num'] + 1)

    cross_feat = df.groupby(['user_id', 'merchant_id']).agg(
        user_merchant=('item_id', 'size'),
        user_merchant_item=('item_id', 'nunique'),
        user_merchant_cat=('cat_id', 'nunique'),
        user_merchant_brand=('brand_id', 'nunique'),
        user_merchant_days=('time_stamp', lambda x: (x.max() - x.min()).days)
    ).reset_index()
    c_action_pivot = pd.pivot_table(df, index=['user_id', 'merchant_id'], columns='action_type',
                                    aggfunc='size', fill_value=0).reset_index()
    c_action_pivot.columns = ['user_id', 'merchant_id', 'user_merchant_click', 'user_merchant_add', 'user_merchant_buy',
                              'user_merchant_save']
    cross_feat = pd.merge(cross_feat, c_action_pivot, on=['user_id', 'merchant_id'], how='left')
    cross_feat['user_merchant_buy_rate'] = cross_feat['user_merchant_buy'] / (cross_feat['user_merchant_click'] + 1)
    return user_feat, merchant_feat, cross_feat


train_user_feat, train_merchant_feat, train_cross_feat = build_features(user_log)
test_user_feat, test_merchant_feat, test_cross_feat = build_features(user_log)


def merge_features(base_df, user_f, merchant_f, cross_f):
    df = pd.merge(base_df, user_info, on='user_id', how='left')
    df = pd.merge(df, user_f, on='user_id', how='left')
    df = pd.merge(df, merchant_f, on='merchant_id', how='left')
    df = pd.merge(df, cross_f, on=['user_id', 'merchant_id'], how='left')
    return df


train_df = merge_features(train, train_user_feat, train_merchant_feat, train_cross_feat)
test_df = merge_features(test, test_user_feat, test_merchant_feat, test_cross_feat)

# ================= 防泄露：K折交叉目标编码 =================
train_df['merchant_rebuy'] = 0.0
kf = KFold(n_splits=5, shuffle=True, random_state=42)
global_rebuy_mean = train_df.loc[train_df['label'] == 1, 'merchant_id'].value_counts() / train_df[
    'merchant_id'].value_counts()
global_rebuy_mean = global_rebuy_mean.fillna(0)

for train_idx, val_idx in kf.split(train_df):
    rebuy_count = train_df.iloc[train_idx].loc[train_df.iloc[train_idx]['label'] == 1, 'merchant_id'].value_counts()
    total_count = train_df.iloc[train_idx]['merchant_id'].value_counts()
    rebuy_rate = (rebuy_count / total_count).fillna(0)
    val_merchants = train_df.iloc[val_idx]['merchant_id']
    train_df.loc[train_df.index[val_idx], 'merchant_rebuy'] = val_merchants.map(rebuy_rate).fillna(0)

test_df['merchant_rebuy'] = test_df['merchant_id'].map(global_rebuy_mean).fillna(0)

# 处理离散特征和缺失值
for df in [train_df, test_df]:
    df['age_range'] = df['age_range'].astype('int8')
    df['gender'] = df['gender'].fillna(2).astype('int8')
    df = df.fillna(0)

# 独热编码
train_df = pd.get_dummies(train_df, columns=['age_range', 'gender'], prefix=['age', 'g'])
test_df = pd.get_dummies(test_df, columns=['age_range', 'gender'], prefix=['age', 'g'])

missing_cols = set(train_df.columns) - set(test_df.columns)
for c in missing_cols:
    test_df[c] = 0
test_df = test_df[train_df.columns]

train_df.to_csv('train_clean.csv', index=None)
test_df.to_csv('test_clean.csv', index=None)
print("【特征工程完成】")
print(f"训练集维度：{train_df.shape}")
print(f"测试集维度：{test_df.shape}")
print("已保存 train_clean.csv 和 test_clean.csv\n")