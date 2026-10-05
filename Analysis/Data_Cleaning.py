import pandas as pd

# 读取数据
train = pd.read_csv('train_format1.csv')
test = pd.read_csv('test_format1.csv', usecols=['user_id','merchant_id'])
user_info = pd.read_csv('user_info_format1.csv')

# 分块读取大文件
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

user_log = read_local('user_log_format1.csv')

# 缺失值填补（修复 FutureWarning）
user_info['age_range'] = user_info['age_range'].fillna(0)
user_info['gender'] = user_info['gender'].fillna(2)
user_log['brand_id'] = user_log['brand_id'].fillna(0)

# 去重
user_log.drop_duplicates(inplace=True)

# 重命名
user_log.rename(columns={'seller_id':'merchant_id'}, inplace=True)

# 数据保存
user_info.to_csv('userinfo.csv', index=None)
user_log.to_csv('userlog.csv', index=None)
print("【数据清洗完成】")
print(f"用户信息数据：{user_info.shape[0]} 行，{user_info.shape[1]} 列")
print(f"用户行为日志：{user_log.shape[0]} 行，{user_log.shape[1]} 列")
print("已保存 userinfo.csv 和 userlog.csv\n")