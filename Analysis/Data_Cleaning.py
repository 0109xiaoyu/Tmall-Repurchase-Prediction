import pandas as pd

# 读取数据
train = pd.read_csv('train_format1.csv')
test = pd.read_csv('test_format1.csv', usecols=['user_id','merchant_id'])
user_info = pd.read_csv('user_info_format1.csv')

# 数据量太大，直接读取可能会导致内存溢出
# user_log = pd.read_csv('user_log_format1.csv')
def read_local(file_name, chunk_size=500000):
    # 读取为可迭代的TextFileReader对象
    reader = pd.read_csv(file_name, iterator=True, header=0)
    chunks = []
    loop = True
    while loop:
        try:
            # 每一次按500000行获取数据
            chunk = reader.get_chunk(chunk_size)
            chunks.append(chunk)
        except:
            loop = False
            print('数据获取完毕！')
    # 把列表的数据合并为数据框
    df = pd.concat(chunks, ignore_index=True)
    return df

# 通过函数，分块读取
user_log = read_local(file_name='user_log_format1.csv', chunk_size=500000)

test.isna().sum()
train.isna().sum()
user_info.isna().sum()
user_log.isna().sum()
user_log.duplicated().sum()

# 缺失值填补
user_info['age_range'].fillna(0, inplace=True)
user_info['gender'].fillna(2, inplace=True)
user_log['brand_id'].fillna(0, inplace=True)

# 去重
user_log.drop_duplicates(inplace=True)

# 重命名
user_log.rename(columns={'seller_id':'merchant_id'}, inplace=True)

# 数据保存
user_info.to_csv('userinfo.csv', index=None)
user_log.to_csv('userlog.csv', index=None)