# %%
import polars as pl
import pandas as pd
from sklearn import linear_model
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

def getXY(data_path):
    raw_data = pl.read_csv(data_path, dtypes={'年龄' : pl.Float64}).with_columns(pl.col('年龄').cast(pl.Int64))
    # %%
    # 筛下奇怪的值
    data = raw_data.filter(pl.col('年龄') >= 1)
    data.null_count()
    cols = data.columns
    feature_cols = cols[:-1]
    # 不筛 Y
    for c in feature_cols:
        data = data.filter(pl.col(c) > 0)
    # %%
    # 算下每个特征和是否高血压的相关度
    # cols = data.columns[:-1]
    # corr = pl.DataFrame()
    # for col in cols:
    #     corr = corr.with_columns(pl.corr(data[col], data['是否高血压']))
    # corr
    # %%
    # 选一下相关度值在0.1以上的
    # rel_cols = []
    # discard_cols = []
    # for col in cols:
    #     if abs(corr[col][0]) >= 0.1:
    #         rel_cols.append(col)
    #     else:
    #         discard_cols.append(col)
    # print(rel_cols)
    # print(discard_cols)
    # %%
    # 获取x,y
    raw_x = data.select(feature_cols)
    raw_y = data.select('是否高血压')
    return raw_x, raw_y

# %%
# 数据归一化
def data_normalizeation(x_pl):
    for xcol in x_pl.columns:
        x_pl = x_pl.with_columns((pl.col(xcol) - pl.col(xcol).mean())/pl.col(xcol).std())
    return x_pl

# %%
def cross_partition(folds : int, x_pl : pl.DataFrame, y_pl : pl.DataFrame):
    data_partitions = []
    data_cnts = x_pl.shape[0]
    part_size = data_cnts // folds
    for f in range(folds):
        # data_partitions.append(x_pl[:part_size * f], y_pl[])
        l = f * part_size
        if f == folds-1:
            r = data_cnts
        else:
            r = (f+1) * part_size
        train_x = pl.concat([x_pl[:l], x_pl[r:]])
        train_y = pl.concat([y_pl[:l], y_pl[r:]])
        test_x = x_pl[l:r]
        test_y = y_pl[l:r]
        data_partitions.append((train_x, train_y, test_x, test_y))
    return data_partitions

# %%
