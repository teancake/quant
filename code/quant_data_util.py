import pandas as pd
from tqdm import tqdm
from uuid import uuid1
from datetime import datetime, timedelta


from log_util import get_logger
from starrocks_local_db_util import StarrocksDbUtil

logger = get_logger(__name__)



db_util = StarrocksDbUtil()

table_name = "temp_stock_zh_a_data"
train_ratio = 0.8
chunk_table_name = table_name + "_uuid_" + uuid1().hex
table_chunksize = 100

def _create_chunk_table():
    sql = '''
    create table if not exists {} as 
    select *, floor(temp_rn / {}) as chunk_num from (
    select *, row_number() over () as temp_rn from {}
    )a;
        '''.format(chunk_table_name, table_chunksize, table_name)
    db_util.run_sql(sql)
    logger.info("chunk table created, table name {}".format(chunk_table_name))
    sql = "select distinct chunk_num from {}".format(chunk_table_name)
    data = db_util.run_sql(sql)
    chunk_list = sorted([item[0] for item in data])
    logger.info("chunksize {}, {} chunks in total.".format(table_chunksize, len(chunk_list)))
    return chunk_list

def _delete_chunk_table():
    sql = "drop table if exists {};".format(chunk_table_name)
    db_util.run_sql(sql)
    logger.info("chunk table deleted, table name {}".format(chunk_table_name))


def load_data_from_db():
    chunk_list = _create_chunk_table()
    chunk_data_df_list = []
    for chunk in tqdm(chunk_list):
        sql = "select * from {} where chunk_num = {};".format(chunk_table_name, chunk)
        chunk_data = db_util.run_sql(sql, log=False)
        chunk_data_df = pd.DataFrame(chunk_data)
        chunk_data_df_list.append(chunk_data_df)

    _delete_chunk_table()
    print("merge dataframes, size {}".format(len(chunk_data_df_list)))
    all_data = pd.concat(chunk_data_df_list, ignore_index=True, sort=False)

    split_only_last_date = False

    if split_only_last_date:
        train_data = all_data[all_data['日期'] != all_data['日期'].max()]
        test_data = all_data[all_data['日期'] == all_data['日期'].max()]
    else:
        days = round((all_data['日期'].max() - all_data['日期'].min()).days * train_ratio)
        split_date = all_data['日期'].min() + timedelta(days=days)
        print("split date is {}".format(split_date))
        train_data = all_data[all_data['日期'] <= split_date]
        test_data = all_data[all_data['日期'] > split_date]

    print("data columns {}".format(all_data.columns))

    print("load data for prediction")
    sql = "select * from temp_stock_zha_pred order by 代码 asc ;"
    data = db_util.run_sql(sql, log=False, chunksize=10)
    pred_data = pd.DataFrame(data)
    print("prediction data loaded.")
    train_df_x = train_data.loc[:, "ma_5":"dc_lower"].fillna(0).values.astype(float)
    train_df_y = train_data.loc[:, "label"].values.astype(float)

    test_df_x = test_data.loc[:, "ma_5":"dc_lower"].fillna(0).values.astype(float)
    test_df_y = test_data.loc[:, "label"].values.astype(float)

    pred_df_x = pred_data.loc[:, "ma_5":"dc_lower"].fillna(0).values.astype(float)
    append_df_x = pred_data.loc[:, "日期":"代码"]

    return train_df_x, train_df_y, test_df_x, test_df_y, pred_df_x, append_df_x



