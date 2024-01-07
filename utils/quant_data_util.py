import os

import pandas as pd
from tqdm import tqdm
from uuid import uuid1
from datetime import datetime, timedelta

from utils.log_util import get_logger
from starrocks_db_util import StarrocksDbUtil
from concurrent.futures import ProcessPoolExecutor, wait
import numpy as np
import uuid
import math

import joblib

from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.metrics import mean_squared_error

from dw.datax_processor import DataxProcessor

logger = get_logger(__name__)

db_util = StarrocksDbUtil()

train_ratio = 0.95
split_only_last_date = False


def get_run_id(prefix=None):
    id = datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid1().hex[0:10]
    if prefix is not None:
        id = prefix + "-" + id
    return id

def _generate_chunk_table_name(table_name):
    return table_name + "_uuid_" + uuid1().hex

def _create_chunk_table(table_name, table_chunksize=100, ds=""):
    chunk_table_name = _generate_chunk_table_name(table_name)
    ds_cond = "where ds = '{}'".format(ds) if ds else ""
    sql = '''
    create table if not exists {} 
    DUPLICATE KEY(chunk_num)
    DISTRIBUTED BY HASH(chunk_num) BUCKETS 32
    as select floor(temp_rn / {}) as chunk_num, * from (
    select *, row_number() over () as temp_rn from {} {}
    )a;
        '''.format(chunk_table_name, table_chunksize, table_name, ds_cond)
    db_util.run_sql(sql)
    logger.info("chunk table created, table name {}".format(chunk_table_name))
    sql = "select distinct chunk_num from {}".format(chunk_table_name)
    data = db_util.run_sql(sql)
    chunk_list = sorted([item[0] for item in data])
    logger.info("chunksize {}, {} chunks in total.".format(table_chunksize, len(chunk_list)))
    return chunk_list, chunk_table_name


def _delete_chunk_table(chunk_table_name):
    sql = "drop table if exists {};".format(chunk_table_name)
    db_util.run_sql(sql)
    logger.info("chunk table deleted, table name {}".format(chunk_table_name))


def _read_table_by_chunk(table_name, ds="", chunksize=100):
    print("load data from {}".format(table_name))
    chunk_list, chunk_table_name = _create_chunk_table(table_name, table_chunksize=chunksize, ds=ds)
    chunk_data_df_list = []
    for chunk in tqdm(chunk_list):
        sql = "select * from {} where chunk_num = {};".format(chunk_table_name, chunk)
        chunk_data = db_util.run_sql(sql, log=False)
        chunk_data_df = pd.DataFrame(chunk_data)
        chunk_data_df_list.append(chunk_data_df)

    _delete_chunk_table(chunk_table_name)
    print("merge dataframes, size {}".format(len(chunk_data_df_list)))
    all_data = pd.concat(chunk_data_df_list, ignore_index=True, sort=False)
    print("data columns {}".format(all_data.columns))
    del chunk_data_df_list
    return all_data




def df_from_csv(file_path, column_types=None, parse_dates=None):
    data = pd.read_csv(file_path, dtype=column_types, parse_dates=parse_dates)
    return data


def standardize_data(data_df, column_names, scaler=None):
    data = data_df.loc[:, column_names]
    if scaler is None:
        scaler = StandardScaler()
        scaler.fit(data)
    data = scaler.transform(data)
    data_df.loc[:, column_names] = data
    return data_df, scaler


def encode_categorical_features(data_df, column_names, encoder=None):
    enc_column_names = ["{}_enc".format(name) for name in column_names]
    data = data_df.loc[:, column_names]
    if encoder is None:
        encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        encoder.fit(data)
    data = encoder.transform(data)
    data_df.loc[:, enc_column_names] = data
    return data_df, enc_column_names, encoder


def _get_sequential_data_nparray(symbol_df, scaler, sequence_length, numerical_feature_columns,
                                 label_columns, ext_columns, categorical_feature_columns):
    list_seq = []
    list_cat = []
    list_y = []
    list_ext = []
    # print("whole df shape{}, symbol {}, min date {}, max date {}".format(symbol_df.shape, symbol_df["代码"].max(), symbol_df["日期"].min(), symbol_df["日期"].max()))

    for i in range(symbol_df.shape[0] - sequence_length + 1):
        date_df = symbol_df[i:i + sequence_length]
        # print("i {}, len {}, symbol {}, min date {}, max date {}".format(i, date_df.shape, date_df["代码"].max(), date_df["日期"].min(), date_df["日期"].max()))

        seq = date_df.loc[:, numerical_feature_columns].fillna(0).values.astype(float)
        y = None if label_columns is None else date_df.loc[:, label_columns].values.astype(float)

        temp_df = date_df.copy()
        if scaler is not None:
            temp_col_names = scaler.get_feature_names_out()
            temp_df.loc[:, temp_col_names] = scaler.inverse_transform(temp_df.loc[:, temp_col_names])
        ext = None if ext_columns is None else temp_df.loc[:, ext_columns].tail(1)
        cat = None if categorical_feature_columns is None else temp_df.loc[:, categorical_feature_columns].tail(1)

        list_seq.append(seq)
        list_cat.append(cat)
        list_y.append(y)
        list_ext.append(ext)
    return list_seq, list_cat, list_y, list_ext


def _generate_sequential_data(df, scaler=None, sequence_length=1, numerical_feature_columns=None,
                              label_columns=None, ext_columns=None, categorical_feature_columns=None):
    POOL_SIZE = 16
    pool = ProcessPoolExecutor(max_workers=POOL_SIZE)
    data_seq = []
    data_cat = []
    data_y = []
    data_ext = []
    futures = []
    print("numerical feature columns {}, label columns {}, ext columns {}, categorical feature columns {}".format(
        numerical_feature_columns, label_columns, ext_columns, categorical_feature_columns))
    for symbol in tqdm(np.sort(df["代码"].unique())):
        symbol_df = df[df["代码"] == symbol].sort_values(by=["日期"], ascending=True)
        future = pool.submit(_get_sequential_data_nparray, symbol_df, scaler, sequence_length,
                             numerical_feature_columns, label_columns, ext_columns, categorical_feature_columns)
        futures.append(future)
        if len(futures) % POOL_SIZE == POOL_SIZE - 1:
            # the main purpose of waiting is to make tqdm correct.
            wait(futures)
        # if symbol > "000010":
        #     break

    wait(futures)

    for future in futures:
        list_seq, list_cat, list_y, list_ext = future.result()
        data_seq.extend(list_seq)
        data_cat.extend(list_cat)
        data_y.extend(list_y)
        data_ext.extend(list_ext)

    pool.shutdown(wait=True)

    data_seq = np.stack(data_seq, axis=0)
    data_cat = np.stack(data_cat, axis=0)
    data_y = np.stack(data_y, axis=0)
    data_ext = pd.concat(data_ext, ignore_index=True, sort=False)

    logger.info("sequential data x shape {}, sequential data y shape {}, ext data shape {}".format(data_seq.shape, data_y.shape, data_ext.shape))
    return data_seq, data_cat, data_y, data_ext


def save_prediction_to_db(data_ext, ds, model_name, stage, pred_name, pred_val_col_name, label_col_name, run_id):
    df = data_ext.loc[:, ['日期','代码', pred_val_col_name, label_col_name]]
    table_name = "ods_stock_zh_a_prediction"
    logger.info(
        "save {} results to db, table name {}, ds {}, model {}, stage {}, run_id {}".format(stage, table_name, ds,
                                                                                            model_name,
                                                                                            stage, run_id))
    df["ds"] = ds
    df["gmt_create"] = datetime.now()
    df["gmt_modified"] = datetime.now()
    df["stage"] = stage
    df["model"] = model_name
    df["run_id"] = run_id
    df["pred_name"] = pred_name
    df["pred_value"] = df[pred_val_col_name]
    if pred_val_col_name != "pred_value":
        df = df.drop(columns=[pred_val_col_name])
    df["label"] = df[label_col_name]
    if label_col_name != "label":
        df = df.drop(columns=[label_col_name])

    df.to_sql(name=table_name, con=db_util.get_db_engine(), if_exists="append", index=False, method="multi",
              chunksize=1000)
    logger.info("{} results saved to db.".format(stage))


def generate_data_file_name(ds="", prefix="quant_data", file_type="pkl"):
    if ds is not None and len(ds) > 0:
        ds = "_" + ds
    return prefix + ds  + "." + file_type


def generate_sequential_data_file_name(ds=""):
    if ds is not None and len(ds) > 0:
        ds = "_" + ds
    return "quant_sequential_data" + ds + ".pkl"


def clean_up_old_training_data(ds, days_ahead=3):
    for i in range(days_ahead, days_ahead + 7):
        tmp_ds = (datetime.strptime(ds, '%Y%m%d') - timedelta(days=i)).strftime("%Y%m%d")
        train_file_name, pred_file_name = get_train_pred_file_names(tmp_ds)
        train_conf_file_name = DataxProcessor.get_conf_file_name(train_file_name)
        pred_conf_file_name = DataxProcessor.get_conf_file_name(pred_file_name)
        print("remove files {}".format([train_file_name, pred_file_name, train_conf_file_name, pred_conf_file_name]))
        try:
            os.remove(train_file_name)
            os.remove(pred_file_name)
            os.remove(train_conf_file_name)
            os.remove(pred_conf_file_name)
        except OSError:
            pass

def get_zh_a_hist_data_file_names(ds):
    prefix = "datax_zh_a_hist_data"
    if ds is not None and len(ds) > 0:
        ds = "_" + ds
    data_file_prefix = prefix + ds
    data_file_name = f"{data_file_prefix}.csv"
    return data_file_name


def clean_up_old_zh_a_hist_data(ds, days_ahead=3):
    for i in range(days_ahead, days_ahead + 7):
        tmp_ds = (datetime.strptime(ds, '%Y%m%d') - timedelta(days=i)).strftime("%Y%m%d")
        tmp_data_file_name = get_zh_a_hist_data_file_names(tmp_ds)
        tmp_conf_file_name = DataxProcessor.get_conf_file_name(tmp_data_file_name)
        print("remove files {}".format([tmp_data_file_name, tmp_conf_file_name]))
        try:
            os.remove(tmp_data_file_name)
            os.remove(tmp_conf_file_name)
        except OSError:
            pass

def db_to_csv_by_sql(data_file_name, table_name="", ds="", chunksize=100):
    _read_table_by_chunk(table_name, ds=ds, chunksize=chunksize)


def db_to_csv_by_datax(data_file_name, table_name, columns="*", where=""):
    processor = DataxProcessor(columns=columns, table=table_name, where=where, data_file_name=data_file_name)
    processor.process()


def db_to_csv(data_file_name, table_name, columns, where=""):
    by_sql = 0
    if by_sql:
        return db_to_csv_by_sql(data_file_name, table_name)
    else:
        return db_to_csv_by_datax(data_file_name, table_name, columns, where)


def get_train_pred_file_names(ds):
    prefix = "quant_data"
    if ds is not None and len(ds) > 0:
        ds = "_" + ds
    data_file_prefix = prefix + ds
    train_file_name = f"{data_file_prefix}_train.csv"
    pred_file_name = f"{data_file_prefix}_pred.csv"
    return train_file_name, pred_file_name

def sync_zh_a_hist_data(ds):
    table_name = "dwd_stock_zh_a_hist_df"
    columns = ["代码", "period", "adjust", "日期", "开盘", "收盘", "最高", "最低", "成交量", "成交额", "振幅", "涨跌幅", "涨跌额", "换手率"]
    where = f"ds='{ds}' and period='daily' and adjust='hfq'"
    data_file_name = get_zh_a_hist_data_file_names(ds)
    db_to_csv(data_file_name, table_name, columns=columns, where=where)


def sync_training_data(ds):
    train_table_name = "ads_stock_zh_a_training_data"
    pred_table_name = "ads_stock_zh_a_pred_data"
    train_table_columns = ["日期", "代码", "ma_5", "ma_10", "ma_20", "ma_60", "ma_120", "ma_240", "hhv_5", "hhv_10",
                           "hhv_20", "hhv_60", "hhv_120", "hhv_240", "llv_5", "llv_10", "llv_20", "llv_60", "llv_120",
                           "llv_240", "bias_6", "bias_12", "bias_24", "boll_upper_20", "boll_mid_20", "boll_lower_20",
                           "rsi_6", "rsi_12", "rsi_24", "wr_10", "wr_6", "mtm_12", "mtm_12_ma_6", "k_9", "d_9", "j_9",
                           "macd_dif", "macd_dea", "macd", "dmi_pdi", "dmi_mdi", "dmi_adx", "dmi_adxr", "obv", "cci",
                           "roc_12", "ma_6_roc_12", "bbi", "expma_12", "expma_50", "ar", "br", "atr", "dma_dif", "dma",
                           "emv", "maemv", "psy", "psyma", "asi", "asit", "mfi", "mass", "mamass", "dpo", "madpo", "vr",
                           "trix", "trma", "kc_upper", "kc_mid", "kc_lower", "dc_upper", "dc_mid", "dc_lower", "open",
                           "close", "high", "low", "roc", "volume", "turnover_rate", "sym", "symbol", "date_mon",
                           "date_dow", "date_rn", "roc_m1d", "roc_m2d", "roc_m3d", "industry", "is_index_000001",
                           "is_index_000016", "is_index_000300", "label_roc", "label_cls", "label_close", "label",
                           "label_roi_1d", "label_roi_2d", "label_roi_3d", "label_roi_5d", "label_roi_10d"]
    pred_table_columns = train_table_columns

    train_file_name, pred_file_name = get_train_pred_file_names(ds)

    logger.info("load data from db.")
    db_to_csv(train_file_name, train_table_name, columns=train_table_columns)
    db_to_csv(pred_file_name, pred_table_name, columns=pred_table_columns)


def load_data(train_file_name, pred_file_name, standardize_column_names=None, categorical_column_names=None):
    column_types = {"代码": str}
    parse_dates = ["日期"]
    all_data = df_from_csv(train_file_name, column_types, parse_dates)
    pred_data = df_from_csv(pred_file_name, column_types, parse_dates)

    logger.info("spliting train test, split_only_last_date {},  train_ratio {}".format(split_only_last_date, train_ratio))
    train_data, test_data = _split_train_test_data(all_data)

    scaler = None
    if standardize_column_names is not None and len(standardize_column_names) > 0:
        print("perform data standardisation on columns {}".format(standardize_column_names))
        train_data, scaler = standardize_data(train_data, column_names=standardize_column_names)
        test_data, _ = standardize_data(test_data, column_names=standardize_column_names, scaler=scaler)
        pred_data, _ = standardize_data(pred_data, column_names=standardize_column_names, scaler=scaler)

    encoder = None
    enc_column_names = []
    if categorical_column_names is not None and len(categorical_column_names) > 0:
        logger.info("encode categorical columns to integers, {}".format(categorical_column_names))
        train_data, enc_column_names, encoder = encode_categorical_features(train_data, categorical_column_names)
        test_data, _, _ = encode_categorical_features(test_data, column_names=categorical_column_names, encoder=encoder)
        pred_data, _, _ = encode_categorical_features(pred_data, column_names=categorical_column_names, encoder=encoder)


    # dealing with missing values
    train_data = fill_missing_values(train_data)
    test_data = fill_missing_values(test_data)
    pred_data = fill_missing_values(pred_data)

    logger.info("train data columns {}".format(train_data.columns))
    logger.info("test data columns {}".format(test_data.columns))
    logger.info("pred data columns {}".format(pred_data.columns))

    return train_data, test_data, pred_data, scaler, encoder, enc_column_names


def fill_missing_values(df):
    return df.fillna(0)


def restore_standardized_data(df, scaler):
    return scaler.inverse_transform(df)


def _split_train_test_data(all_data):
    if split_only_last_date:
        train_data = all_data[all_data['日期'] != all_data['日期'].max()]
        test_data = all_data[all_data['日期'] == all_data['日期'].max()]
    else:
        days = round((all_data['日期'].max() - all_data['日期'].min()).days * train_ratio)
        split_date = all_data['日期'].min() + timedelta(days=days)
        print("split date is {}".format(split_date))
        train_data = all_data[all_data['日期'] <= split_date]
        test_data = all_data[all_data['日期'] > split_date]
    return train_data, test_data




def get_sequential_data(train_file_name, pred_file_name, sequential_data_file_name=None, sequence_length=5,
                        use_roc_label=True, feature_top_n=10, label_name="", categorical_feature_names=None):

    all_feature_names = ["ma_5", "ma_10", "ma_20", "ma_60", "ma_120", "ma_240", "hhv_5", "hhv_10", "hhv_20",
                         "hhv_60", "hhv_120", "hhv_240", "llv_5", "llv_10", "llv_20", "llv_60", "llv_120",
                         "llv_240", "bias_6", "bias_12", "bias_24", "boll_upper_20", "boll_mid_20",
                         "boll_lower_20", "rsi_6", "rsi_12", "rsi_24", "wr_10", "wr_6", "mtm_12", "mtm_12_ma_6",
                         "k_9", "d_9", "j_9", "macd_dif", "macd_dea", "macd", "dmi_pdi", "dmi_mdi", "dmi_adx",
                         "dmi_adxr", "obv", "cci", "roc_12", "ma_6_roc_12", "bbi", "expma_12", "expma_50", "ar",
                         "br", "atr", "dma_dif", "dma", "emv", "maemv", "psy", "psyma", "asi", "asit", "mfi",
                         "mass", "mamass", "dpo", "madpo", "vr", "trix", "trma", "kc_upper", "kc_mid",
                         "kc_lower", "dc_upper", "dc_mid", "dc_lower", "open", "close", "high", "low", "roc",
                         "volume", "turnover_rate"]
    # top 20 feature
    roc_top_column_names = ['roc', 'bias_6', 'turnover_rate', 'emv', 'roc_12', 'bias_12', 'cci', 'j_9', 'dmi_mdi',
                            'wr_6', 'trma', 'ma_6_roc_12', 'vr', 'trix', 'dmi_adx', 'd_9', 'mass', 'mfi', 'volume',
                            'maemv', 'br', 'ar', 'mamass', 'k_9', 'obv', 'dmi_adxr', 'bias_24', 'rsi_24', 'rsi_6',
                            'dmi_pdi', 'asi', 'macd', 'rsi_12', 'dma', 'wr_10', 'asit', 'mtm_12', 'atr', 'mtm_12_ma_6',
                            'psyma', 'macd_dea', 'dpo', 'llv_240', 'macd_dif', 'dma_dif', 'madpo']
    close_top_column_names = ['close', 'roc', 'bias_6', 'wr_6', 'mtm_12', 'high', 'low', 'macd', 'turnover_rate',
                              'dmi_adx', 'emv', 'atr', 'mfi', 'mass', 'dmi_mdi', 'volume', 'mamass', 'roc_12', 'j_9',
                              'dmi_adxr', 'vr', 'dmi_pdi', 'ar', 'bias_24', 'maemv', 'bias_12', 'asi', 'br', 'k_9',
                              'd_9', 'trix', 'obv', 'wr_10', 'rsi_6', 'ma_6_roc_12', 'cci', 'trma', 'rsi_24',
                              'macd_dif', 'macd_dea', 'mtm_12_ma_6', 'rsi_12', 'asit', 'dma', 'psyma', 'dpo']
    if use_roc_label:
        standardize_column_names = roc_top_column_names
    else:
        standardize_column_names = close_top_column_names

    # top 20
    standardize_column_names = standardize_column_names[0:min(feature_top_n, len(standardize_column_names))]
    # standardize_column_names = ["close", "ma_5", 'turnover_rate',  "volume"]
    feature_columns = standardize_column_names

    if label_name is None or len(label_name) == 0:
        label_columns = ["label_roc"] if use_roc_label else ["label_close"]
    else:
        label_columns = [label_name]

    ext_columns = ["日期", "代码", "roc", "close", "label_roc", "label_close", "label_roi_1d", "label_roi_2d",
                   "label_roi_3d", "label_roi_5d", "label_roi_10d"]

    cat_columns = ["sym", "date_mon", "date_dow", "industry", "is_index_000001", "is_index_000016", "is_index_000300"]

    logger.info("sequential features {}, categorical features {},  label {}, ext columns {}".format(feature_columns, cat_columns, label_columns, ext_columns))

    if os.path.exists(sequential_data_file_name):
        sequential_data = joblib.load(sequential_data_file_name)
    else:
        load_from_db = 0
        dataset = load_data(train_file_name, pred_file_name,
                            standardize_column_names=standardize_column_names,
                            categorical_column_names=cat_columns)
        train_data, test_data, pred_data, scaler, encoder, enc_column_names = dataset

        train_data = train_data[train_data["日期"] > train_data["日期"].max() - timedelta(days=730)]
        print("train_data min date {}, max date {}".format(train_data["日期"].min(), train_data["日期"].max()))
        print("test_data min date {}, max date {}".format(test_data["日期"].min(), test_data["日期"].max()))
        print("pred_data min date {}, max date {}".format(pred_data["日期"].min(), pred_data["日期"].max()))

        print("convert training data into sequences. sequence length {}".format(sequence_length))
        train_data_seq, train_data_cat, train_data_y, train_data_ext = _generate_sequential_data(train_data, scaler,
                                                                                                 sequence_length,
                                                                                                 numerical_feature_columns=feature_columns,
                                                                                                 label_columns=label_columns,
                                                                                                 ext_columns=ext_columns,
                                                                                                 categorical_feature_columns=enc_column_names)
        print("convert testing data into sequences. sequence length {}".format(sequence_length))
        test_data_seq, test_data_cat, test_data_y, test_data_ext = _generate_sequential_data(test_data, scaler,
                                                                                             sequence_length,
                                                                                             numerical_feature_columns=feature_columns,
                                                                                             label_columns=label_columns,
                                                                                             ext_columns=ext_columns,
                                                                                             categorical_feature_columns=enc_column_names)

        print("convert prediction data into sequences. sequence length {}".format(sequence_length))
        pred_data_seq, pred_data_cat, _, pred_data_ext = _generate_sequential_data(pred_data, scaler, sequence_length,
                                                                                   numerical_feature_columns=feature_columns,
                                                                                   label_columns=None,
                                                                                   ext_columns=ext_columns,
                                                                                   categorical_feature_columns=enc_column_names)

        sequential_data = (
        train_data_seq, train_data_cat, train_data_y, train_data_ext, test_data_seq, test_data_cat, test_data_y,
        test_data_ext, pred_data_seq, pred_data_cat, pred_data_ext)
        print("saving sequential data to {}.".format(sequential_data_file_name))
        # pickle.dump(sequential_data, open(sequential_data_file_name, "wb"))
        joblib.dump(sequential_data, sequential_data_file_name)
        print("data saved to {}".format(sequential_data_file_name))
    return sequential_data





def fill_ext_with_predictions(ext, score, use_roc_label, use_sqrt_roc=False, use_log_close=False):
    close = ext["close"].values.astype(float)
    if use_roc_label:
        if use_sqrt_roc:
            score = np.array([abs(p)*p for p in score])
        label_roc_pred = score
        label_close_pred = np.add(close, np.multiply(close, label_roc_pred/100))
    else:
        if use_log_close:
            score = np.array([math.exp(p) for p in score])
        label_close_pred = score
        label_roc_pred = np.divide(np.subtract(score, close), close) * 100

    ext["label_roc_pred"] = label_roc_pred
    ext["label_close_pred"] = label_close_pred
    return ext


def compute_precision_recall_updated(label, pred):
    print("#### label mse {}".format(mean_squared_error(label, pred)))

    auc_curve = []
    for th in range(-10, 10):
        label_th = th
        tp = len(label[np.logical_and(label >= label_th, pred >= th)])
        pp = len(pred[pred >= th])
        p = len(label[label >= th])
        n = len(label[label < th])
        fp = len(label[np.logical_and(label < label_th, pred >= th)])

        precision = tp / pp if pp > 0 else 0
        recall = tp / p if p > 0 else 0
        p_ratio = p/len(label)
        pp_ratio = pp / len(pred)
        tpr = recall
        fpr = fp / n if n > 0 else 0
        loss_p = len(label[np.logical_and(label < -label_th, pred >= th)]) / pp if pp > 0 else 0
        logger.info("threshold {}, precision {}, loss probability {}, recall {}, p ratio {}, pp ratio {}, pred len {}, label len {}, p cnt {}, pp cnt {}".format(th, precision, loss_p, recall, p_ratio,
                                                                                      pp_ratio, len(pred), len(label), p, pp))
        auc_curve.append([th, precision, recall, tpr, fpr, pp_ratio])
    auc_curve = np.array(auc_curve)
    # print(auc_curve)



def compute_precision_recall(ext):

    label_roc = ext["label_roc"].values.astype(float)
    label_close = ext["label_close"].values.astype(float)
    label_roc_pred = ext["label_roc_pred"].values.astype(float)
    label_close_pred = ext["label_close_pred"].values.astype(float)
    print("#### roc pred {} ... {}\n roc label {} ... {}\n".format(label_roc_pred[0:10], label_roc_pred[-10:],
                                                                                  label_roc[0:10], label_roc[-10:]))
    print("#### roc mse {}, close mse {}".format(mean_squared_error(label_roc, label_roc_pred),
                                              mean_squared_error(label_close, label_close_pred)))

    auc_curve = []
    for th in range(-10,10):
        label_th = th
        pred = label_roc_pred
        label = label_roc
        tp = len(label[np.logical_and(label >= label_th, pred >= th)])
        pp = len(pred[pred >= th])
        p = len(label[label >= th])
        n = len(label[label < th])
        fp = len(label[np.logical_and(label < label_th, pred >= th)])

        precision = tp / pp if pp > 0 else 0
        recall = tp / p if p > 0 else 0
        p_ratio = p/len(label)
        pp_ratio = pp / len(pred)
        tpr = recall
        fpr = fp / n if n > 0 else 0
        loss_p = len(label[np.logical_and(label < -label_th, pred >= th)]) / pp if pp > 0 else 0
        logger.info("roc threshold {}, precision {}, loss probability {}, recall {}, p ratio {}, pp ratio {}, pred len {}, label len {}, p cnt {}, pp cnt {}".format(th, precision, loss_p, recall, p_ratio,
                                                                                      pp_ratio, len(pred), len(label), p, pp))
        auc_curve.append([th, precision, recall, tpr, fpr, pp_ratio])
    auc_curve = np.array(auc_curve)
    # print(auc_curve)
