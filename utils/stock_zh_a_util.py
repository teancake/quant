from datetime import datetime, timedelta

import pandas as pd

from utils.db_util import DbUtil
from utils.starrocks_db_util import StarrocksDbUtil
from utils.config_util import get_data_config
from utils.log_util import get_logger

logger = get_logger(__name__)

def get_benchmark_data(symbol, ds, start_date):
    results = StarrocksDbUtil().run_sql(f"SELECT * FROM dwd_index_zh_a_hist_df WHERE ds='{ds}' and 代码='{symbol}' and 日期 >= '{start_date}'")
    df = pd.DataFrame(results)
    if not df.empty:
        df.set_index("日期", inplace=True)
        df.index = pd.to_datetime(df.index)
    return df


def get_stock_data(symbol, ds, start_date):
    if symbol == "all":
        results = StarrocksDbUtil().run_sql(f"SELECT * FROM dwd_stock_zh_a_hist_df WHERE ds='{ds}' and 日期 >= '{start_date}'")
    else:
        results = StarrocksDbUtil().run_sql(f"SELECT * FROM dwd_stock_zh_a_hist_df WHERE ds='{ds}' and 代码='{symbol}' and 日期 >= '{start_date}'")
    df = pd.DataFrame(results)
    if not df.empty:
        df.set_index("日期", inplace=True)
        df.index = pd.to_datetime(df.index)
    return df


def get_stock_map():
    ds = (datetime.now() - timedelta(days=7)).strftime("%Y%m%d")
    results = DbUtil().run_sql("SELECT distinct 代码, 名称 from stock_zh_a where ds >= {} order by 代码".format(ds))
    return {item[0]: item[1] for item in results}

def get_trade_dates():
    ds = (datetime.now() - timedelta(days=7)).strftime("%Y%m%d")
    results = DbUtil().run_sql("SELECT distinct trade_date from stock_zh_a_trade_date where ds >= {}".format(ds))
    return [item[0] for item in results]

def get_stock_list():
    return sorted(list(get_stock_map().keys()))


def get_index_map():
    ds = (datetime.now() - timedelta(days=7)).strftime("%Y%m%d")
    results = DbUtil().run_sql("SELECT distinct 代码, 名称 from stock_zh_index where ds >= {} order by 代码".format(ds))
    return {item[0]: item[1] for item in results}


def get_index_list():
    return sorted(list(get_index_map().keys()))


def is_trade_date(ds: str):
    ds = datetime.strptime(ds, "%Y%m%d").date()
    return ds in get_trade_dates()


def get_fund_etf_map():
    ds = (datetime.now() - timedelta(days=7)).strftime("%Y%m%d")
    results = DbUtil().run_sql("SELECT distinct 代码, 名称 from fund_etf_spot_em where ds >= {} order by 代码".format(ds))
    return {item[0]: item[1] for item in results}


def get_fund_lof_map():
    ds = (datetime.now() - timedelta(days=7)).strftime("%Y%m%d")
    results = DbUtil().run_sql("SELECT distinct 代码, 名称 from fund_lof_spot_em where ds >= {} order by 代码".format(ds))
    return {item[0]: item[1] for item in results}


def get_fund_etf_list():
    etf_list = set(get_fund_etf_map().keys())
    etf_list.add("511010")
    return sorted(etf_list)


def get_fund_lof_list():
    return sorted(list(get_fund_lof_map().keys()))


def is_backfill(ds: str):
    backfill_weekday = get_data_config()["backfill"]
    weekday = datetime.strptime(ds, "%Y%m%d").isoweekday()
    backfill = True if str(weekday) in backfill_weekday else False
    logger.info(f"back fill conf {backfill_weekday}, weekday {weekday}, backfill {backfill}")
    return backfill
