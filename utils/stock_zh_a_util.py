from datetime import datetime, timedelta

import pandas as pd

from utils.db_util import DbUtil
from utils.starrocks_db_util import StarrocksDbUtil
from utils.config_util import get_data_config
from utils.log_util import get_logger

logger = get_logger(__name__)

def get_sw_industries():
    results = StarrocksDbUtil().run_sql(f"select 行业名称 from dwd_sw_index_first_info_df where ds in (select max(ds) from dwd_sw_index_first_info_df)")
    return [item[0] for item in results]

def get_stock_position():
    results = StarrocksDbUtil().run_sql(f"""select 代码 from ads_stock_zh_a_position where position > 0""")
    results = [item[0] for item in results]
    stock_list = set(get_stock_list()) & set(results)
    return stock_list


def get_list_date():
    sql = f"select distinct 代码 as ticker, 上市时间 as list_date from dwd_stock_individual_info_em_df where ds in (select max(ds) from dwd_stock_individual_info_em_df) and length(上市时间)=8;"
    results = StarrocksDbUtil().run_sql(sql)
    df = pd.DataFrame(results)
    df.list_date = pd.to_datetime(df.list_date, format='%Y%m%d')
    df.set_index("ticker", inplace=True)
    return df


def get_benchmark_data(symbol, ds, start_date, yf_compatible=False):
    results = StarrocksDbUtil().run_sql(f"SELECT * FROM dwd_index_zh_a_hist_df WHERE ds='{ds}' and 代码='{symbol}' and 日期 >= '{start_date}'")
    df = pd.DataFrame(results)
    if not df.empty:
        df.set_index("日期", inplace=True)
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
    if yf_compatible:
        df = df.rename(columns={"开盘": "open", "收盘": "close", "最高": "high", "最低": "low",
                                "成交量": "volume", "成交额": "amount"})
        df.index.name = "date"
        df = df[["open", "close", "high", "low", "volume", "amount"]]
    return df


def get_fund_etf_data(symbol, ds, start_date, yf_compatible=False):
    table_name = "dwd_fund_etf_hist_em_df"
    symbol_field = "symbol"
    if symbol == "all":
        results = StarrocksDbUtil().run_sql(f"SELECT * FROM {table_name} WHERE ds='{ds}' and 日期 >= '{start_date}'")
    else:
        results = StarrocksDbUtil().run_sql(f"SELECT * FROM {table_name} WHERE ds='{ds}' and {symbol_field}='{symbol}' and 日期 >= '{start_date}'")
    df = pd.DataFrame(results)
    if not df.empty:
        df.set_index("日期", inplace=True)
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        df["代码"] = df[symbol_field]
    if yf_compatible:
        df = df.rename(columns={"开盘": "open", "收盘": "close", "最高": "high", "最低": "low",
                                "成交量": "volume", "成交额": "amount"})
        df.index.name = "date"
        df = df[["open", "close", "high", "low", "volume", "amount"]]
    return df


def get_stock_data(symbol, ds, start_date, adjust="hfq", period="daily", yf_compatible=False):
    if symbol == "all":
        results = StarrocksDbUtil().run_sql(f"SELECT * FROM dwd_stock_zh_a_hist_df WHERE ds='{ds}' and adjust='{adjust}' and period='{period}' and 日期 >= '{start_date}'")
    else:
        results = StarrocksDbUtil().run_sql(f"SELECT * FROM dwd_stock_zh_a_hist_df WHERE ds='{ds}' and adjust='{adjust}' and period='{period}' and 代码='{symbol}' and 日期 >= '{start_date}'")
    df = pd.DataFrame(results)
    if not df.empty:
        df.set_index("日期", inplace=True)
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
    if yf_compatible:
        df = df.rename(columns={"开盘": "open", "收盘": "close", "最高": "high", "最低": "low",
                                "成交量": "volume", "成交额": "amount", "代码": "symbol"})
        df.index.name = "date"
        df = df[["open", "close", "high", "low", "volume", "amount", "symbol"]]
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
