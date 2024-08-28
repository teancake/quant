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
    return list(stock_list)

def get_securities_position():
    results = StarrocksDbUtil().run_sql(f"""select 代码 from ads_stock_zh_a_position where position > 0""")
    results = [item[0] for item in results]
    stock_list = set(get_securities_list()) & set(results)
    return list(stock_list)

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


def get_zh_securities_data(symbol, ds, start_date, security_type="stock_zh_a", adjust="hfq", period="daily", yf_compatible=False):
    table_name_map = {"stock_zh_a": "dwd_stock_zh_a_hist_df",
                      "fund_etf": "dwd_fund_etf_hist_em_df",
                      "fund_lof": "dwd_fund_lof_hist_em_df",
                      "all": "dwd_security_hist_em_df"}
    # type_cond = "" if security_type is None else f" and type='{security_type}' "
    adjust_cond = "" if adjust is None else f" and adjust='{adjust}' "
    period_cond = "" if period is None else f" and period='{period}' "
    start_date_cond = "" if period is None else f" and 日期 >='{start_date}' "
    if symbol is None or symbol == "all":
        symbol_cond = ""
    elif isinstance(symbol, list):
        in_str = "('" + "', '".join(symbol) + "')"
        symbol_cond = f" and 代码 in {in_str} "
    else:
        symbol_cond = f" and 代码 = '{symbol}' "

    where_cond = f" WHERE ds='{ds}' {adjust_cond} {period_cond} {start_date_cond} {symbol_cond}"
    sql = f"SELECT * FROM {table_name_map[security_type]} {where_cond}"
    results = StarrocksDbUtil().run_sql(sql)
    df = pd.DataFrame(results)

    if df.empty:
        return df

    df["日期"] = pd.to_datetime(df["日期"])
    df.set_index(["日期", "代码"], inplace=True)
    df = df.sort_index()
    if yf_compatible:
        df = df.rename(columns={"开盘": "open", "收盘": "close", "最高": "high", "最低": "low",
                                "成交量": "volume", "成交额": "amount"})
        df.index.names = ["date", "ticker"]
        df = df[["open", "high", "low", "close", "volume", "amount"]]
    return df


def get_fund_etf_data(symbol, ds, start_date, adjust="hfq", period="daily", yf_compatible=False):
    return get_zh_securities_data(symbol, ds, start_date, security_type="fund_etf", adjust=adjust, period=period, yf_compatible=yf_compatible)


def get_fund_lof_data(symbol, ds, start_date, adjust="hfq", period="daily", yf_compatible=False):
    return get_zh_securities_data(symbol, ds, start_date, security_type="fund_lof", adjust=adjust, period=period, yf_compatible=yf_compatible)


def get_stock_data(symbol, ds, start_date, adjust="hfq", period="daily", yf_compatible=False):
    return get_zh_securities_data(symbol, ds, start_date, security_type="stock_zh_a", adjust=adjust, period=period, yf_compatible=yf_compatible)

def get_stock_map():
    ds = (datetime.now() - timedelta(days=7)).strftime("%Y%m%d")
    results = DbUtil().run_sql("SELECT distinct 代码, 名称 from stock_zh_a where ds >= {} order by 代码".format(ds))
    return {item[0]: item[1] for item in results}

def get_securities_map():
    results = StarrocksDbUtil().run_sql("SELECT distinct 代码, 名称, type as security_type from dwd_security_spot_em_di where ds in (select max(ds) from dwd_security_spot_em_di) order by type desc, 代码 asc")
    return {item[0]: item[1] for item in results}

def get_securities_list():
    return list(get_securities_map().keys())


def get_trade_dates():
    ds = (datetime.now() - timedelta(days=7)).strftime("%Y%m%d")
    results = DbUtil().run_sql("SELECT distinct trade_date from stock_zh_a_trade_date where ds >= {}".format(ds))
    return [item[0] for item in results]

def get_stock_list():
    return sorted(list(get_stock_map().keys()))

def get_normal_stock_list():
    ds = (datetime.now() - timedelta(days=7)).strftime("%Y%m%d")
    results = DbUtil().run_sql("SELECT distinct 代码 from stock_zh_a where ds >= {} AND 名称 not regexp 'ST|退|PT' and 代码 regexp '^(600|601|603)' order by 代码".format(ds))
    return [item[0] for item in results]


def get_index_map():
    ds = (datetime.now() - timedelta(days=7)).strftime("%Y%m%d")
    results = DbUtil().run_sql("SELECT distinct 代码, 名称 from stock_zh_index where ds >= {} order by 代码".format(ds))
    return {item[0]: item[1] for item in results}


def get_index_list():
    return sorted(["sh000001", "sh000016", "sh000300", "sh000903", "sh000905", "sh000906", "sh000015", "sh000012"])
    # return sorted(list(get_index_map().keys()))


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
