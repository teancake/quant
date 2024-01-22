from datetime import datetime, timedelta
from utils.db_util import DbUtil


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
