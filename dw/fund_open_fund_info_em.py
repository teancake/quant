import sys, os
parent_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parent_dir)

import akshare as ak
import pandas as pd
from datetime import datetime, timedelta


from utils.log_util import get_logger
from base_data import BaseData
from utils import stock_zh_a_util
from utils.stock_zh_a_util import is_trade_date

logger = get_logger(__name__)

import time
import sys
import re
import traceback



'''
开放式基金-历史数据
接口: fund_open_fund_info_em

目标地址: http://fund.eastmoney.com/pingzhongdata/710001.js

描述: 东方财富网-天天基金网-基金数据-具体基金信息

限量: 单次返回当前时刻所有历史数据, 在查询基金数据的时候注意基金前后端问题

'''


class FundOpenFundInfoEm(BaseData):
    def __init__(self, symbol=None):
        self.symbol = symbol
        super().__init__()

    def set_symbol(self, symbol):
        self.symbol = symbol

    def get_table_name(self):
        return "fund_open_fund_info_em"

    def before_retrieve_data(self):
        pass

    def get_df_schema(self):
        df = ak.fund_open_fund_info_em(self.symbol, indicator="单位净值走势")
        logger.info("data retrieved, number of rows {}".format(df.shape[0]))
        df.insert(0, "symbol", symbol)
        return df


if __name__ == '__main__':
    ds = sys.argv[1]
    logger.info("execute task on ds {}".format(ds))
    if not is_trade_date(ds):
        logger.info(f"{ds} is not trade date. task exits.")
        exit(os.EX_OK)

    data = FundOpenFundInfoEm()
    data.set_ds(ds)
    symbol_list = stock_zh_a_util.get_fund_lof_list()
    symbol_list.extend(stock_zh_a_util.get_fund_etf_list())
    for symbol in symbol_list:
        logger.info("process symbol {}".format(symbol))
        data.set_symbol(symbol)
        data.retrieve_data()
        logger.info("symbol {} done".format(symbol))
        time.sleep(0.5)

    data.clean_up_history(lifecycle=15)

