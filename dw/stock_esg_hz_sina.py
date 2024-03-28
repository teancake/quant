import sys, os

import pandas as pd

parent_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parent_dir)


import akshare as ak
from datetime import datetime, timedelta

from utils.log_util import get_logger
from utils.stock_zh_a_util import is_trade_date

from base_data import BaseData

logger = get_logger(__name__)

import re

'''
华证指数
接口: stock_esg_hz_sina

目标地址: https://finance.sina.com.cn/esg/grade.shtml

描述: 新浪财经-ESG评级中心-ESG评级-华证指数

限量: 单次返回所有数据
'''

class StockEsgHzSina(BaseData):
    def __init__(self):
        super().__init__()

    def get_table_name(self):
        return "stock_esg_hz_sina"

    def get_df_schema(self):
        df = ak.stock_esg_hz_sina()
        df.insert(loc=1, column='symbol', value=df["股票代码"].apply(lambda x: re.sub(r"\D", '', x)))
        return df

    def before_retrieve_data(self):
        pass


if __name__ == '__main__':
    ds = sys.argv[1]
    logger.info("execute task on ds {}".format(ds))

    if not is_trade_date(ds):
        logger.info(f"{ds} is not trade date. task exits.")
        exit(os.EX_OK)

    spot_em = StockEsgHzSina()
    spot_em.set_ds(ds)
    spot_em.retrieve_data()