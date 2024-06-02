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
利润表
接口: stock_lrb_em
目标地址: http://data.eastmoney.com/bbsj/202003/lrb.html

描述: 东方财富-数据中心-年报季报-业绩快报-利润表

限量: 单次获取指定 date 的利润表数据
'''

class StockLrbEm(BaseData):
    def __init__(self):
        super().__init__()

    def get_table_name(self):
        return "stock_lrb_em"

    def get_df_schema(self):
        stock_individual_info_em_df = ak.stock_individual_info_em(symbol="000001")

        df = ak.stock_lrb_em(date="20240331")
        df.rename(columns={"营业总支出-营业支出": "营业总支出_营业支出",
                           "营业总支出-销售费用": "营业总支出_销售费用",
                           "营业总支出-管理费用": "营业总支出_管理费用",
                           "营业总支出-财务费用": "营业总支出_财务费用",
                           "营业总支出-营业总支出": "营业总支出_营业总支出"}, inplace=True)
        return df

    def before_retrieve_data(self):
        pass


if __name__ == '__main__':
    ds = sys.argv[1]
    logger.info("execute task on ds {}".format(ds))

    if not is_trade_date(ds):
        logger.info(f"{ds} is not trade date. task exits.")
        exit(os.EX_OK)

    spot_em = StockLrbEm()
    spot_em.set_ds(ds)
    spot_em.retrieve_data()