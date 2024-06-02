import sys, os
parent_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parent_dir)


import akshare as ak
import time
from datetime import datetime, timedelta
import pandas as pd
from utils import stock_zh_a_util
from utils.log_util import get_logger
from base_data import BaseData

logger = get_logger(__name__)
from utils.stock_zh_a_util import is_trade_date

'''
接口: stock_individual_fund_flow_rank

目标地址: http://data.eastmoney.com/zjlx/detail.html

描述: 东方财富网-数据中心-资金流向-排名

限量: 单次获取指定类型的个股资金流排名数据

indicator="今日"
'''


class StockIndividualFundFlowRankSpot(BaseData):
    def __init__(self):
        super().__init__()

    def get_table_name(self):
        return "stock_individual_fund_flow_rank_spot"

    def before_retrieve_data(self):
        pass

    def get_df_schema(self):
        df = ak.stock_individual_fund_flow_rank(indicator="今日")
        df.rename(columns={"今日主力净流入-净额": "今日主力净流入_净额", "今日主力净流入-净占比": "今日主力净流入_净占比",
                           "今日超大单净流入-净额": "今日超大单净流入_净额",
                           "今日超大单净流入-净占比": "今日超大单净流入_净占比", "今日大单净流入-净额": "今日大单净流入_净额",
                           "今日大单净流入-净占比": "今日大单净流入_净占比",
                           "今日中单净流入-净额": "今日中单净流入_净额", "今日中单净流入-净占比": "今日中单净流入_净占比",
                           "今日小单净流入-净额": "今日小单净流入_净额",
                           "今日小单净流入-净占比": "今日小单净流入_净占比"}, inplace=True)
        return df




if __name__ == '__main__':
    ds = sys.argv[1]
    logger.info("execute task on ds {}".format(ds))
    if not is_trade_date(ds):
        logger.info(f"{ds} is not trade date. task exits.")
        exit(os.EX_OK)
    data = StockIndividualFundFlowRankSpot()
    data.retrieve_data()
    data.clean_up_history()

