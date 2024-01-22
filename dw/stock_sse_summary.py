import sys, os
parent_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parent_dir)


import akshare as ak
from datetime import datetime

from utils.log_util import get_logger
from base_data import BaseData

logger = get_logger(__name__)

'''
股票市场总貌
上海证券交易所
接口: stock_sse_summary

目标地址: http://www.sse.com.cn/market/stockdata/statistic/

描述: 上海证券交易所-股票数据总貌

限量: 单次返回最近交易日的股票数据总貌(当前交易日的数据需要交易所收盘后统计)
'''


class StockZhA(BaseData):
    def __init__(self):
        super().__init__()

    def get_table_name(self):
        return "stock_zh_a"

    def get_df_schema(self):
        df = ak.stock_zh_a_spot_em()
        df_schema = df[["序号", "代码", "名称"]]
        return df_schema


if __name__ == '__main__':
    stock_zh_a = StockZhA()
    stock_zh_a.retrieve_data()
