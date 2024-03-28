import sys, os
parent_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parent_dir)


import akshare as ak
from datetime import datetime

from utils.log_util import get_logger
from base_data import BaseData

logger = get_logger(__name__)

'''
A股股票指数数据
实时行情数据
接口: stock_zh_index_spot
目标地址: http://vip.stock.finance.sina.com.cn/mkt/#hs_s
描述: 新浪财经-中国股票指数数据
限量: 单次返回所有指数的实时行情数据
'''


class StockZhIndex(BaseData):
    def __init__(self):
        super().__init__()

    def get_table_name(self):
        return "stock_zh_index"

    def get_df_schema(self):
        df = ak.stock_zh_index_spot_sina()
        df_schema = df[["代码", "名称"]]
        return df_schema

    def before_retrieve_data(self):
        pass
        # self.delete_records(self.table_name, "ds='{}';".format(self.ds))



if __name__ == '__main__':
    stock_zh_a = StockZhIndex()
    stock_zh_a.retrieve_data()
    stock_zh_a.clean_up_history()
