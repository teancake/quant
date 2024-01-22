import sys, os
parent_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parent_dir)


import akshare as ak
from datetime import datetime

from utils.log_util import get_logger
from base_data import BaseData

logger = get_logger(__name__)

'''
沪深京 A 股
接口: stock_zh_a_spot_em

目标地址: http://quote.eastmoney.com/center/gridlist.html#hs_a_board

描述: 东方财富网-沪深京 A 股-实时行情数据

限量: 单次返回所有沪深京 A 股上市公司的实时行情数据

存储：只保存A股的代码
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

    def before_retrieve_data(self):
        pass
        # self.delete_records(self.table_name, "ds='{}';".format(self.ds))


if __name__ == '__main__':
    stock_zh_a = StockZhA()
    stock_zh_a.retrieve_data()
    stock_zh_a.clean_up_history()




