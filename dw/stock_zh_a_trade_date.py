import sys, os
parent_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parent_dir)


import akshare as ak
from datetime import datetime

from utils.log_util import get_logger
from base_data import BaseData

logger = get_logger(__name__)


class StockZhATradeDate(BaseData):
    def __init__(self):
        super().__init__()

    def get_table_name(self):
        return "stock_zh_a_trade_date"

    def get_df_schema(self):
        df = ak.tool_trade_date_hist_sina()
        df_schema = df[["trade_date"]]
        return df_schema

    def before_retrieve_data(self):
        pass
        # self.delete_records(self.table_name, "ds='{}';".format(self.ds))


if __name__ == '__main__':
    stock_zh_a = StockZhATradeDate()
    stock_zh_a.retrieve_data()
    stock_zh_a.clean_up_history()




