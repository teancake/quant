import sys, os
parent_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parent_dir)


import random
import traceback

import akshare as ak
from datetime import datetime
import time
import pandas as pd
import sys

from utils.log_util import get_logger
from utils.stock_zh_a_util import is_trade_date
from base_data import BaseData

logger = get_logger(__name__)

'''
中证指数成份股权重
接口: index_stock_cons_weight_csindex

目标地址: http://www.csindex.com.cn/zh-CN/indices/index-detail/000300

描述: 中证指数网站-成份股权重
'''


class IndexStockCons(BaseData):
    def __init__(self):
        super().__init__()

    def get_table_name(self):
        return "index_stock_cons_weight_csindex"

    def get_df_schema(self):
        # 目前只考虑 上证指数、上证50、 沪深300、深证成指、中证500
        index_symbols = ["000001", "000016", "000300", "399001", "399905"]
        df_list = []
        for symbol in index_symbols:
            try:
                df = self.get_single_df(symbol)
                df_list.append(df)
                time.sleep(random.randint(3, 8))
            except Exception as e:
                logger.error("exception occurred getting data for {}, exception {}".format(symbol, traceback.format_exc()))

        df_schema = pd.concat(df_list)
        return df_schema


    def before_retrieve_data(self):
        pass
        # self.delete_records(self.table_name, "ds='{}';".format(self.ds))


    def get_single_df(self, index_symbol):
        logger.info("retrieve data for index symbol {}".format(index_symbol))
        df = ak.index_stock_cons_weight_csindex(symbol=index_symbol)
        logger.info("data retrieved, number of rows {}".format(df.shape[0]))
        return df

if __name__ == '__main__':
    ds = sys.argv[1]
    logger.info("execute task on ds {}".format(ds))
    if not is_trade_date(ds):
        logger.info(f"{ds} is not trade date. task exits.")
        exit(os.EX_OK)

    index_stock_cons = IndexStockCons()
    index_stock_cons.set_ds(ds)
    index_stock_cons.retrieve_data()




