import sys, os
parent_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parent_dir)


import akshare as ak
from datetime import datetime, timedelta

from utils.log_util import get_logger
from utils.stock_zh_a_util import is_trade_date

from base_data import BaseData

logger = get_logger(__name__)

'''
A股股票指数数据
实时行情数据-东财
接口: stock_zh_index_spot_em

目标地址: https://quote.eastmoney.com/center/gridlist.html#index_sz

描述: 东方财富网-行情中心-沪深京指数

限量: 单次返回所有指数的实时行情数据


'''


class StockZhIndexSpotEm(BaseData):
    def __init__(self, ds):
        super().__init__(ds)

    def get_table_name(self):
        return "stock_zh_index_spot_em"

    def get_df_schema(self):
        df = ak.stock_zh_index_spot_em(symbol="上证系列指数")
        df_schema = df
        return df_schema

    def before_retrieve_data(self):
        pass




if __name__ == '__main__':
    ds = sys.argv[1]
    logger.info("execute task on ds {}".format(ds))
    if not is_trade_date(ds):
        logger.info(f"{ds} is not trade date. task exits.")
        exit(os.EX_OK)

    spot_em = StockZhIndexSpotEm(ds)
    spot_em.retrieve_data()
    spot_em.clean_up_history(lifecycle=60)