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
个股信息查询
接口: stock_individual_info_em
目标地址: http://quote.eastmoney.com/concept/sh603777.html?from=classic
描述: 东方财富-个股-股票信息
限量: 单次返回指定 symbol 的个股信息
'''


class StockIndividualInfoEm(BaseData):
    def __init__(self, symbol=None):
        self.symbol = symbol
        super().__init__()

    def set_symbol(self, symbol):
        self.symbol = symbol

    def get_table_name(self):
        return "stock_individual_info_em"

    def before_retrieve_data(self):
        pass

    def get_df_schema(self):
        df = ak.stock_individual_info_em(symbol=self.symbol)
        df = df.astype(str)
        keys = ["股票代码", "股票简称", "行业", "上市时间", "总股本", "流通股", "总市值", "流通市值"]
        temp_dict = {}
        for key in keys:
            temp_dict[key] = df.loc[df["item"] == key]["value"].values.tolist()
        print(temp_dict)
        return pd.DataFrame(temp_dict)




if __name__ == '__main__':
    ds = sys.argv[1]
    logger.info("execute task on ds {}".format(ds))
    if not is_trade_date(ds):
        logger.info(f"{ds} is not trade date. task exits.")
        exit(os.EX_OK)
    data = StockIndividualInfoEm()
    symbol_list = stock_zh_a_util.get_stock_list()
    for symbol in symbol_list:
        logger.info("process symbol {}".format(symbol))
        data.set_symbol(symbol)
        data.retrieve_data()
        logger.info("symbol {} done".format(symbol))
        time.sleep(0.5)
    data.clean_up_history()
