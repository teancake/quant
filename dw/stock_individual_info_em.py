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
import random
import traceback
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

    def get_downloaded_symbols(self):
        symbol_col_name = "股票代码"
        sql = f"SELECT distinct {symbol_col_name} from {self.get_table_name()}  where ds = '{self.ds}'"
        recs = self.db.run_sql(sql)
        return [rec[0] for rec in recs]


class DataHelper:

    def rate_limiter_sleep(self, timer_start, loops_per_second_min, loops_per_second_max):
        loop_time_second_min = 1.0 / loops_per_second_max
        loop_time_second_max = 1.0 / loops_per_second_min
        dt = time.time() - timer_start
        if dt < loop_time_second_min:
            random_time = random.random() * (loop_time_second_max - loop_time_second_min) + loop_time_second_min - dt
            logger.info(f"dt is {dt} less than minimum loop time {loop_time_second_min}, sleep {random_time} seconds")
            time.sleep(random_time)

    def get_all_symbols(self, ds):
        data = StockIndividualInfoEm()
        data.set_ds(ds)
        symbol_list = stock_zh_a_util.get_stock_list()
        downloaded_symbols = data.get_downloaded_symbols()
        for symbol in symbol_list:
            timer_start = time.time()
            try:
                logger.info("process symbol {}".format(symbol))
                if symbol in downloaded_symbols:
                    logger.info(f"symbol {symbol} already downloaded. skip downloading.")
                    continue
                data.set_symbol(symbol)
                data.retrieve_data()
                logger.info("symbol {} done".format(symbol))
            except Exception:
                logger.error(f"exception occurred while processing symbol {symbol}")
                logger.error(traceback.format_exc())
            self.rate_limiter_sleep(timer_start, loops_per_second_min=1, loops_per_second_max=2)

        data.clean_up_history(lifecycle=30)




if __name__ == '__main__':
    ds = sys.argv[1]
    logger.info("execute task on ds {}".format(ds))
    DataHelper().get_all_symbols(ds)
