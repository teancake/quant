import sys, os
import akshare as ak
from datetime import datetime
import time
parent_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parent_dir)

from utils.stock_zh_a_util import is_trade_date, is_backfill, get_stock_list
from utils.log_util import get_logger
from base_data import BaseData

logger = get_logger(__name__)

'''
主营介绍-同花顺
接口: stock_zyjs_ths

目标地址: http://basic.10jqka.com.cn/new/000066/operate.html

描述: 同花顺-主营介绍

限量: 单次返回所有数据
'''

class StockZyjsThs(BaseData):
    def __init__(self, ds, symbol):
        super().__init__()
        self.ds = ds
        self.symbol = symbol
        self.symbol_column_name = "股票代码"

    def set_params(self, **kwargs):
        self.params = kwargs

    def get_downloaded_symbols(self):
        if not self.table_exists():
            return []
        sql = f"SELECT distinct {self.symbol_column_name} from {self.table_name}  where ds = '{self.ds}'"
        recs = self.db.run_sql(sql)
        return [rec[0] for rec in recs]

    def before_retrieve_data(self):
        pass

    def get_table_name(self):
        return "stock_zyjs_ths"

    def get_df_schema(self):
        df = ak.stock_zyjs_ths(symbol=self.symbol)
        return df


class DataHelper:

    def rate_limiter_sleep(self, timer_start, loops_per_second=1.0):
        loop_time_second = 1.0 / loops_per_second
        dt = time.time() - timer_start
        if dt < loop_time_second:
            logger.info(f"dt is {dt}, sleep {loop_time_second - dt} seconds")
            time.sleep(loop_time_second - dt)


    def get_data_for_all_symbols(self, ds: str):
        symbol_list = get_stock_list()
        downloaded_symbols = StockZyjsThs(ds=ds, symbol="").get_downloaded_symbols()
        for symbol in symbol_list:
            timer_start = time.time()
            try:
                logger.info(f"retrieving symbol {symbol} on ds {ds}.")
                if symbol in downloaded_symbols:
                    logger.info(f"symbol {symbol} already downloaded. skip downloading.")
                    continue
                data = StockZyjsThs(ds=ds, symbol=symbol)
                data.retrieve_data()
                logger.info("symbol {} done".format(symbol))
            except Exception as e:
                logger.warning(f"exception occurred {e}")
            self.rate_limiter_sleep(timer_start, loops_per_second=1)


if __name__ == '__main__':
    ds = sys.argv[1]
    logger.info("execute task on ds {}".format(ds))
    backfill = is_backfill(ds)
    if backfill:
        logger.info("backfill, get data for all symbols")
        DataHelper().get_data_for_all_symbols(ds)
    else:
        logger.info("not backfill, no data will be downloaded")
