import sys, os
parent_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parent_dir)


import akshare as ak
import pandas as pd
from datetime import datetime, timedelta


from utils.log_util import get_logger
from base_data import BaseData, BaseDataHelper
from utils import stock_zh_a_util
from utils.stock_zh_a_util import is_trade_date, is_backfill, get_stock_list

logger = get_logger(__name__)

import time
import sys

'''
历史行情数据-东财
接口: stock_zh_a_hist

目标地址: https://quote.eastmoney.com/concept/sh603777.html?from=classic(示例)

描述: 东方财富-沪深京 A 股日频率数据; 历史数据按日频率更新, 当日收盘价请在收盘后获取

限量: 单次返回指定沪深京 A 股上市公司、指定周期和指定日期间的历史行情日频率数据

'''


class StockZhAHist(BaseData):
    def __init__(self, ds, symbol=None, backfill=False, period=None, adjust=None):
        super().__init__()
        self.ds = ds
        self.symbol = symbol
        self.backfill = backfill
        self.period = period
        self.adjust = adjust

    def get_table_name(self):
        return "stock_zh_a_hist"

    def before_retrieve_data(self):
        pass

    def get_df_schema(self):
        logger.info(f"retrieving symbol {self.symbol} on ds {self.ds} for period {self.period}, adjust {self.adjust}.")
        df = self.get_single_df(self.symbol, self.period, self.adjust, self.ds, self.backfill)
        return df

    def get_single_df(self, symbol, period, adjust, ds, backfill):
        # restrict end data to ds
        if backfill:
            df = ak.stock_zh_a_hist(symbol=symbol, period=period, adjust=adjust, end_date=ds, timeout=60)
        else:
            start_date = (datetime.strptime(ds, '%Y%m%d') - timedelta(days=7)).strftime("%Y%m%d")
            df = ak.stock_zh_a_hist(symbol=symbol, period=period, adjust=adjust, start_date=start_date, end_date=ds, timeout=60)

        logger.info("data retrieved, number of rows {}".format(df.shape[0]))
        df.insert(0, "symbol", symbol)
        df.insert(1, "period", period)
        df.insert(2, "adjust", adjust)
        return df

    def get_downloaded_symbols(self):
        sql = f"SELECT distinct symbol from {self.get_table_name()}  where ds = '{self.ds}' and adjust='{self.adjust}' and period='{self.period}'"
        recs = self.db.run_sql(sql)
        return [rec[0] for rec in recs]




class DataHelper(BaseDataHelper):
    def __init__(self, ds, adjust, period, backfill):
        self.ds = ds
        self.adjust = adjust
        self.period = period
        self.backfill = backfill
        super().__init__(parallel=1, loops_per_second_min=0.5, loops_per_second_max=2)

    def _get_all_symbols(self):
        return get_stock_list()

    def _get_downloaded_symbols(self):
        data = StockZhAHist(ds=self.ds, adjust=self.adjust, period=self.period)
        return data.get_downloaded_symbols()

    def _fetch_symbol_data(self, symbol):
        data = StockZhAHist(ds=self.ds, symbol=symbol, adjust=self.adjust, period=self.period, backfill=self.backfill)
        data.retrieve_data()

    def _clean_up_history(self):
        data = StockZhAHist(ds=self.ds)
        data.clean_up_history(lifecycle=15)



if __name__ == '__main__':
    ds = sys.argv[1]
    adjust = "hfq"
    if len(sys.argv) > 1:
        adjust = sys.argv[2]
    logger.info("execute task on ds {}, adjust {}".format(ds, adjust))
    if not is_trade_date(ds):
        logger.info(f"{ds} is not trade date. task exits.")
        exit(os.EX_OK)

    backfill = is_backfill(ds)
    logger.info(f"ds {ds}, backfill {backfill}, ajust {adjust}, period daily")
    DataHelper(ds, adjust=adjust, period="daily", backfill=backfill).fetch_all_data()
