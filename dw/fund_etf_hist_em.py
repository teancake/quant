import sys, os
parent_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parent_dir)

import akshare as ak
import pandas as pd
from datetime import datetime, timedelta


from utils.log_util import get_logger
from base_data import BaseData, BaseDataHelper
from utils import stock_zh_a_util
from utils.stock_zh_a_util import is_trade_date, is_backfill

logger = get_logger(__name__)

import time
import sys
import re
import traceback



'''
ETF基金历史行情-东财
接口: fund_etf_hist_em

目标地址: http://quote.eastmoney.com/sz159707.html

描述: 东方财富-ETF 行情; 历史数据按日频率更新, 当日收盘价请在收盘后获取

限量: 单次返回指定 ETF、指定周期和指定日期间的历史行情日频率数据

'''


class FundEtfHistEm(BaseData):
    def __init__(self, ds, symbol=None, adjust="hfq", period="daily", backfill=False):
        self.ds = ds
        self.symbol = symbol
        self.adjust = adjust
        self.period = period
        self.backfill = backfill
        super().__init__()

    def get_table_name(self):
        return "fund_etf_hist_em"

    def before_retrieve_data(self):
        pass

    def get_df_schema(self):
        # restrict end data to ds
        if backfill:
            df = ak.fund_etf_hist_em(symbol=self.symbol, period=self.period, adjust=self.adjust, end_date=self.ds)
        else:
            start_date = (datetime.strptime(ds, '%Y%m%d') - timedelta(days=7)).strftime("%Y%m%d")
            logger.info(f"symbol {self.symbol}, period {self.period}, adjust {self.adjust}, start_date {start_date}, end_date {self.ds}")
            df = ak.fund_etf_hist_em(symbol=self.symbol, period=self.period, adjust=self.adjust, start_date=start_date, end_date=self.ds)

        logger.info("data retrieved, number of rows {}".format(len(df)))
        df.insert(0, "symbol", self.symbol)
        df.insert(1, "period", self.period)
        df.insert(2, "adjust", self.adjust)
        return df

    def get_downloaded_symbols(self):
        if not self.table_exists():
            logger.info(f"table {self.table_name} does not exist, no downloaded symbols")
            return []
        sql = f"select distinct symbol from {self.table_name} where ds = {self.ds} and adjust = '{self.adjust}' and period = '{self.period}'"
        results = self.db.run_sql(sql)
        return [result[0] for result in results]

class DataHelper(BaseDataHelper):
    def __init__(self, ds, adjust, period):
        self.ds = ds
        self.adjust = adjust
        self.period = period
        super().__init__(loops_per_second_min=1, loops_per_second_max=3, parallel=1)

    def _get_all_symbols(self):
        return stock_zh_a_util.get_fund_etf_list()

    def _get_downloaded_symbols(self):
        data = FundEtfHistEm(ds=self.ds, adjust=self.adjust, period=self.period)
        return data.get_downloaded_symbols()

    def _fetch_symbol_data(self, symbol):
        data = FundEtfHistEm(symbol=symbol, ds=self.ds, adjust=self.adjust, period=self.period)
        data.retrieve_data()

    def _clean_up_history(self):
        data = FundEtfHistEm(ds=self.ds)
        data.clean_up_history(lifecycle=15)



if __name__ == '__main__':
    ds = sys.argv[1]
    logger.info("execute task on ds {}".format(ds))
    if not is_trade_date(ds):
        logger.info(f"{ds} is not trade date. task exits.")
        exit(os.EX_OK)

    backfill = is_backfill(ds)
    logger.info("ds {}, backfill {}, period {}, adjust {}".format(ds, backfill, "daily", "hfq"))
    DataHelper(ds=ds, adjust="hfq", period="daily").fetch_all_data()
    logger.info("ds {}, backfill {}, period {}, adjust {}".format(ds, backfill, "daily", "hfq"))
    DataHelper(ds=ds, adjust="qfq", period="daily").fetch_all_data()
