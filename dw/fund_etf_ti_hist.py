import sys, os

import numpy as np

parent_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parent_dir)

from datetime import datetime, timedelta
import pandas as pd

from utils.stock_zh_a_util import get_stock_data, get_stock_list, is_trade_date, get_fund_etf_list, get_fund_etf_data
from base_data import BaseData, BaseDataHelper

from utils.log_util import get_logger
logger = get_logger(__name__)

from utils import ta_util

class DataHelper(BaseDataHelper):
    def __init__(self, ds, adjust, period):
        self.ds = ds
        self.adjust = adjust
        self.period = period
        super().__init__(parallel=1, loops_per_second_max=None)

    def _get_all_symbols(self):
        return get_fund_etf_list()

    def _get_downloaded_symbols(self):
        data = FundEtfTiHist(ds=self.ds, adjust=self.adjust, period=self.period)
        return data.get_downloaded_symbols()

    def _fetch_symbol_data(self, symbol):
        logger.info(f"symbol {symbol}, adjust {self.adjust} period {self.period}")
        input_data = self.get_fund_etf_hist(symbol)
        data = FundEtfTiHist(ds=self.ds, symbol=symbol, adjust=self.adjust, period=self.period, input_data=input_data)
        data.retrieve_data()

    def _clean_up_history(self):
        data = FundEtfTiHist(ds=self.ds)
        data.clean_up_history(lifecycle=15)

    def get_fund_etf_hist(self, symbol):
        days_ahead = 365 * 3
        start_date = (datetime.strptime(ds, '%Y%m%d') - timedelta(days=days_ahead)).date()
        data = get_fund_etf_data(symbol, ds, start_date, adjust=self.adjust, period=self.period)
        data.reset_index(inplace=True)
        logger.info(f"get_fund_etf_hist completed, data size {len(data)}")
        return data


class FundEtfTiHist(BaseData):
    def __init__(self, ds, symbol=None, adjust=None, period=None, input_data=None):
        super(FundEtfTiHist, self).__init__()
        self.ds = ds
        self.symbol = symbol
        self.adjust = adjust
        self.period = period
        self.input_data = input_data

    def get_table_name(self):
        return "fund_etf_ti_hist"

    def before_retrieve_data(self):
        pass

    def get_downloaded_symbols(self):
        if not self.table_exists():
            logger.info(f"table {self.table_name} does not exist, no downloaded symbols")
            return []
        sql = f"select distinct 代码 from {self.table_name} where ds = {self.ds} and adjust = '{self.adjust}' and period = '{self.period}'"
        results = self.db.run_sql(sql)
        return [result[0] for result in results]


    def get_df_schema(self):
        logger.info("compute technical indicators for symbol {} ".format(self.symbol))
        if self.input_data is None or len(self.input_data) == 0:
            logger.info("empty dataset, no computation required.")
            return pd.DataFrame()
        ti_df = ta_util.get_ta_indicator_map(self.input_data)
        ti_df = ti_df.replace([np.inf, -np.inf], np.nan)
        ti_df.insert(1, "代码", self.symbol)
        ti_df.insert(2, "adjust", self.adjust)
        ti_df.insert(3, "period", self.period)
        return ti_df

if __name__ == '__main__':
    ds = sys.argv[1]
    logger.info("execute task on ds {}".format(ds))
    if not is_trade_date(ds):
        logger.info(f"{ds} is not trade date. task exits.")
        exit(os.EX_OK)
    DataHelper(ds, adjust="hfq", period="daily").fetch_all_data()
    DataHelper(ds, adjust="qfq", period="daily").fetch_all_data()
