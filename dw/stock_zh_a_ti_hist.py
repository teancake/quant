import sys, os

import numpy as np

parent_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parent_dir)

from datetime import datetime, timedelta
import pandas as pd
from joblib import Parallel, delayed

from utils.stock_zh_a_util import get_stock_data, get_stock_list, is_trade_date
from base_data import BaseData

from utils.log_util import get_logger
logger = get_logger(__name__)

from utils import ta_util

class DataHelper:

    def compute_all_symbols(self, ds, adjust, period, stock_symbols=None):
        days_ahead = 365 * 3
        start_date = (datetime.strptime(ds, '%Y%m%d') - timedelta(days=days_ahead)).date()
        stocks = get_stock_data("all", ds, start_date, adjust=adjust, period=period)
        stocks.reset_index(inplace=True)
        if len(stocks) == 0:
            logger.warning(f"stocks data empty. ds {ds}, adjust {adjust}, period {period} computation finished. No results generated.")
            return

        if stock_symbols is None or len(stock_symbols) == 0:
            stock_symbols = get_stock_list()
        downloaded_symbols = self._get_downloaded_symbols(ds, adjust, period)
        logger.info(f"downloaded symbols {downloaded_symbols}")
        stock_symbols = sorted(set(stock_symbols) - set(downloaded_symbols))
        Parallel(n_jobs=4, backend="multiprocessing")(delayed(self._compute_and_save_one_stock)(ds, symbol, stocks[stocks["代码"]==symbol], adjust, period) for symbol in stock_symbols)
        self._clean_up_data(ds)

    def _get_downloaded_symbols(self, ds, adjust, period):
        data = StockZhATiHist(ds=ds)
        data.set_params(ds=ds, adjust=adjust, period=period)
        return data.get_downloaded_symbols()

    def _compute_and_save_one_stock(self, ds, symbol, stock_data, adjust, period):
        data = StockZhATiHist(ds=ds)
        data.set_params(ds=ds, symbol=symbol, stock_data=stock_data, adjust=adjust, period=period)
        data.retrieve_data()

    def _clean_up_data(self, ds):
        data = StockZhATiHist(ds=ds)
        data.clean_up_history(lifecycle=15)


class StockZhATiHist(BaseData):
    def __init__(self, ds):
        super(StockZhATiHist, self).__init__()
        self.ds = ds
        self.params = None

    def get_downloaded_symbols(self):
        if not self.table_exists():
            logger.info(f"table {self.table_name} does not exist, no downloaded symbols")
            return []
        if self.params is None:
            logger.warning("please set params before querying")
            return[]
        adjust = self.params["adjust"]
        period = self.params["period"]
        sql = f"select distinct 代码 from {self.table_name} where ds = {self.ds} and adjust = '{adjust}' and period = '{period}'"
        results = self.db.run_sql(sql)
        return [result[0] for result in results]

    def set_params(self, **kwargs):
        self.params = kwargs

    def compute_one_stock_indicators(self):
        if self.params is None:
            logger.warning("please set params before computation")
            return

        symbol = self.params["symbol"]
        data_df = self.params["stock_data"]
        logger.info("compute technical indicators for symbol {} ".format(symbol))
        if len(data_df) == 0:
            logger.info("empty dataset, no computation required.")
            return pd.DataFrame()
        ti_df = ta_util.get_ta_indicator_map(data_df)
        ti_df = ti_df.replace([np.inf, -np.inf], np.nan)
        ti_df.insert(1, "代码", symbol)
        ti_df.insert(2, "adjust", self.params["adjust"])
        ti_df.insert(3, "period", self.params["period"])
        return ti_df

    def get_df_schema(self):
        df = self.compute_one_stock_indicators()
        return df

    def get_table_name(self):
        return "stock_zh_a_ti_hist"

    def before_retrieve_data(self):
        pass


if __name__ == '__main__':
    ds = sys.argv[1]
    logger.info("execute task on ds {}".format(ds))
    if not is_trade_date(ds):
        logger.info(f"{ds} is not trade date. task exits.")
        exit(os.EX_OK)
    helper = DataHelper()
    helper.compute_all_symbols(ds, adjust="qfq", period="daily")
    helper.compute_all_symbols(ds, adjust="hfq", period="daily")
