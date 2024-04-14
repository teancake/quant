import sys, os
parent_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parent_dir)


import akshare as ak
from datetime import datetime
import time


from utils.stock_zh_a_util import is_trade_date, get_sw_industries

from utils.log_util import get_logger
from base_data import BaseData

logger = get_logger(__name__)

'''
行业历史资金流
接口: stock_sector_fund_flow_hist

目标地址: https://data.eastmoney.com/bkzj/BK1034.html

描述: 东方财富网-数据中心-资金流向-行业资金流-行业历史资金流

限量: 单次获取指定行业的行业历史资金流数据 180天
'''


class StockSectorFundFlowHist(BaseData):
    def __init__(self, ds, symbol):
        super().__init__()
        self.ds = ds
        self.symbol = symbol

    def get_table_name(self):
        return "stock_sector_fund_flow_hist"

    def get_df_schema(self):
        df = ak.stock_sector_fund_flow_hist(symbol=self.symbol)
        df.rename(columns={"主力净流入-净额": "主力净流入_净额", "主力净流入-净占比": "主力净流入_净占比", "超大单净流入-净额": "超大单净流入_净额",
                           "超大单净流入-净占比": "超大单净流入_净占比", "大单净流入-净额": "大单净流入_净额", "大单净流入-净占比": "大单净流入_净占比",
                           "中单净流入-净额": "中单净流入_净额", "中单净流入-净占比": "中单净流入_净占比", "小单净流入-净额": "小单净流入_净额",
                           "小单净流入-净占比": "小单净流入_净占比"}, inplace=True)
        df.insert(0, "行业", self.symbol)
        return df

    def before_retrieve_data(self):
        pass

    def get_downloaded_symbols(self):
        if not self.table_exists():
            logger.info(f"table {self.table_name} does not exist, no downloaded symbols")
            return []
        sql = f"select distinct 行业 from {self.table_name} where ds = {self.ds} and 行业 = '{self.symbol}'"
        results = self.db.run_sql(sql)
        return [result[0] for result in results]


class DataHelper:

    def rate_limiter_sleep(self, timer_start, loops_per_second=1.0):
        loop_time_second = 1.0 / loops_per_second
        dt = time.time() - timer_start
        if dt < loop_time_second:
            logger.info(f"dt is {dt}, sleep {loop_time_second - dt} seconds")
            time.sleep(loop_time_second - dt)


    def get_all_symbols(self, ds):
        symbols = get_sw_industries()
        downloaded_symbols = self._get_downloaded_symbols(ds)
        logger.info(f"downloaded symbols {downloaded_symbols}")
        symbols = sorted(set(symbols) - set(downloaded_symbols))
        for symbol in symbols:
            timer_start = time.time()
            try:
                logger.info(f"retrieving symbol {symbol} on ds {ds}.")
                self._get_one_symbol(ds=ds, symbol=symbol)
            except Exception as e:
                logger.warning(f"exception occurred {e}")
            self.rate_limiter_sleep(timer_start, loops_per_second=5)

        self._clean_up_data(ds)

    def _get_downloaded_symbols(self, ds):
        return StockSectorFundFlowHist(ds=ds, symbol="").get_downloaded_symbols()

    def _get_one_symbol(self, ds, symbol):
        data = StockSectorFundFlowHist(ds=ds, symbol=symbol)
        data.retrieve_data()

    def _clean_up_data(self, ds):
        StockSectorFundFlowHist(ds=ds, symbol="").clean_up_history(lifecycle=15)



if __name__ == '__main__':
    ds = sys.argv[1]
    logger.info("execute task on ds {}".format(ds))

    if not is_trade_date(ds):
        logger.info(f"{ds} is not trade date. task exits.")
        exit(os.EX_OK)
    DataHelper().get_all_symbols(ds)




