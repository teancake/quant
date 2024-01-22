import sys, os
parent_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parent_dir)

import akshare as ak
import pandas as pd
from datetime import datetime, timedelta


from utils.log_util import get_logger
from base_data import BaseData
from utils import stock_zh_a_util
from utils.stock_zh_a_util import is_trade_date

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
    def __init__(self, symbol=None, backfill=False, period_list=None):
        self.symbol = symbol
        self.backfill = backfill
        if isinstance(period_list, list) and all(item in ["daily", "weekly", "monthly"] for item in period_list):
            self.period_list = period_list
        else:
            logger.warn("somme elements in period_list {} not recognized, should be daily, weekly, or monthly. Use daily as default.".format(period_list))
            self.period_list = ["daily"]
        super().__init__()

    def set_symbol(self, symbol):
        self.symbol = symbol

    def get_table_name(self):
        return "fund_etf_hist_em"

    def before_retrieve_data(self):
        pass

    def get_df_schema(self):
        # period_list = ["daily", "weekly", "monthly"]
        period_list = self.period_list
        # qfq: 返回前复权后的数据; hfq: 返回后复权后的数据
        adjust_list = ["hfq"]
        df_list = []
        for period in period_list:
            for adjust in adjust_list:
                logger.info(f"retrieving symbol {symbol} on ds {self.ds} for period {period}, adjust {adjust}.")
                df = self.get_single_df(self.symbol, period, adjust, self.ds, self.backfill)
                df_list.append(df)
            time.sleep(1)
        df_schema = pd.concat(df_list)
        return df_schema

    def get_single_df(self, symbol, period, adjust, ds, backfill):
        # restrict end data to ds
        if backfill:
            df = ak.fund_etf_hist_em(symbol=symbol, period=period, adjust=adjust, end_date=ds)
        else:
            start_date = (datetime.strptime(ds, '%Y%m%d') - timedelta(days=7)).strftime("%Y%m%d")
            df = ak.fund_etf_hist_em(symbol=symbol, period=period, adjust=adjust, start_date=start_date, end_date=ds)

        logger.info("data retrieved, number of rows {}".format(df.shape[0]))
        df.insert(0, "symbol", symbol)
        df.insert(1, "period", period)
        df.insert(2, "adjust", adjust)
        return df



if __name__ == '__main__':
    ds = sys.argv[1]
    logger.info("execute task on ds {}".format(ds))
    if not is_trade_date(ds):
        logger.info(f"{ds} is not trade date. task exits.")
        exit(os.EX_OK)

    period_list = ["daily"] if len(sys.argv) <= 2 else [sys.argv[2]]
    weekday = datetime.strptime(ds, "%Y%m%d").isoweekday()

    ## do backfill every Friday
    backfill = True if weekday == 5 else False
    logger.info("ds {}, execute {} task, backfill {}".format(ds, period_list, backfill))

    data = FundEtfHistEm(backfill=backfill, period_list=period_list)
    data.set_ds(ds)
    symbol_list = stock_zh_a_util.get_fund_etf_list()
    for symbol in symbol_list:
        logger.info("process symbol {}".format(symbol))
        data.set_symbol(symbol)
        data.retrieve_data()
        logger.info("symbol {} done".format(symbol))

    data.clean_up_history(lifecycle=15)
