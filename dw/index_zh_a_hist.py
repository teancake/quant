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
历史行情数据-通用
接口: index_zh_a_hist

目标地址: http://quote.eastmoney.com/center/hszs.html

描述: 东方财富网-中国股票指数-行情数据

限量: 单次返回具体指数指定 period 从 start_date 到 end_date 的之间的近期数据

'''


class IndexZhAHist(BaseData):
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
        return "index_zh_a_hist"

    def before_retrieve_data(self):
        pass

    def get_df_schema(self):
        # period_list = ["daily", "weekly", "monthly"]
        period_list = self.period_list
        df_list = []
        for period in period_list:
            logger.info("retrieving index_zh_a_hist on ds {} for symbol {}, period {}.".format(self.ds, symbol, period))
            df = self.get_single_df(self.symbol, period, self.ds, self.backfill)
            df_list.append(df)
            time.sleep(1)
        df_schema = pd.concat(df_list)
        return df_schema

    def get_single_df(self, symbol, period, ds, backfill):
        id = re.sub("[a-zA-Z]", "", symbol)
        # restrict end data to ds
        if backfill:
            df = ak.index_zh_a_hist(symbol=id, period=period, end_date=ds)
        else:
            start_date = (datetime.strptime(ds, '%Y%m%d') - timedelta(days=7)).strftime("%Y%m%d")
            df = ak.index_zh_a_hist(symbol=id, period=period, start_date=start_date, end_date=ds)

        logger.info("data retrieved, number of rows {}".format(df.shape[0]))
        df.insert(0, "代码", symbol)
        df.insert(1, "symbol", id)
        df.insert(2, "period", period)
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
    data = IndexZhAHist(backfill=backfill, period_list=period_list)
    symbol_list = stock_zh_a_util.get_index_list()
    for symbol in symbol_list:
        try:
            logger.info("process symbol {}".format(symbol))
            data.set_symbol(symbol)
            data.retrieve_data()
            logger.info("symbol {} done".format(symbol))
        except Exception as e:
            logger.warning("symbol {} exception".format(symbol))
            logger.warning(traceback.format_exc())


    if backfill:
        delete_ds = (datetime.strptime(ds, '%Y%m%d') - timedelta(days=14)).strftime("%Y%m%d")
        logger.info("clearing historical data after back fill, data before ds {} will be deleted.".format(delete_ds))
        cond = "ds <= '{}'".format(delete_ds)
        data.delete_records(conditions=cond)
        logger.info("clearing historical data done.")
