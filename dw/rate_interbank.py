import sys, os

import pandas as pd

parent_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parent_dir)


import akshare as ak
from datetime import datetime

from utils.log_util import get_logger
from base_data import BaseData

logger = get_logger(__name__)

from utils.stock_zh_a_util import is_trade_date, is_backfill


'''
银行间拆借利率
接口: rate_interbank

目标地址: http://data.eastmoney.com/shibor/shibor.aspx?m=sg&t=88&d=99333&cu=sgd&type=009065&p=79

描述: 东方财富-拆借利率一览-具体市场的具体品种的具体指标的拆借利率数据

限量: 返回所有历史数据
'''


class RateInterbank(BaseData):
    def __init__(self):
        super().__init__()

    def get_table_name(self):
        return "rate_interbank"

    def get_df_schema(self):
        markets = ["上海银行同业拆借市场"]
        symbols = ["Shibor人民币"]
        indicators = ["隔夜", "1周", "2周", "1月", "3月", "6月", "9月", "1年"]
        df_list = []
        for market in markets:
            for symbol in symbols:
                for indicator in indicators:
                    df = ak.rate_interbank(market=market, symbol=symbol, indicator=indicator)
                    df.insert(0, "market", market)
                    df.insert(1, "symbol", symbol)
                    df.insert(2, "indicator", indicator)
                    df.rename(columns={"报告日": "日期"}, inplace=True)
                    df_list.append(df)
        df_schema = pd.concat(df_list)
        return df_schema

    def before_retrieve_data(self):
        pass



if __name__ == '__main__':
    ds = sys.argv[1]
    logger.info("execute task on ds {}".format(ds))
    if not is_trade_date(ds):
        logger.info(f"{ds} is not trade date. task exits.")
        exit(os.EX_OK)

    data = RateInterbank()
    data.set_ds(ds)
    data.retrieve_data()
