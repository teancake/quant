import sys, os
parent_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parent_dir)


import akshare as ak
from datetime import datetime, timedelta

from utils.log_util import get_logger
from utils.stock_zh_a_util import is_trade_date

from base_data import BaseData

logger = get_logger(__name__)

'''
ETF基金实时行情-东财
接口: fund_etf_spot_em

目标地址: https://quote.eastmoney.com/center/gridlist.html#fund_etf

描述: 东方财富-ETF 实时行情

'''


class FundEtfSpotEm(BaseData):
    def __init__(self, ds):
        super().__init__(ds)

    def get_table_name(self):
        return "fund_etf_spot_em"

    def get_df_schema(self):
        df = ak.fund_etf_spot_em()
        df_schema = df[["代码", "名称", "最新价", "涨跌额", "涨跌幅", "成交量", "成交额", "开盘价", "最高价", "最低价", "昨收", "换手率", "流通市值", "总市值"]]
        return df_schema

    def before_retrieve_data(self):
        pass




if __name__ == '__main__':

    ds = sys.argv[1]
    logger.info("execute task on ds {}".format(ds))
    if not is_trade_date(ds):
        logger.info(f"{ds} is not trade date. task exits.")
        exit(os.EX_OK)

    spot_em = FundEtfSpotEm(ds)
    spot_em.retrieve_data()
    spot_em.clean_up_history(lifecycle=60)
