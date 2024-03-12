import sys, os

import pandas as pd

parent_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parent_dir)


import akshare as ak
from datetime import datetime, timedelta

from utils.log_util import get_logger
from utils.stock_zh_a_util import is_trade_date

from base_data import BaseData

logger = get_logger(__name__)

import hashlib

'''
电报
接口：stock_telegraph_cls

目标地址：https://www.cls.cn/telegraph

描述：财联社-电报

限量：单次返回指定 symbol 的财联社-电报的数据


'''

class StockTelegraphCls(BaseData):
    def __init__(self):
        super().__init__()

    def get_table_name(self):
        return "stock_telegraph_cls"

    def get_df_schema(self):
        print(f"akshare version {ak.__version__}")
        # df = ak.stock_telegraph_cls(symbol="全部")
        df = ak.stock_telegraph_cls()
        df["发布时间"] = df.apply(lambda row: datetime.combine(row["发布日期"], row["发布时间"]), axis=1)
        df["发布日期"] = pd.to_datetime(df["发布日期"])
        df = df[df["发布日期"] == datetime.strptime(ds, "%Y%m%d")]
        return df

    def before_retrieve_data(self):
        pass


if __name__ == '__main__':
    ds = sys.argv[1]
    logger.info("execute task on ds {}".format(ds))
    spot_em = StockTelegraphCls()
    spot_em.set_ds(ds)
    spot_em.retrieve_data()