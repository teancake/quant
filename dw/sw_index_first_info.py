import sys, os
parent_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parent_dir)


import akshare as ak
from datetime import datetime

from utils.stock_zh_a_util import is_trade_date

from utils.log_util import get_logger
from base_data import BaseData
from utils.net_util import set_default_user_agent
logger = get_logger(__name__)

'''
申万一级行业信息
接口: sw_index_first_info

目标地址: https://legulegu.com/stockdata/sw-industry-overview#level1

描述: 申万一级行业信息
'''


class SwIndexFirstInfo(BaseData):
    def __init__(self, ds):
        super().__init__()
        self.ds = ds

    def get_table_name(self):
        return "sw_index_first_info"

    def get_df_schema(self):
        set_default_user_agent()
        df = ak.sw_index_first_info()
        df["TTM滚动市盈率"] = df["TTM(滚动)市盈率"]
        df.drop(columns=["TTM(滚动)市盈率"], inplace=True)
        return df

    def before_retrieve_data(self):
        pass


if __name__ == '__main__':
    ds = sys.argv[1]
    logger.info("execute task on ds {}".format(ds))

    if not is_trade_date(ds):
        logger.info(f"{ds} is not trade date. task exits.")
        exit(os.EX_OK)

    stock_zh_a = SwIndexFirstInfo(ds)
    stock_zh_a.retrieve_data()
    stock_zh_a.clean_up_history()




