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
########################################
电报-财联社
接口：stock_info_global_cls

目标地址：https://www.cls.cn/telegraph

描述：财联社-电报

限量：单次返回指定 symbol 的最近 300 条财联社-电报的数据

########################################
全球财经直播-同花顺财经
接口：stock_info_global_ths

目标地址：https://news.10jqka.com.cn/realtimenews.html

描述：同花顺财经-全球财经直播

限量：单次返回最近 20 条新闻数据

########################################
快讯-富途牛牛
接口：stock_info_global_futu

目标地址：https://news.futunn.com/main/live

描述：富途牛牛-快讯

限量：单次返回最近 50 条新闻数据


########################################
全球财经快讯-新浪财经
接口：stock_info_global_sina

目标地址：https://finance.sina.com.cn/7x24

描述：新浪财经-全球财经快讯

限量：单次返回最近 20 条新闻数据

########################################
全球财经快讯-东财财富
接口：stock_info_global_em

目标地址：https://kuaixun.eastmoney.com/7_24.html

描述：东方财富-全球财经快讯

限量：单次返回最近 200 条新闻数据

'''

class StockInfoGlobal(BaseData):
    def __init__(self):
        super().__init__()

    def get_table_name(self):
        return "stock_info_global"

    def get_df_schema(self):
        print(f"akshare version {ak.__version__}")
        # 所有数据源统一为 标题、内容、发布时间、链接、source
        df_cls = self.get_df_cls()
        df_ths = self.get_df_ths()
        df_futu = self.get_df_futu()
        df_sina = self.get_df_sina()
        df_em = self.get_df_em()
        df = pd.concat([df_cls, df_ths, df_futu, df_sina, df_em])
        return df

    def get_df_cls(self):
        logger.info("stock_info_global_cls")
        df = ak.stock_info_global_cls()
        df["发布时间"] = df.apply(lambda row: datetime.combine(row["发布日期"], row["发布时间"]), axis=1)
        df["链接"] = ""
        df["source"] = "财联社-电报"
        return df[["标题", "内容", "发布时间", "链接", "source"]]

    def get_df_ths(self):
        logger.info("stock_info_global_ths")
        df = ak.stock_info_global_ths()
        df["source"] = "同花顺财经-全球财经直播"
        return df[["标题", "内容", "发布时间", "链接", "source"]]

    def get_df_futu(self):
        logger.info("stock_info_global_futu")
        df = ak.stock_info_global_futu()
        df["source"] = "富途牛牛-快讯"
        return df[["标题", "内容", "发布时间", "链接", "source"]]

    def get_df_sina(self):
        logger.info("stock_info_global_sina")
        df = ak.stock_info_global_sina()
        df.rename(columns={"时间": "发布时间"}, inplace=True)
        df["标题"] = ""
        df["链接"] = ""
        df["source"] = "新浪财经-全球财经快讯"
        return df[["标题", "内容", "发布时间", "链接", "source"]]

    def get_df_em(self):
        logger.info("stock_info_global_em")
        df = ak.stock_info_global_em()
        df.rename(columns={"摘要": "内容"}, inplace=True)
        df["source"] = "东方财富-全球财经快讯"
        return df[["标题", "内容", "发布时间", "链接", "source"]]


    def before_retrieve_data(self):
        pass


if __name__ == '__main__':
    ds = sys.argv[1]
    logger.info("execute task on ds {}".format(ds))
    spot_em = StockInfoGlobal()
    spot_em.set_ds(ds)
    spot_em.retrieve_data()