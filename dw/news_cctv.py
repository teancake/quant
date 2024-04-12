import sys, os
import akshare as ak
from datetime import datetime
import time
parent_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parent_dir)

from utils.stock_zh_a_util import is_trade_date, is_backfill, get_stock_list
from utils.log_util import get_logger
from base_data import BaseData

logger = get_logger(__name__)

'''
新闻联播文字稿
接口: news_cctv

目标地址: https://tv.cctv.com/lm/xwlb/?spm=C52056131267.P4y8I53JvSWE.0.0

描述: 新闻联播文字稿, 数据区间从 20160330-至今

限量: 单次返回指定日期新闻联播文字稿数据


'''


class NewsCctv(BaseData):
    def __init__(self, ds):
        super().__init__()
        self.ds = ds

    def before_retrieve_data(self):
        pass

    def get_table_name(self):
        return "news_cctv"

    def get_df_schema(self):
        df = ak.news_cctv(date=self.ds)
        return df


if __name__ == '__main__':
    ds = sys.argv[1]
    logger.info("execute task on ds {}".format(ds))
    data = NewsCctv(ds)
    data.retrieve_data()

