import sys, os
parent_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parent_dir)


import random

import pandas as pd
import requests.utils
import json
import time
import sys

from dw.base_data import BaseData
from utils.log_util import get_logger

logger = get_logger(__name__)

from utils.stock_zh_a_util import get_stock_position, get_stock_list, get_normal_stock_list
from utils.config_util import get_ollama_config
from ollama import Client
import re
import traceback
from datetime import datetime, timedelta
from tqdm import tqdm

requests.utils.default_user_agent = lambda: "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36"



def stock_notice_report(symbol: str = "全部", date: str = "20220511") -> pd.DataFrame:
    """
    东方财富网-数据中心-公告大全-沪深京 A 股公告
    http://data.eastmoney.com/notices/hsa/5.html
    :param symbol: 报告类型; choice of {"全部", "重大事项", "财务报告", "融资公告", "风险提示", "资产重组", "信息变更", "持股变动"}
    :type symbol: str
    :param date: 制定日期
    :type date: str
    :return: 沪深京 A 股公告
    :rtype: pandas.DataFrame
    """
    logger.info(f"get stock_notice_report for {symbol} on date {date}")
    url = "http://np-anotice-stock.eastmoney.com/api/security/ann"
    report_map = {
        "全部": "0",
        "财务报告": "1",
        "融资公告": "2",
        "风险提示": "3",
        "信息变更": "4",
        "重大事项": "5",
        "资产重组": "6",
        "持股变动": "7",
    }
    params = {
        "sr": "-1",
        "page_size": "100",
        "page_index": "1",
        "ann_type": "A",
        "client_source": "web",
        "f_node": report_map[symbol],
        "s_node": "0",
        "begin_time": "-".join([date[:4], date[4:6], date[6:]]),
        "end_time": "-".join([date[:4], date[4:6], date[6:]]),
    }
    r = requests.get(url, params=params)
    data_json = r.json()
    import math

    total_page = math.ceil(data_json["data"]["total_hits"] / 100)

    big_df = pd.DataFrame()
    for page in tqdm(range(1, int(total_page) + 1), leave=False):
        params.update(
            {
                "page_index": page,
            }
        )
        r = requests.get(url, params=params)
        data_json = r.json()
        temp_df = pd.DataFrame(data_json["data"]["list"])
        temp_codes_df = pd.DataFrame(
            [item["codes"][0] for item in data_json["data"]["list"]]
        )
        try:
            temp_columns_df = pd.DataFrame(
                [item["columns"][0] for item in data_json["data"]["list"]]
            )
        except:
            continue
        del temp_df["codes"]
        del temp_df["columns"]
        temp_df = pd.concat([temp_df, temp_columns_df, temp_codes_df], axis=1)
        big_df = pd.concat([big_df, temp_df], ignore_index=True)
        time.sleep(2)

    if len(big_df) == 0:
        return big_df

    big_df.rename(
        columns={
            # "art_code": "_",
            "display_time": "-",
            "eiTime": "-",
            "notice_date": "公告日期",
            "title": "公告标题",
            "column_code": "-",
            "column_name": "公告类型",
            "ann_type": "-",
            "inner_code": "-",
            "market_code": "-",
            "short_name": "名称",
            "stock_code": "代码",
        },
        inplace=True,
    )
    big_df = big_df[
        [
            "代码",
            "名称",
            "公告标题",
            "公告类型",
            "公告日期",
            "art_code"
        ]
    ]
    big_df["公告日期"] = pd.to_datetime(big_df["公告日期"]).dt.date
    return big_df


def stock_notice_report_detail(art_code: str = "AN202405191633832804"):
    logger.info(f"getting stock_notice_report_detail for art_code {art_code}")
    url = "https://np-cnotice-stock.eastmoney.com/api/content/ann"
    params = {
        "callback": "jQuery112300745853825921321_1716106691044",
        "art_code": art_code,
        "client_source": "web",
        "page_index": 1,
        "_": int(time.time()*1000),
    }
    r = requests.get(url, params=params)
    data_text = r.text
    logger.info(f"data from response {data_text}")
    #
    data_json = json.loads(data_text)
    desired_keys = ["art_code",
                    "attach_url",
                    "eitime",
                    "notice_content",
                    "notice_date",
                    "notice_title",
                    "short_name"
                    ]
    data_json = data_json["data"]
    data_json = {key: value for key, value in data_json.items() if key in desired_keys}
    temp_df = pd.DataFrame(data_json, index=[0])
    return temp_df


class SentimentAnalyser:
    ollama_conf = get_ollama_config()
    client = Client(host=f"http://{ollama_conf['server_address']}:{ollama_conf['port']}")
    model = "qwen:14b"

    @classmethod
    def sentiment_analysis(cls, df):
        logger.info("analyze sentiment...")
        output_cols = ["summary", "sentiment", "reason"]
        for index, row in df.iterrows():
            # qwen1.5 14b在长文本上有点拉胯，限制一下文本长度
            content = row["notice_content"]
            content = re.sub(r"\s+", " ", content)
            res = cls._chat(content[:1500])
            for output in output_cols:
                df.loc[index, output] = res.get(output, "")
        return df

    @classmethod
    def _chat(cls, message: str):
        logger.info(f"message {message}")
        content = None
        try:
            response = cls.client.chat(model=cls.model, messages=[
                {
                    "role": "system",
                    "content": """
                        你是一个证券分析师，擅长解读证券交易所的公告，并分析其对相关上市公司的影响。下面是一份公告，请从中提取以下内容：
                        1. 一句话汇总公告。
                        2. 对公司未来经营状况的判断（正面、中性、负面）。
                        3. 一句话说明作出判断的原因。
                        结果以JSON格式输出，只能包含3个字段，summary对应摘要, sentiment对应对经营状况的判断, reason是判断的原因。 只输出JSON即可，用中文回答。
                    """,
                },
                {
                    "role": "user",
                    "content": f'{message}',
                },
            ])
            content = response["message"]["content"].replace("\n", "")
            pattern = r"({.*})"
            match = re.search(pattern, content)
            if not match:
                logger.warning(f"regexp match failed, content {content}")
                return dict()
            js = json.loads(match.group(1))
            return js
        except Exception as e:
            logger.error(traceback.format_exception(e))
            logger.warning(f"exception occurred, content {content}")
            return dict()

    @classmethod
    def sentiment_analysis_batch(cls, df):
        batch_size = 1
        for i in range(0, len(df), batch_size):
            batch_df = df.iloc[i:i + batch_size]
            batch_df = cls._chat(batch_df.reset_index(drop=True))
            df.iloc[i:i + batch_size] = batch_df
        return df

    @classmethod
    def _chat_batch(cls, df):
        post_titles = df.apply(lambda row: f"{row.name + 1}. {row['post_title']}", axis=1).to_list()
        message = "\n ".join(post_titles)
        res_df = pd.DataFrame(index=range(len(df)), columns=["sentiment", "reason"])
        logger.info(f"message {message}")
        content = None
        try:
            response = cls.client.chat(model=cls.model, messages=[
                {
                    "role": "system",
                    "content": f"你是一个股票分析师，下面有{len(post_titles)}句话，判断每句话中说话人对该股票的情绪，并解释原因，不能使用引号括号等特殊字符。输出使用JSON数组格式，数组的元素是每句话的结果，字段为row_number, sentiment, reason。row_number为行号，从1开始， sentiment为情绪，可接受的值为正面，中性或者负面中最可能的一个，reason是原因。请仔细检查输出格式确保为正确的JSON数组。",
                },
                {
                    "role": "user",
                    "content": f'{message}',
                },
            ])
            content = response["message"]["content"].replace("\n", "")
            logger.info(f"response content {content}")
            pattern = r"(\[.*\])"
            match = re.search(pattern, content)
            if not match:
                logger.warning(f"regexp match failed, content {content}")
            js = json.loads(match.group(1))
            res_df = pd.DataFrame(js)
            res_df = res_df.drop("row_number", axis=1)
            print(f"res_df {res_df}")
        except Exception as e:
            logger.error(traceback.format_exception(e))
            logger.warning(f"exception occurred, content {content}")
        return df.join(res_df)


class StockNoticeReport(BaseData):
    def __init__(self, ds, art_df=None, enable_sentiment=False):
        super().__init__()
        self.ds = ds
        self.art_df = art_df
        self.enable_sentiment = enable_sentiment

    def before_retrieve_data(self):
        pass

    def get_table_name(self):
        return "stock_notice_report"

    def get_df_schema(self):
        df = stock_notice_report_detail(art_code=self.art_df["art_code"][0])
        df = df.reset_index(drop=True)
        df = pd.concat([self.art_df, df], axis=1)
        #
        if self.enable_sentiment:
            df = SentimentAnalyser.sentiment_analysis(df)
        return df

    def get_downloaded_symbols(self):
        if not self.table_exists():
            logger.info(f"table {self.table_name} does not exist, no downloaded symbols")
            return []
        sql = f"select distinct art_code from {self.table_name}"
        results = self.db.run_sql(sql)
        return [result[0] for result in results]

    def clean_up_history(self, lifecycle=30):
        pass


class DataHelper:
    def rate_limiter_sleep(self, timer_start, loops_per_second_min, loops_per_second_max):
        loop_time_second_min = 1.0 / loops_per_second_max
        loop_time_second_max = 1.0 / loops_per_second_min
        dt = time.time() - timer_start
        if dt < loop_time_second_min:
            random_time = random.random() * (loop_time_second_max - loop_time_second_min) + loop_time_second_min - dt
            logger.info(f"dt is {dt} less than minimum loop time {loop_time_second_min}, sleep {random_time} seconds")
            time.sleep(random_time)

    def _get_days_after_ds(self, ds, days):
        return (datetime.strptime(ds, '%Y%m%d') + timedelta(days=days)).strftime("%Y%m%d")

    def get_all_articles(self, ds, enable_sentiment=False):
        downloaded_art_codes = self._get_downloaded_articles(ds)
        logger.info(f"downloaded art_codes {downloaded_art_codes}")
        df0 = stock_notice_report(symbol="全部", date=ds)
        time.sleep(2)
        df1 = stock_notice_report(symbol="全部", date=self._get_days_after_ds(ds, 1))
        time.sleep(3)
        df2 = stock_notice_report(symbol="全部", date=self._get_days_after_ds(ds, 2))
        df = pd.concat([df0, df1, df2])
        # only get normal stocks, to reduce data size
        logger.info(f"before filtering normal stocks, data count {len(df)}")
        normal_stocks = get_normal_stock_list()
        df = df[df["代码"].apply(lambda row: row in normal_stocks)]
        logger.info(f"after filtering normal stocks, data count {len(df)}")
        art_codes = df["art_code"].to_list()
        art_codes = sorted(set(art_codes) - set(downloaded_art_codes))
        logger.info(f"after filtering downloaded data, data count {len(df)}")

        for art_code in tqdm(art_codes):
            timer_start = time.time()
            art_df = df.loc[df["art_code"] == art_code].reset_index(drop=True)
            try:
                logger.info(f"retrieving art_code {art_code} on ds {ds}.")
                self._get_one_article(ds=ds, art_df=art_df, enable_sentiment=enable_sentiment)
            except Exception as e:
                logger.error(traceback.format_exception(e))
            self.rate_limiter_sleep(timer_start, loops_per_second_min=0.1, loops_per_second_max=0.3)

        self._clean_up_data(ds)

    def _get_downloaded_articles(self, ds):
        return StockNoticeReport(ds=ds).get_downloaded_symbols()

    def _get_one_article(self, ds, art_df, enable_sentiment):
        data = StockNoticeReport(ds=ds, art_df=art_df, enable_sentiment=enable_sentiment)
        data.retrieve_data()

    def _clean_up_data(self, ds):
        StockNoticeReport(ds=ds).clean_up_history(lifecycle=365)


if __name__ == '__main__':
    ds = sys.argv[1]
    logger.info("execute task on ds {}".format(ds))
    DataHelper().get_all_articles(ds, enable_sentiment=True)

