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
from datetime import datetime
from tqdm import tqdm

def guba_em(symbol: str = "600000"):
    requests.utils.default_user_agent = lambda: "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36"
    url = "https://gbapi.eastmoney.com/webarticlelist/api/Article/Articlelist"
    params = {
        "callback": "jQuery183040220970832973046_1715567595839",
        "code": symbol,
        "sorttype": 1,
        "ps": 36,
        "from": "CommonBaPost",
        "deviceid": "0.3410789631307125",
        "version": 200,
        "product": "Guba",
        "plat": "Web",
        "_": int(time.time()*1000),
    }
    r = requests.get(url, params=params)
    data_text = r.text
    data_json = json.loads(
        data_text.strip("jQuery183040220970832973046_1715567595839(")[:-1]
    )
    temp_df = pd.DataFrame(data_json["re"])
    return temp_df[["post_id",
                    "post_title",
                    "stockbar_code",
                    "stockbar_name",
                    "stockbar_type",
                    "stockbar_exchange",
                    "user_id",
                    "user_nickname",
                    "post_click_count",
                    "post_forward_count",
                    "post_comment_count",
                    "post_publish_time",
                    "post_last_time",
                    "post_type",
                    "post_state",
                    "post_from_num",
                    "post_top_status",
                    "post_has_pic",
                    "post_has_video",
                    "user_is_majia",
                    "post_ip",
                    "post_display_time"]]


class SentimentAnalyser:
    ollama_conf = get_ollama_config()
    client = Client(host=f"http://{ollama_conf['server_address']}:{ollama_conf['port']}")
    model = "qwen:14b"

    @classmethod
    def sentiment_analysis(cls, df):
        input_cols = ["post_title"]
        output_cols = ["sentiment", "reason"]
        for index, row in df.iterrows():
            res = cls._chat(*row[input_cols].to_list())
            for output in output_cols:
                df.loc[index, output] = res.get(output, "")
        return df

    @classmethod
    def _chat(cls, *args):
        message = args[0]
        content = None
        try:
            response = cls.client.chat(model=cls.model, messages=[
                {
                    "role": "system",
                    "content": "你是一个股票分析师。根据下面的话，判断说话人对该股票的情绪，正面，中性或者负面，并解释原因，以JSON格式输出，字段为sentiment, reason。只输出JSON字符串即可，不要换行。",
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


class GubaEm(BaseData):
    def __init__(self, ds, symbol, enable_sentiment=False):
        super().__init__()
        self.ds = ds
        self.symbol = symbol
        self.enable_sentiment = enable_sentiment

    def before_retrieve_data(self):
        pass

    def get_table_name(self):
        return "guba_em"

    def get_df_schema(self):
        df = guba_em(symbol=self.symbol)
        df = df[df["post_publish_time"].apply(lambda row: pd.to_datetime(row).date() == datetime.strptime(self.ds, "%Y%m%d").date())]
        df = df.reset_index(drop=True)
        df.insert(loc=0, column="symbol", value=self.symbol)
        if self.enable_sentiment:
            df = SentimentAnalyser.sentiment_analysis(df)
        return df

    def get_downloaded_symbols(self):
        if not self.table_exists():
            logger.info(f"table {self.table_name} does not exist, no downloaded symbols")
            return []
        sql = f"select distinct symbol from {self.table_name} where ds = {self.ds}"
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

    def get_all_symbols(self, ds, symbol_type="position", enable_sentiment=False):
        if symbol_type == "all":
            symbols = get_normal_stock_list()
            downloaded_symbols = self._get_downloaded_symbols(ds)
        else:
            symbols = get_stock_position()
            downloaded_symbols = []

        logger.info(f"downloaded symbols {downloaded_symbols}")
        symbols = sorted(set(symbols) - set(downloaded_symbols))
        for symbol in tqdm(symbols):
            timer_start = time.time()
            try:
                logger.info(f"retrieving symbol {symbol} on ds {ds}.")
                self._get_one_symbol(ds=ds, symbol=symbol, enable_sentiment=enable_sentiment)
            except Exception as e:
                logger.error(traceback.format_exception(e))
            self.rate_limiter_sleep(timer_start, loops_per_second_min=0.1, loops_per_second_max=0.3)

        self._clean_up_data(ds)

    def _get_downloaded_symbols(self, ds):
        return GubaEm(ds=ds, symbol="").get_downloaded_symbols()

    def _get_one_symbol(self, ds, symbol, enable_sentiment):
        data = GubaEm(ds=ds, symbol=symbol, enable_sentiment=enable_sentiment)
        data.retrieve_data()

    def _clean_up_data(self, ds):
        GubaEm(ds=ds, symbol="").clean_up_history(lifecycle=15)


if __name__ == '__main__':
    ds = sys.argv[1]
    symbol_type = sys.argv[2] if len(sys.argv) > 2 else "position"
    logger.info("execute task on ds {}, symbol type {}".format(ds, symbol_type))
    DataHelper().get_all_symbols(ds, symbol_type=symbol_type, enable_sentiment=True)
