from abc import ABC, abstractmethod
from utils.starrocks_db_util import StarrocksDbUtil

from datetime import datetime, timedelta

from utils.log_util import get_logger
logger = get_logger(__name__)



import json
import traceback
from ollama import Client
import pandas as pd
import re
from utils.config_util import get_ollama_config


def generate_partition_spec(ds: str):
    ds_m30 = (datetime.strptime(ds, '%Y%m%d') - timedelta(days=30)).strftime("%Y%m%d")
    ds_p7 = (datetime.strptime(ds, '%Y%m%d') + timedelta(days=7)).strftime("%Y%m%d")
    partition_str = "START ('{}') END ('{}') EVERY (INTERVAL 1 day)".format(ds_m30, ds_p7)
    return partition_str

class BaseUdfRunner(ABC):
    def __init__(self, input_table, output_table, input_cols, output_cols, ds):
        self.input_table = input_table
        self.output_table = output_table
        self.input_cols = input_cols
        self.output_cols = output_cols
        self.ds = ds
        self.db = StarrocksDbUtil()


    @abstractmethod
    def func(self, *args):
        pass

    @abstractmethod
    def input_func(self):
        pass
    def run(self):
        df = self.input_func()
        for output in self.output_cols:
            df[output] = ""
        batch_size = 10
        for index, row in df.iterrows():
            logger.info(f"index {index} of {len(df)}, row data {row[self.input_cols].to_dict()}")
            res = self.func(*row[self.input_cols].to_list())
            for output in self.output_cols:
                df.loc[index, output] = res.get(output, "")

            if index % batch_size == 0:
                start_index = max(index - batch_size, 0)
                logger.info(f"start index {start_index}, index {index}")
                df.iloc[start_index:index].to_sql(name=self.output_table, con=self.db.engine, if_exists='append',
                                                  index=False, method="multi", chunksize=10000)
                logger.info(f"{len(df.iloc[start_index:index])} records written to db")
        return df


    @abstractmethod
    def get_select_clause(self):
        pass

    def generate_final_table(self, table_name, lifecycle=365):
        select_clause = self.get_select_clause()
        if not self.db.table_exists(table_name):
            logger.info(f"create table {table_name}")
            temp_sql = f"""
            CREATE TABLE {table_name}
            PARTITION BY RANGE(ds)({generate_partition_spec(self.ds)})
            DISTRIBUTED BY HASH(ds) BUCKETS 32
            PROPERTIES(
                "replication_num" = "1",
                "dynamic_partition.enable" = "true",
                "dynamic_partition.time_unit" = "DAY",
                "dynamic_partition.start" = "-{lifecycle}",
                "dynamic_partition.end" = "7",
                "dynamic_partition.prefix" = "p",
                "dynamic_partition.buckets" = "32"
            )
            AS
            """
        else:
            logger.info(f"table {table_name} exists, insert data into a new partition")
            temp_sql = f"""
            INSERT OVERWRITE {table_name} PARTITION(p{self.ds})
            """
        temp_sql += select_clause
        self.db.run_sql(temp_sql)
        logger.info(f"table {table_name} created or updated from selection.")



class SentimentUdfRunner(BaseUdfRunner):
    def __init__(self, ds):
        ollama_conf = get_ollama_config()
        self.client = Client(host=f"http://{ollama_conf['server_address']}:{ollama_conf['port']}")
        self.model = "qwen:14b"

        input_table = "dwd_guba_em_di"
        output_table = "temp_dwd_guba_em_di_udf"
        input_cols = ["post_title"]
        output_cols = ["sentiment", "reason"]
        ds = ds
        super().__init__(input_table, output_table, input_cols, output_cols, ds)

    def input_func(self):
        sql = f"SELECT * FROM {self.input_table} WHERE ds = {self.ds} and cast(post_last_time as date) = ds order by symbol asc"
        df = pd.DataFrame(self.db.run_sql(sql))
        return df

    def func(self, *args):
        return self.sync_chat(*args)

    def sync_chat(self, *args):
        message = args[0]
        try:
            cache = self.get_cache(message)
            if cache is not None:
                logger.info("hit cache, no further computation.")
                return cache
            response = self.client.chat(model=self.model, messages=[
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

    def get_cache(self, content):
        sql = f"select * from {self.output_table} where {self.input_cols[0]} = '{content}' order by gmt_create desc limit 1"
        res = self.db.run_sql(sql, log=False)
        if len(res) == 0:
            return None
        df = pd.DataFrame(res)
        return df.loc[0, self.output_cols].to_dict()


    def get_select_clause(self):

        sql = f"""
        select
            a.symbol,
            a.sentiment,
            a.post_count,
            a.click_count,
            a.user_count,
            a.per_post_click_count,
            b.post_count as total_post_count,
            b.click_count as total_click_count,
            b.user_count as total_user_count,
            b.per_post_click_count as total_per_post_click_count,
            a.post_count/b.post_count as post_ratio,
            a.click_count/b.click_count as click_ratio,
            2/(1/(a.post_count/b.post_count)+1/(a.click_count/b.click_count)) as avg_ratio,
            a.ds 
        from
            (select
                ds,
                symbol,
                sentiment,
                count(*) as post_count,
                sum(post_click_count) as click_count,
                count(DISTINCT user_id) as user_count,
                sum(post_click_count)/count(*) as per_post_click_count 
            from
            (
                select *, row_number() over (partition by symbol, post_id order by gmt_create desc) as rn
                from {self.output_table}  
                where cast(post_last_time as date) = ds
                and ds = '{self.ds}'
            ) aa where rn=1
            group by
                ds,
                symbol,
                sentiment )a  
        join
            (
                select
                    ds,
                    symbol,
                    count(*) as post_count,
                    sum(post_click_count) as click_count,
                    count(DISTINCT user_id) as user_count,
                    sum(post_click_count)/count(*) as per_post_click_count 
                from
                (
                    select *, row_number() over (partition by symbol, post_id order by gmt_create desc) as rn
                    from {self.output_table}  
                    where cast(post_last_time as date) = ds
                    and ds = '{self.ds}'
                ) aa where rn=1
                group by
                    ds,
                    symbol
            )b 
                on a.symbol = b.symbol
        """
        return sql
