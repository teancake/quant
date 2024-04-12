import sys, os
parent_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parent_dir)


import time
from abc import ABC, abstractmethod

from utils.log_util import get_logger
from utils.db_util import DbUtil
from datetime import datetime, timedelta
import pytz

from requests.exceptions import RequestException

logger = get_logger(__name__)



class BaseData(ABC):

    def __init__(self):
        self.db = DbUtil()
        self.engine = self.db.get_db_engine()
        self.table_name = self.get_table_name()
        self.ds = self.generate_default_ds()

    def table_exists(self):
        sql = f"SHOW TABLES LIKE '{self.table_name}'"
        res = self.db.run_sql(sql)
        return len(res) > 0

    def ds_exists(self, ds):
        results = self.db.run_sql("SELECT count(*) from {} where ds = {}".format(self.get_table_name(), ds))
        return len(results) > 0 and results[0][0] > 0

    def get_max_ds(self):
        results = self.db.run_sql("SELECT max(ds) from {}".format(self.get_table_name()))
        return results[0][0] if len(results) > 0 else None

    def delete_records(self, conditions):
        results = self.db.run_sql("delete from {} where {}".format(self.get_table_name(), conditions))
        print(results)

    def generate_default_ds(self):
        return datetime.now().strftime("%Y%m%d")

    def set_ds(self, ds):
        self.ds = ds

    def add_ds_index(self):
        self.db.run_sql("ALTER TABLE {} ADD INDEX IF NOT EXISTS (ds);".format(self.table_name))

    @abstractmethod
    def get_df_schema(self):
        pass

    @abstractmethod
    def get_table_name(self):
        pass

    @abstractmethod
    def before_retrieve_data(self):
        pass

    def get_df_schema_with_retry(self, retry_times=5, backoff_factor=30):
        for i in range(retry_times + 1):
            try:
                df_schema = self.get_df_schema()
                return df_schema
            except RequestException as e:
                logger.warn("api request exception {}".format(e))
                delay_seconds = (i + 1) * backoff_factor
                logger.warn("delay {} seconds, retry {} of {}".format(delay_seconds, i+1, retry_times))
                time.sleep(delay_seconds)
        raise Exception("max retry times reached, there might be a real problem, exception raised.")

    def retrieve_data(self):
        self.before_retrieve_data()
        df_schema = self.get_df_schema_with_retry()
        logger.info("data retrieved.")
        dt = datetime.now().astimezone(pytz.timezone("Asia/Shanghai"))
        df_schema.insert(0, "gmt_create", dt)
        df_schema.insert(1, "gmt_modified", dt)
        df_schema.insert(len(df_schema.columns), "ds", self.ds)
        df_schema.to_sql(name=self.table_name, con=self.engine, if_exists='append', index=False, method="multi", chunksize=10000)
        self.add_ds_index()
        logger.info("{} data records written to table {}, ds={}".format(df_schema.shape[0], self.table_name, self.ds))


    def clean_up_history(self, lifecycle=30):
        ds_cl = (datetime.strptime(self.ds, '%Y%m%d') - timedelta(days=lifecycle)).strftime("%Y%m%d")
        self.delete_records("ds < {}".format(ds_cl))