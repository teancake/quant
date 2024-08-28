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
import random
import traceback
from joblib import Parallel, delayed

logger = get_logger(__name__)
from tqdm import tqdm


class BaseData(ABC):

    def __init__(self, ds=None):
        self.db = DbUtil()
        self.engine = self.db.get_db_engine()
        self.table_name = self.get_table_name()
        self.ds = self.generate_default_ds() if ds is None else ds

    def get_ds_ahead(self, ds, days):
        return (datetime.strptime(ds, '%Y%m%d') - timedelta(days=days)).strftime("%Y%m%d")

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
                logger.warning("api request exception {}".format(e))
                delay_seconds = (i + 1) * backoff_factor
                logger.warning("delay {} seconds, retry {} of {}".format(delay_seconds, i+1, retry_times))
                time.sleep(delay_seconds)
        raise Exception("max retry times reached, there might be a real problem, exception raised.")

    def retrieve_data(self):
        self.before_retrieve_data()
        df_schema = self.get_df_schema_with_retry()
        if len(df_schema) == 0:
            logger.info("data size 0, no db write required.")
            return
        logger.info("data retrieved.")
        dt = datetime.now().astimezone(pytz.timezone("Asia/Shanghai"))
        df_schema.insert(0, "gmt_create", dt)
        df_schema.insert(1, "gmt_modified", dt)
        df_schema.insert(len(df_schema.columns), "ds", self.ds)
        df_schema.to_sql(name=self.table_name, con=self.engine, if_exists='append', index=False, method="multi", chunksize=5000)
        self.add_ds_index()
        logger.info("{} data records written to table {}, ds={}".format(df_schema.shape[0], self.table_name, self.ds))


    def clean_up_history(self, lifecycle=30):
        if not self.table_exists():
            logger.info(f"table {self.table_name} does not exist, clean up history aborted.")
            return
        ds_cl = (datetime.strptime(self.ds, '%Y%m%d') - timedelta(days=lifecycle)).strftime("%Y%m%d")
        self.delete_records("ds < {}".format(ds_cl))


class BaseDataHelper(ABC):
    def __init__(self, loops_per_second_min=1, loops_per_second_max=2, parallel=1):
        if loops_per_second_min is None:
            loops_per_second_min = loops_per_second_max
        self.loops_per_second_min = loops_per_second_min
        self.loops_per_second_max = loops_per_second_max
        self.parallel = parallel

    def rate_limiter_sleep(self, timer_start):
        # the reason to set loops_per_second_min is to have a random sleep time
        loops_per_second_min, loops_per_second_max = self.loops_per_second_min, self.loops_per_second_max
        if loops_per_second_max is None:
            return
        loop_time_second_min = 1.0 / loops_per_second_max
        loop_time_second_max = 1.0 / loops_per_second_min
        dt = time.time() - timer_start
        if dt < loop_time_second_min:
            random_time = random.random() * (loop_time_second_max - loop_time_second_min) + loop_time_second_min - dt
            logger.info(f"dt is {dt} less than minimum loop time {loop_time_second_min}, sleep {random_time} seconds")
            time.sleep(random_time)

    @abstractmethod
    def _get_downloaded_symbols(self):
        pass

    @abstractmethod
    def _get_all_symbols(self):
        pass

    @abstractmethod
    def _fetch_symbol_data(self, symbol):
        pass

    @abstractmethod
    def _clean_up_history(self):
        pass
    def fetch_all_data(self):
        symbol_list = self._get_all_symbols()
        downloaded_symbols = self._get_downloaded_symbols()
        logger.info(f"{len(downloaded_symbols)} of {len(symbol_list)} symbols already exist, only process the remaining")
        symbol_list = set(symbol_list) - set(downloaded_symbols)
        if self.parallel > 1:
            self._fetch_all_data_parallel(symbol_list, self.parallel)
        else:
            self._fetch_all_data_serial(symbol_list)

        self._clean_up_history()

    def _fetch_all_data_serial(self, symbol_list):
        for symbol in tqdm(symbol_list):
            timer_start = time.time()
            try:
                logger.info("process symbol {}".format(symbol))
                self._fetch_symbol_data(symbol)
                logger.info("symbol {} done".format(symbol))
            except Exception:
                logger.error(f"exception occurred while processing symbol {symbol}")
                logger.error(traceback.format_exc())
            self.rate_limiter_sleep(timer_start)

    def _fetch_all_data_parallel(self, symbol_list, parallel):
        Parallel(n_jobs=parallel, backend="multiprocessing")(delayed(self._fetch_symbol_data)(symbol) for symbol in tqdm(symbol_list))
