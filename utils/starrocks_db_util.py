import sqlalchemy

from utils.log_util import get_logger
from datetime import datetime, timedelta

logger = get_logger(__name__)
from utils.config_util import get_starrocks_config


def generate_partition_spec(ds:str):
    ds_m30 = (datetime.strptime(ds, '%Y%m%d') - timedelta(days=30)).strftime("%Y%m%d")
    ds_p7 = (datetime.strptime(ds, '%Y%m%d') + timedelta(days=7)).strftime("%Y%m%d")
    partition_str = "START ('{}') END ('{}') EVERY (INTERVAL 1 day)".format(ds_m30, ds_p7)
    return partition_str


def get_days_ahead_ds(ds, days):
    return (datetime.strptime(ds, '%Y%m%d') - timedelta(days=days)).strftime("%Y%m%d")



class StarrocksDbUtil:
    def __init__(self):
        self.engine = None

    def get_db_engine(self):
        if self.engine is None:
            self.engine = self._create_db_engine()
        return self.engine

    def _create_db_engine(self):
        server_address, port, db_name, user, password = get_starrocks_config()
        return sqlalchemy.create_engine(f"starrocks://{user}:{password}@{server_address}:{port}/{db_name}?charset=utf8", connect_args={'connect_timeout': 600})


    def table_exists(self, table_name):
        return sqlalchemy.inspect(self.get_db_engine()).has_table(table_name)


    def run_sql(self, sql, log=True, chunksize=None):
        if log:
            logger.info("run sql: {}".format(sql))
        sql = sqlalchemy.text(sql)
        result = []
        with self.get_db_engine().connect() as con:
            try:
                if chunksize:
                    # cursor_result = con.execution_options(yield_per=1).execute(sql)
                    # for partition in cursor_result.partitions():
                    #     for row in partition:
                    #         result.append(row)
                    cursor_result = con.execution_options(stream_results=True, max_row_buffer=100).execute(sql)
                    while True:
                        data = cursor_result.fetchmany(chunksize)
                        if not data:
                            break
                        result.extend(list(data))
                else:
                    cursor_result = con.execute(sql)
                    result = list(cursor_result)
            except Exception as e:
                # if there are no results returned, exception will be ignored.
                logger.info(e)

        return result

    def dqc_row_count(self, table_name, ds):
        dqc_sql = "select count(*) from {} where ds = '{}'".format(table_name, ds)
        row_count = self.run_sql(dqc_sql)[0][0]
        if row_count == 0:
            raise Exception("data quality check failed: row count is 0. table {}, ds {}".format(table_name, ds))

        logger.info("data quality check successful: table {}, ds {}, row_count {}".format(table_name, ds, row_count))




