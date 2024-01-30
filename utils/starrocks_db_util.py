import sqlalchemy

from utils.log_util import get_logger
from datetime import datetime, timedelta

logger = get_logger(__name__)
from utils.config_util import get_starrocks_config
from utils.db_util import get_table_columns, get_mysql_config

# CREATE DATABASE akshare_data;
# CREATE USER 'quant'@'%' IDENTIFIED BY 'quant';
# GRANT ALL PRIVILEGES ON DATABASE akshare_data TO 'quant'@'%'  WITH GRANT OPTION;



# backup and restore
# CREATE REPOSITORY akshare_data_backup
# WITH BROKER
# ON LOCATION "oss://bourgogne/starrocks_backup"
# PROPERTIES(
#     "fs.oss.accessKeyId" = "LTAxxxxx"
#     "fs.oss.accessKeySecret" = "xxxx",
#     "fs.oss.endpoint" = "oss-cn-beijing.aliyuncs.com"
# );
#
#
# BACKUP SNAPSHOT akshare_data.stock_zh_a_transaction_backup
# TO akshare_data_backup
# ON (stock_zh_a_transaction);
#
#
# BACKUP SNAPSHOT akshare_data.ods_stock_zh_a_prediction_backup
# TO akshare_data_backup
# ON (ods_stock_zh_a_prediction);
#
# BACKUP SNAPSHOT akshare_data.ads_stock_zh_a_position_backup
# TO akshare_data_backup
# ON (ads_stock_zh_a_position);
#
# BACKUP SNAPSHOT akshare_data.ads_stock_zh_a_pred_data_backup
# TO akshare_data_backup
# ON (ads_stock_zh_a_pred_data);
#
#
# SHOW BACKUP;
#
#
# SHOW SNAPSHOT ON akshare_data_backup;
#
#
#
# RESTORE SNAPSHOT akshare_data.stock_zh_a_transaction_backup
# FROM akshare_data_backup
# ON (stock_zh_a_transaction)
# PROPERTIES (
#     "backup_timestamp"="2024-01-28-18-24-11-465",
#     "replication_num" = "1"
# );
#
#
# RESTORE SNAPSHOT akshare_data.ods_stock_zh_a_prediction_backup
# FROM akshare_data_backup
# ON (ods_stock_zh_a_prediction)
# PROPERTIES (
#     "backup_timestamp"="2024-01-28-18-36-31-019",
#     "replication_num" = "1"
# );
#
#
# SHOW RESTORE;


def generate_partition_spec(ds:str):
    ds_m30 = (datetime.strptime(ds, '%Y%m%d') - timedelta(days=30)).strftime("%Y%m%d")
    ds_p7 = (datetime.strptime(ds, '%Y%m%d') + timedelta(days=7)).strftime("%Y%m%d")
    partition_str = "START ('{}') END ('{}') EVERY (INTERVAL 1 day)".format(ds_m30, ds_p7)
    return partition_str


def get_days_ahead_ds(ds, days):
    return (datetime.strptime(ds, '%Y%m%d') - timedelta(days=days)).strftime("%Y%m%d")


def mysql_to_ods_dwd(mysql_table_name, ds, di_df="di", unique_columns=None):
    cols, col_names = get_table_columns(mysql_table_name, ignore_ds=True)
    mysql_columns_str = ", ".join(cols)
    mysql_column_names_str = ", ".join(col_names)
    ods_table_name = f"ods_{mysql_table_name}"
    dwd_table_name = f"dwd_{mysql_table_name}_{di_df}"

    if unique_columns is not None and len(unique_columns) > 0:
        unique_columns_str = ",".join(unique_columns)
    else:
        unique_columns_str = mysql_column_names_str

    server_address, port, db_name, user, password = get_mysql_config()

    ods_sql_str = f'''
        CREATE TABLE IF NOT EXISTS `external_{mysql_table_name}` ( 
         {mysql_columns_str}
        )
        ENGINE = mysql 
        PROPERTIES
        (
        "host" = "{server_address}",
        "port" = "{port}",
        "user" = "{user}",
        "password" = "{password}",
        "database" = "{db_name}",
        "table" = "{mysql_table_name}"
        );

        CREATE TABLE IF NOT EXISTS {ods_table_name} (
        {mysql_columns_str},
        ds date
        ) 
        PARTITION BY RANGE(ds)({generate_partition_spec(ds)})
        DISTRIBUTED BY HASH(ds) BUCKETS 32
        PROPERTIES(
            "replication_num" = "1",
            "dynamic_partition.enable" = "true",
            "dynamic_partition.time_unit" = "DAY",
            "dynamic_partition.start" = "-62",
            "dynamic_partition.end" = "7",
            "dynamic_partition.prefix" = "p",
            "dynamic_partition.buckets" = "32"
        )
        ;

        INSERT OVERWRITE {ods_table_name} PARTITION(p{ds})
        select 
        {mysql_column_names_str},
        {ds} as ds
        from 
        external_{mysql_table_name}
        where DATE_FORMAT(gmt_create, '%Y%m%d') = {ds}
        ;
        '''

    dwd_sql_str = f'''
        CREATE TABLE IF NOT EXISTS {dwd_table_name} LIKE {ods_table_name};
        INSERT OVERWRITE {dwd_table_name} PARTITION(p{ds})'''

    ds_start = ds if di_df == "di" else get_days_ahead_ds(ds, 8)

    select_str = f'''
        select 
        {mysql_column_names_str},
        {ds} as ds 
        from (select *,
        row_number() over (partition by {unique_columns_str} order by gmt_create desc) as rn
        from 
        {ods_table_name} 
        where ds >= '{ds_start}'
        )a where rn = 1
    '''

    dwd_sql_str += select_str
    db = StarrocksDbUtil()
    db.run_sql(ods_sql_str)
    logger.info("ods sql finished.")
    db.dqc_row_count(ods_table_name, ds)
    db.run_sql(dwd_sql_str)
    logger.info("dwd sql finished.")
    db.dqc_row_count(dwd_table_name, ds)




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




