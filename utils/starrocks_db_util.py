import sqlalchemy

from utils.log_util import get_logger
from datetime import datetime, timedelta

logger = get_logger(__name__)
from utils.config_util import get_starrocks_config
from utils.db_util import get_table_columns, get_mysql_config

# CREATE DATABASE akshare_data;
# CREATE USER 'quant'@'%' IDENTIFIED BY 'quant';
# GRANT ALL PRIVILEGES ON DATABASE akshare_data TO 'quant'@'%'  WITH GRANT OPTION;
# GRANT OPERATE ON SYSTEM TO USER 'quant'@'%' WITH GRANT OPTION;


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

def _get_cols_str(cols, rename_cols):
    if rename_cols is None:
        rename_cols = {}
    def escape(key):
        return f"`{key}`"

    mysql_columns_str = ", ".join([f"{escape(key)} {value}" for key, value in cols.items()])
    mysql_column_names_str = ", ".join([f"{escape(key)}" for key in cols.keys()])
    dw_columns_str = ", ".join([f"{escape(rename_cols.get(key, key))} {value}" for key, value in cols.items() if key != "ds"])
    dw_column_names_str = ", ".join([f"{escape(rename_cols.get(key, key))}" for key in cols.keys() if key != "ds"])
    rename_column_names_str = ", ".join([f"{escape(key)} AS {rename_cols.get(key)}" if key in rename_cols.keys() else escape(key) for key in cols.keys() if key != "ds"])
    return mysql_columns_str, mysql_column_names_str, dw_columns_str, dw_column_names_str, rename_column_names_str


def mysql_to_ods_dwd(mysql_table_name, ds, di_df="di", unique_columns=None, lifecycle=62, days_ahead=15,
                     rename_columns=None, use_mysql_table_ds=True, mysql_where_cond=None, ods_dqc=True):
    cols = get_table_columns(mysql_table_name)
    mysql_columns_str, _, dw_columns_str, dw_column_names_str, rename_column_names_str = _get_cols_str(cols, rename_columns)
    ods_table_name = f"ods_{mysql_table_name}"
    dwd_table_name = f"dwd_{mysql_table_name}_{di_df}"

    if unique_columns is not None and len(unique_columns) > 0:
        unique_columns_str = ",".join(unique_columns)
    else:
        unique_columns_str = dw_column_names_str

    server_address, port, db_name, user, password = get_mysql_config()

    if mysql_where_cond is None or mysql_where_cond == "":
        if use_mysql_table_ds:
            mysql_where_cond = f"ds = {ds}"
        else:
            mysql_where_cond = f"DATE_FORMAT(gmt_create, '%Y%m%d') = {ds}"
    mysql_where_cond = f"WHERE {mysql_where_cond}"

    ods_ddl_sql_str = f'''
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
        {dw_columns_str},
        ds date
        ) 
        PARTITION BY RANGE(ds)({generate_partition_spec(ds)})
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
        ;
    '''
    ods_insert_sql_str = f'''
        INSERT OVERWRITE {ods_table_name} PARTITION(p{ds})
        SELECT 
        {rename_column_names_str},
        {ds} AS ds
        FROM 
        external_{mysql_table_name}
        {mysql_where_cond}
        ;
        '''

    dwd_sql_str = f'''
        CREATE TABLE IF NOT EXISTS {dwd_table_name} LIKE {ods_table_name};
        INSERT OVERWRITE {dwd_table_name} PARTITION(p{ds})'''

    if di_df == "di":
        where_cond = f"where ds = '{ds}'"
    else:
        where_cond = f"where ds >= '{get_days_ahead_ds(ds, days_ahead)}'"

    select_str = f'''
        SELECT 
        {dw_column_names_str},
        '{ds}' AS ds 
        FROM (
            SELECT *,
            ROW_NUMBER() OVER (PARTITION BY {unique_columns_str} ORDER BY gmt_create DESC) AS rn
            FROM 
            {ods_table_name} 
            {where_cond}
        ) a 
        WHERE rn = 1
    '''

    dwd_sql_str += select_str
    db = StarrocksDbUtil()
    db.run_sql(ods_ddl_sql_str)
    db.run_sql(ods_insert_sql_str)
    if ods_dqc:
        db.dqc_row_count(ods_table_name, ds)
    else:
        ods_row_count = db.run_sql(f"select count(*) from {ods_table_name} where ds ='{ds}'")[0][0]
        logger.info(f"ods sql finished. row count {ods_row_count}.")
    db.run_sql(dwd_sql_str)
    logger.info("dwd sql finished.")
    db.dqc_row_count(dwd_table_name, ds)
    return dwd_table_name


def content_to_rag(content_sql, content_table_name, ds):
    rag_table_name = "dwd_quant_rag_di"
    rag_ddl = f"""
    CREATE TABLE IF NOT EXISTS {rag_table_name} (
        `gmt_create` datetime NULL COMMENT "",
        `gmt_modified` datetime NULL COMMENT "", 
        `source` string NULL COMMENT "",
        `id` string NULL COMMENT "",
        `pub_time` STRING NULL COMMENT "", 
        `content` string NULL COMMENT "", 
        `ds` date
        ) 
        PARTITION BY RANGE(ds)({generate_partition_spec(ds)})
        DISTRIBUTED BY HASH(ds) BUCKETS 32
        PROPERTIES(
            "replication_num" = "1",
            "dynamic_partition.enable" = "true",
            "dynamic_partition.time_unit" = "DAY",
            "dynamic_partition.start" = "-3650",
            "dynamic_partition.end" = "7",
            "dynamic_partition.prefix" = "p",
            "dynamic_partition.buckets" = "32"
        )
        ;
    """
    db = StarrocksDbUtil()
    if not db.table_exists(rag_table_name):
        logger.info(f"rag table {rag_table_name} does not exist, create it.")
        db.run_sql(rag_ddl)

    logger.info("prepare content")
    db.run_sql(content_sql)
    logger.info("prepare content done")

    insert_sql = f"""
    INSERT INTO {rag_table_name} PARTITION(p{ds})
    SELECT gmt_create, gmt_modified, source, id, pub_time, content, ds
    from {content_table_name} where ds = {ds}
    """
    logger.info(f"insert content from {content_table_name} into rag table {rag_table_name}")
    db.run_sql(insert_sql)
    logger.info("insert done")


class StarrocksDbUtil:
    server_address, port, db_name, user, password = get_starrocks_config()
    engine = sqlalchemy.create_engine(f"starrocks://{user}:{password}@{server_address}:{port}/{db_name}?charset=utf8", connect_args={'connect_timeout': 600})

    @classmethod
    def get_db_engine(cls):
        return cls.engine

    @classmethod
    def table_exists(cls, table_name):
        sql = f"SHOW TABLES LIKE '{table_name}'"
        res = cls.run_sql(sql)
        return len(res) > 0

    @classmethod
    def run_sql(cls, sql, log=True, chunksize=None):
        if log:
            logger.info("run sql: {}".format(sql))
        sql = sqlalchemy.text(sql)
        result = []
        with cls.get_db_engine().connect() as con:
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
            except sqlalchemy.exc.ResourceClosedError as e:
                logger.info(e)

            # except Exception as e:
            #     # if there are no results returned, exception will be ignored.
            #     logger.info(e)

        return result

    @classmethod
    def dqc_row_count(cls, table_name, ds):
        dqc_sql = "select count(*) from {} where ds = '{}'".format(table_name, ds)
        row_count = cls.run_sql(dqc_sql)[0][0]
        if row_count == 0:
            raise Exception("data quality check failed: row count is 0. table {}, ds {}".format(table_name, ds))

        logger.info("data quality check successful: table {}, ds {}, row_count {}".format(table_name, ds, row_count))




