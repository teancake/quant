import sys, os
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(parent_dir)


from utils.log_util import get_logger
from utils.starrocks_db_util import StarrocksDbUtil, generate_partition_spec, get_days_ahead_ds
from utils.stock_zh_a_util import is_trade_date
import sys

from utils.db_util import get_mysql_config


logger = get_logger(__name__)

if __name__ == '__main__':
    db = StarrocksDbUtil()
    ds = sys.argv[1]
    logger.info("execute task on ds {}".format(ds))
    if not is_trade_date(ds):
        logger.info(f"{ds} is not trade date. task exits.")
        exit(os.EX_OK)

    partition_str = generate_partition_spec(ds)
    server_address, port, db_name, user, password = get_mysql_config()
    ddl_sql_str = f'''
 CREATE TABLE IF NOT EXISTS `external_index_zh_a_hist` ( 
`gmt_create` datetime DEFAULT NULL,
`gmt_modified` datetime DEFAULT NULL,
`代码` varchar DEFAULT NULL,
`symbol` varchar DEFAULT NULL,
`period` varchar DEFAULT NULL,
`日期` varchar DEFAULT NULL,
`开盘` double DEFAULT NULL,
`收盘` double DEFAULT NULL,
`最高` double DEFAULT NULL,
`最低` double DEFAULT NULL,
`成交量` bigint(20) DEFAULT NULL,
`成交额` double DEFAULT NULL,
`振幅` double DEFAULT NULL,
`涨跌幅` double DEFAULT NULL,
`涨跌额` double DEFAULT NULL,
`换手率` double DEFAULT NULL,
`ds` varchar DEFAULT NULL
) ENGINE = mysql 
PROPERTIES
(
"host" = "{server_address}",
"port" = "{port}",
"user" = "{user}",
"password" = "{password}",
"database" = "{db_name}",
"table" = "index_zh_a_hist"
);

    

CREATE TABLE if not exists `ods_index_zh_a_hist` ( 
`gmt_create` datetime DEFAULT NULL,
`gmt_modified` datetime DEFAULT NULL,
`代码` varchar(20) DEFAULT NULL,
`symbol` varchar(20) DEFAULT NULL,
`period` varchar(20) DEFAULT NULL,
`日期` varchar(20) DEFAULT NULL,
`开盘` double DEFAULT NULL,
`收盘` double DEFAULT NULL,
`最高` double DEFAULT NULL,
`最低` double DEFAULT NULL,
`成交量` bigint(20) DEFAULT NULL,
`成交额` double DEFAULT NULL,
`振幅` double DEFAULT NULL,
`涨跌幅` double DEFAULT NULL,
`涨跌额` double DEFAULT NULL,
`换手率` double DEFAULT NULL,
`ds` date DEFAULT NULL
) 
PARTITION BY RANGE(ds)(
{partition_str}
)
DISTRIBUTED BY HASH(ds) BUCKETS 32
PROPERTIES(
    "replication_num" = "1",
    "dynamic_partition.enable" = "true",
    "dynamic_partition.time_unit" = "DAY",
    "dynamic_partition.start" = "-10",
    "dynamic_partition.end" = "7",
    "dynamic_partition.prefix" = "p",
    "dynamic_partition.buckets" = "32"
)
;

CREATE TABLE if not exists `dwd_index_zh_a_hist_df` ( 
`gmt_create` datetime DEFAULT NULL,
`gmt_modified` datetime DEFAULT NULL,
`代码` varchar(20) DEFAULT NULL,
`symbol` varchar(20) DEFAULT NULL,
`period` varchar(20) DEFAULT NULL,
`日期` varchar(20) DEFAULT NULL,
`开盘` double DEFAULT NULL,
`收盘` double DEFAULT NULL,
`最高` double DEFAULT NULL,
`最低` double DEFAULT NULL,
`成交量` bigint(20) DEFAULT NULL,
`成交额` double DEFAULT NULL,
`振幅` double DEFAULT NULL,
`涨跌幅` double DEFAULT NULL,
`涨跌额` double DEFAULT NULL,
`换手率` double DEFAULT NULL,
`ds` date DEFAULT NULL
 )
PARTITION BY RANGE(ds)(
{partition_str}
)
DISTRIBUTED BY HASH(ds) BUCKETS 32
PROPERTIES(
    "replication_num" = "1",
    "dynamic_partition.enable" = "true",
    "dynamic_partition.time_unit" = "DAY",
    "dynamic_partition.start" = "-10",
    "dynamic_partition.end" = "7",
    "dynamic_partition.prefix" = "p",
    "dynamic_partition.buckets" = "32"
)
;
'''

    ods_sql_str = '''
SET SESSION query_timeout=1200;
INSERT OVERWRITE ods_index_zh_a_hist PARTITION(p{})
select
`gmt_create`,
`gmt_modified`,
`代码`,
`symbol`,
`period`,
`日期`,
`开盘`,
`收盘`,
`最高`,
`最低`,
`成交量`,
`成交额`,
`振幅`,
`涨跌幅`,
`涨跌额`,
`换手率`,
'{}' as ds
from 
external_index_zh_a_hist 
where ds >= '{}'
;
    '''.format(ds, ds, get_days_ahead_ds(ds, 15))

    dwd_sql_str = '''
SET SESSION query_timeout=600;
INSERT OVERWRITE dwd_index_zh_a_hist_df PARTITION(p{})
select 
`gmt_create`,
`gmt_modified`,
`代码`,
`symbol`,
`period`,
`日期`,
`开盘`,
`收盘`,
`最高`,
`最低`,
`成交量`,
`成交额`,
`振幅`,
`涨跌幅`,
`涨跌额`,
`换手率`,
ds
from (
select *, row_number()over (partition by symbol, period, 日期 ORDER by gmt_create desc) as rn
FROM ods_index_zh_a_hist
where ds = '{}'
)a
where rn = 1
order by 代码, 日期
;
        '''.format(ds, ds)

    db.run_sql(ddl_sql_str)
    logger.info("ddl sql finished.")
    db.run_sql(ods_sql_str)
    logger.info("ods sql finished.")
    db.run_sql(dwd_sql_str)
    logger.info("dwd sql finished.")

    db.dqc_row_count("dwd_index_zh_a_hist_df", ds)

