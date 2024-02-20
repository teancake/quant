import sys, os
parent_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
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
CREATE TABLE if not exists `external_stock_zh_a_hist` ( 
`gmt_create` datetime ,
 `gmt_modified` datetime ,
 `symbol` varchar ,
 `period` varchar ,
 `adjust` varchar ,
 `日期` date ,
 `开盘` double ,
 `收盘` double ,
 `最高` double ,
 `最低` double ,
 `成交量` bigint(20) ,
 `成交额` double ,
 `振幅` double ,
 `涨跌幅` double ,
 `涨跌额` double ,
 `换手率` double ,
 `ds` varchar ) 
ENGINE = mysql 
PROPERTIES
(
"host" = "{server_address}",
"port" = "{port}",
"user" = "{user}",
"password" = "{password}",
"database" = "{db_name}",
"table" = "stock_zh_a_hist"
);

    

CREATE TABLE if not exists `ods_stock_zh_a_hist` ( 
`gmt_create` datetime ,
 `gmt_modified` datetime ,
 `symbol` varchar(20) ,
 `period` varchar(20) ,
 `adjust` varchar(20) ,
 `日期` date ,
 `开盘` double ,
 `收盘` double ,
 `最高` double ,
 `最低` double ,
 `成交量` bigint(20) ,
 `成交额` double ,
 `振幅` double ,
 `涨跌幅` double ,
 `涨跌额` double ,
 `换手率` double ,
 `ds` date) 
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

CREATE TABLE if not exists `dwd_stock_zh_a_hist_df` ( 
`gmt_create` datetime ,
 `gmt_modified` datetime ,
 `代码` varchar(20) ,
 `period` varchar(20) ,
 `adjust` varchar(20) ,
 `日期` date ,
 `开盘` double ,
 `收盘` double ,
 `最高` double ,
 `最低` double ,
 `成交量` bigint(20) ,
 `成交额` double ,
 `振幅` double ,
 `涨跌幅` double ,
 `涨跌额` double ,
 `换手率` double ,
 `ds` date
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
INSERT OVERWRITE ods_stock_zh_a_hist PARTITION(p{})
select
gmt_create,
gmt_modified,
symbol,
period,
adjust,
日期,
开盘,
收盘,
最高,
最低,
成交量,
成交额,
振幅,
涨跌幅,
涨跌额,
换手率,
'{}' as ds
from 
external_stock_zh_a_hist 
where ds >= '{}'
;
    '''.format(ds, ds, get_days_ahead_ds(ds, 15))

    dwd_sql_str = '''
SET SESSION query_timeout=600;
INSERT OVERWRITE dwd_stock_zh_a_hist_df PARTITION(p{})
select 
gmt_create,
gmt_modified,
symbol as 代码,
period,
adjust,
日期,
开盘,
收盘,
最高,
最低,
成交量,
成交额,
振幅,
涨跌幅,
涨跌额,
换手率,
ds
from (
select *, row_number()over (partition by symbol, period, adjust, 日期 ORDER by gmt_create desc) as rn
FROM ods_stock_zh_a_hist
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

    db.dqc_row_count("dwd_stock_zh_a_hist_df", ds)

