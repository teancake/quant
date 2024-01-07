import sys, os
parent_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parent_dir)


from utils.log_util import get_logger
from utils.starrocks_db_util import StarrocksDbUtil, generate_partition_spec
import sys
logger = get_logger(__name__)
from utils.stock_zh_a_util import is_trade_date


if __name__ == '__main__':
    db = StarrocksDbUtil()
    ds = sys.argv[1]
    logger.info("execute task on ds {}".format(ds))
    if not is_trade_date(ds):
        logger.info(f"{ds} is not trade date. task exits.")
        exit(os.EX_OK)

    partition_str = generate_partition_spec(ds)
    ddl_sql_str = '''
CREATE TABLE if not exists `external_stock_individual_info_em` (
  `gmt_create` datetime ,
  `gmt_modified` datetime ,
  `股票代码` varchar ,
  `股票简称` varchar ,
  `行业` varchar ,
  `上市时间` varchar ,
  `总股本` varchar ,
  `流通股` varchar ,
  `总市值` varchar ,
  `流通市值` varchar ,
  `ds` varchar 
)ENGINE = mysql 
PROPERTIES
(
"host" = "192.168.50.100",
"port" = "3306",
"user" = "quant",
"password" = "quant",
"database" = "akshare_data",
"table" = "stock_individual_info_em"
);


CREATE TABLE if not exists `ods_stock_individual_info_em` ( 
  `gmt_create` datetime ,
  `gmt_modified` datetime ,
  `代码` varchar(50) ,
  `简称` varchar(50) ,
  `行业` varchar(50) ,
  `上市时间` varchar(50) ,
  `总股本` varchar(50) ,
  `流通股` varchar(50) ,
  `总市值` varchar(50) ,
  `流通市值` varchar(50) ,
  `ds` date
 )
PARTITION BY RANGE(ds)(
    {}
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


CREATE TABLE if not exists `dwd_stock_individual_info_em_df` ( 
  `gmt_create` datetime ,
  `gmt_modified` datetime ,
  `代码` varchar(50) ,
  `简称` varchar(50) ,
  `行业` varchar(50) ,
  `上市时间` varchar(50) ,
  `总股本` varchar(50) ,
  `流通股` varchar(50) ,
  `总市值` varchar(50) ,
  `流通市值` varchar(50) ,
  `ds` date
 )
PARTITION BY RANGE(ds)(
{}
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
'''.format(partition_str, partition_str)

    ods_sql_str = '''
SET SESSION query_timeout=600;
INSERT OVERWRITE ods_stock_individual_info_em PARTITION(p{})
select
`gmt_create`,
`gmt_modified`,
`股票代码` as 代码,
`股票简称`as 简称,
`行业`,
`上市时间`,
`总股本`,
`流通股`,
`总市值`,
`流通市值`,
'{}' as ds
from 
external_stock_individual_info_em 
;
    '''.format(ds, ds)

    dwd_sql_str = '''
SET SESSION query_timeout=600;
INSERT OVERWRITE dwd_stock_individual_info_em_df PARTITION(p{})
select
`gmt_create`,
`gmt_modified`,
`代码`,
`简称`,
`行业`,
`上市时间`,
`总股本`,
`流通股`,
`总市值`,
`流通市值`,
ds
from
(select *, row_number() over(partition by 代码 order by gmt_create desc) as rn from 
ods_stock_individual_info_em
where ds = '{}'
)a
where rn = 1
order by 代码
;
        '''.format(ds, ds)

    db.run_sql(ddl_sql_str)
    logger.info("ddl sql finished.")
    db.run_sql(ods_sql_str)
    logger.info("ods sql finished.")
    db.run_sql(dwd_sql_str)
    logger.info("dwd sql finished.")




