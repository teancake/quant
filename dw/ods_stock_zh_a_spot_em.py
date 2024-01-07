import sys, os
parent_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parent_dir)


from utils.log_util import get_logger
from utils.starrocks_db_util import StarrocksDbUtil
from utils.stock_zh_a_util import is_trade_date

import sys

logger = get_logger(__name__)

if __name__ == '__main__':
    db = StarrocksDbUtil()
    ds = sys.argv[1]
    logger.info("execute task on ds {}".format(ds))
    if not is_trade_date(ds):
        logger.info(f"{ds} is not trade date. task exits.")
        exit(os.EX_OK)

    ods_sql_str = '''
CREATE TABLE if not exists `external_stock_zh_a_spot_em` ( 
 `gmt_create` datetime ,
 `gmt_modified` datetime ,
 `序号` bigint(20) ,
 `代码` varchar(20) ,
 `名称` varchar(20) ,
 `最新价` double ,
 `涨跌幅` double ,
 `涨跌额` double ,
 `成交量` double ,
 `成交额` double ,
 `振幅` double ,
 `最高` double ,
 `最低` double ,
 `今开` double ,
 `昨收` double ,
 `量比` double ,
 `换手率` double ,
 `市盈率-动态` double ,
 `市净率` double ,
 `总市值` double ,
 `流通市值` double ,
 `涨速` double ,
 `5分钟涨跌` double ,
 `60日涨跌幅` double ,
 `年初至今涨跌幅` double ,
 `ds` varchar(20)  
 )ENGINE = mysql 
PROPERTIES
(
"host" = "192.168.50.100",
"port" = "3306",
"user" = "quant",
"password" = "quant",
"database" = "akshare_data",
"table" = "stock_zh_a_spot_em"
);



CREATE TABLE if not exists `ods_stock_zh_a_spot_em` ( 
 `gmt_create` datetime ,
 `gmt_modified` datetime ,
 `序号` bigint(20) ,
 `代码` varchar(20) ,
 `名称` varchar(20) ,
 `最新价` double ,
 `涨跌幅` double ,
 `涨跌额` double ,
 `成交量` double ,
 `成交额` double ,
 `振幅` double ,
 `最高` double ,
 `最低` double ,
 `今开` double ,
 `昨收` double ,
 `量比` double ,
 `换手率` double ,
 `市盈率-动态` double ,
 `市净率` double ,
 `总市值` double ,
 `流通市值` double ,
 `涨速` double ,
 `5分钟涨跌` double ,
 `60日涨跌幅` double ,
 `年初至今涨跌幅` double ,
 `ds` date) 
PARTITION BY RANGE(ds)(
    START ("20230601") END ("{}") EVERY (INTERVAL 1 day)
)
DISTRIBUTED BY HASH(ds) BUCKETS 32
PROPERTIES(
    "replication_num" = "1",
    "dynamic_partition.enable" = "true",
    "dynamic_partition.time_unit" = "DAY",
    "dynamic_partition.start" = "-31",
    "dynamic_partition.end" = "7",
    "dynamic_partition.prefix" = "p",
    "dynamic_partition.buckets" = "32"
)
;

INSERT OVERWRITE ods_stock_zh_a_spot_em PARTITION(p{})
select 
`gmt_create`,
`gmt_modified`,
`序号`,
`代码`,
`名称`,
`最新价`,
`涨跌幅`,
`涨跌额`,
`成交量`,
`成交额`,
`振幅`,
`最高`,
`最低`,
`今开`,
`昨收`,
`量比`,
`换手率`,
`市盈率-动态`,
`市净率`,
`总市值`,
`流通市值`,
`涨速`,
`5分钟涨跌`,
`60日涨跌幅`,
`年初至今涨跌幅`,
`ds`
from 
external_stock_zh_a_spot_em 
where ds = '{}'
;
    '''.format(ds, ds, ds)

    dwd_sql_str = '''
    CREATE TABLE if not exists dwd_stock_zh_a_spot_em_di LIKE ods_stock_zh_a_spot_em;

    INSERT OVERWRITE dwd_stock_zh_a_spot_em_di PARTITION(p{})
    select distinct 
    `gmt_create`,
    `gmt_modified`,
    `序号`,
    `代码`,
    `名称`,
    `最新价`,
    `涨跌幅`,
    `涨跌额`,
    `成交量`,
    `成交额`,
    `振幅`,
    `最高`,
    `最低`,
    `今开`,
    `昨收`,
    `量比`,
    `换手率`,
    `市盈率-动态`,
    `市净率`,
    `总市值`,
    `流通市值`,
    `涨速`,
    `5分钟涨跌`,
    `60日涨跌幅`,
    `年初至今涨跌幅`,
    `ds`
    from 
    ods_stock_zh_a_spot_em 
    where ds = '{}'
    ;
        '''.format(ds, ds)

    db.run_sql(ods_sql_str)
    logger.info("ods sql finished.")
    db.run_sql(dwd_sql_str)
    logger.info("dwd sql finished.")




