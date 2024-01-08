import sys, os
parent_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parent_dir)


from utils.log_util import get_logger
from utils.starrocks_db_util import StarrocksDbUtil, generate_partition_spec
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

CREATE TABLE if not exists `external_index_stock_cons_weight_csindex` ( 
 `gmt_create` datetime ,
 `gmt_modified` datetime ,
 `日期` date ,
 `指数代码` varchar,
 `指数名称` varchar,
 `指数英文名称` varchar,
 `成分券代码` varchar,
 `成分券名称` varchar,
 `成分券英文名称` varchar,
 `交易所` varchar,
 `交易所英文名称` varchar,
 `权重` double ,
 `ds` varchar 
 ) 
ENGINE = mysql 
PROPERTIES
(
"host" = "{server_address}",
"port" = "{port}",
"user" = "{user}",
"password" = "{password}",
"database" = "{db_name}",
"table" = "index_stock_cons_weight_csindex"
);

    

CREATE TABLE if not exists `ods_index_stock_cons_weight_csindex` ( 
`gmt_create` datetime ,
 `gmt_modified` datetime ,
 `日期` date ,
 `指数代码` varchar(20),
 `指数名称` varchar(50),
 `指数英文名称` varchar(200),
 `成分券代码` varchar(20),
 `成分券名称` varchar(50),
 `成分券英文名称` varchar(200),
 `交易所` varchar(50),
 `交易所英文名称` varchar(200),
 `权重` double ,
 `ds` date ) 
PARTITION BY RANGE(ds)(
{partition_str}
)
DISTRIBUTED BY HASH(ds) BUCKETS 32
PROPERTIES(
    "replication_num" = "1",
    "dynamic_partition.enable" = "true",
    "dynamic_partition.time_unit" = "DAY",
    "dynamic_partition.start" = "-365",
    "dynamic_partition.end" = "7",
    "dynamic_partition.prefix" = "p",
    "dynamic_partition.buckets" = "32"
)
;

CREATE TABLE if not exists `dwd_index_stock_cons_weight_csindex_df` ( 
`gmt_create` datetime ,
 `gmt_modified` datetime ,
 `日期` date ,
 `指数代码` varchar(20),
 `指数名称` varchar(50),
 `指数英文名称` varchar(200),
 `成分券代码` varchar(20),
 `成分券名称` varchar(50),
 `成分券英文名称` varchar(200),
 `交易所` varchar(50),
 `交易所英文名称` varchar(200),
 `权重` double ,
 `ds` date ) 
PARTITION BY RANGE(ds)(
{partition_str}
)
DISTRIBUTED BY HASH(ds) BUCKETS 32
PROPERTIES(
    "replication_num" = "1",
    "dynamic_partition.enable" = "true",
    "dynamic_partition.time_unit" = "DAY",
    "dynamic_partition.start" = "-365",
    "dynamic_partition.end" = "7",
    "dynamic_partition.prefix" = "p",
    "dynamic_partition.buckets" = "32"
)
;
'''

    ods_sql_str = '''
SET SESSION query_timeout=1200;
INSERT OVERWRITE ods_index_stock_cons_weight_csindex PARTITION(p{})
select
`gmt_create`,
`gmt_modified`,
`日期`,
`指数代码`,
`指数名称`,
`指数英文名称`,
`成分券代码`,
`成分券名称`,
`成分券英文名称`,
`交易所`,
`交易所英文名称`,
`权重`,
`ds`
from 
external_index_stock_cons_weight_csindex 
where ds = '{}'
;
    '''.format(ds, ds)

    dwd_sql_str = '''
SET SESSION query_timeout=600;
INSERT OVERWRITE dwd_index_stock_cons_weight_csindex_df PARTITION(p{})
select 
`gmt_create`,
`gmt_modified`,
`日期`,
`指数代码`,
`指数名称`,
`指数英文名称`,
`成分券代码`,
`成分券名称`,
`成分券英文名称`,
`交易所`,
`交易所英文名称`,
`权重`,
`ds`
from (
select *, row_number()over (partition by 指数代码, 成分券代码 ORDER by gmt_create desc) as rn
FROM ods_index_stock_cons_weight_csindex
where ds = '{}'
)a
where rn = 1
;
        '''.format(ds, ds)

    db.run_sql(ddl_sql_str)
    logger.info("ddl sql finished.")
    db.run_sql(ods_sql_str)
    logger.info("ods sql finished.")
    db.run_sql(dwd_sql_str)
    logger.info("dwd sql finished.")

    db.dqc_row_count("dwd_index_stock_cons_weight_csindex_df", ds)

