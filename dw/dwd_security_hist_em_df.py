import sys, os
parent_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parent_dir)


from utils.log_util import get_logger
from utils.starrocks_db_util import StarrocksDbUtil, generate_partition_spec
from utils.stock_zh_a_util import is_trade_date
import sys

logger = get_logger(__name__)


if __name__ == '__main__':
    ds = sys.argv[1]
    logger.info("execute task on ds {}".format(ds))

    if not is_trade_date(ds):
        logger.info(f"{ds} is not trade date. task exits.")
        exit(os.EX_OK)

    db = StarrocksDbUtil()
    partition_str = generate_partition_spec(ds)

    ddl_sql_str = f'''        
    CREATE TABLE IF NOT EXISTS `dwd_security_hist_em_df` ( 
      `gmt_create` datetime NULL COMMENT "",
      `gmt_modified` datetime NULL COMMENT "",
      `代码` varchar(255) NULL COMMENT "",
      `period` varchar(255) NULL COMMENT "",
      `adjust` varchar(255) NULL COMMENT "",
      `日期` varchar(255) NULL COMMENT "",
      `开盘` double NULL COMMENT "",
      `收盘` double NULL COMMENT "",
      `最高` double NULL COMMENT "",
      `最低` double NULL COMMENT "",
      `成交量` bigint(20) NULL COMMENT "",
      `成交额` double NULL COMMENT "",
      `振幅` double NULL COMMENT "",
      `涨跌幅` double NULL COMMENT "",
      `涨跌额` double NULL COMMENT "",
      `换手率` double NULL COMMENT "",
      `type` varchar(255) NULL COMMENT "",
      `ds` date NULL COMMENT ""
     ) ENGINE=OLAP DUPLICATE KEY(`gmt_create`,
     `gmt_modified`,
     `代码`, `period`, `adjust`) COMMENT "OLAP" 
     PARTITION BY RANGE(ds)({partition_str})
     DISTRIBUTED BY HASH(ds) BUCKETS 32
     PROPERTIES(
        "replication_num" = "1",
        "dynamic_partition.enable" = "true",
        "dynamic_partition.time_unit" = "DAY",
        "dynamic_partition.start" = "-60",
        "dynamic_partition.end" = "7",
        "dynamic_partition.prefix" = "p",
        "dynamic_partition.buckets" = "32"
    )
    '''
    dwd_sql_str = f'''
    INSERT OVERWRITE dwd_security_hist_em_df PARTITION(p{ds})
    SELECT 
        gmt_create,
        gmt_modified,
        代码,
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
        'stock_zh_a' as type,
        ds
    FROM dwd_stock_zh_a_hist_df 
    WHERE ds = {ds}
    UNION ALL 
    SELECT 
        gmt_create,
        gmt_modified,
        代码,
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
        'fund_etf' as type,
        ds
    FROM dwd_fund_etf_hist_em_df
    WHERE ds = {ds}
    UNION ALL 
    SELECT 
        gmt_create,
        gmt_modified,
        代码,
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
        'fund_lof' as type,
        ds
    FROM dwd_fund_lof_hist_em_df
    WHERE ds = {ds}
    '''
    db.run_sql(ddl_sql_str)
    logger.info("ddl sql finished.")
    db.run_sql(dwd_sql_str)
    logger.info("dwd sql finished.")

    db.dqc_row_count("dwd_security_hist_em_df", ds)


