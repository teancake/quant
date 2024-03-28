
import sys, os
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(parent_dir)


from utils.log_util import get_logger
from utils.starrocks_db_util import StarrocksDbUtil, generate_partition_spec
from utils.stock_zh_a_util import is_trade_date
import sys

logger = get_logger(__name__)

def run_task(ds):
    temp_sql = f"""
    ##--
    DROP TABLE IF EXISTS temp_stock_zh_a_analysis_a;
    CREATE TABLE IF NOT EXISTS temp_stock_zh_a_analysis_a AS 
    SELECT a.ds,
             a.symbol,
             b.简称,
             a.sharpe,
             a.beta,
             b.行业,
             c.pe,
             pe_ttm,
             pb,
             total_mv
    FROM 
    (SELECT *
        FROM dwd_stock_zh_a_stats_df
        WHERE ds = "{ds}"
        )a
    JOIN 
        (SELECT *,
             代码 AS symbol
        FROM dwd_stock_individual_info_em_df
        WHERE ds = "{ds}"
        )b
        ON a.symbol=b.symbol
    JOIN 
        (SELECT *, regexp_replace(ts_code,"[^0-9]","") AS symbol
        FROM dwd_tushare_daily_basic_df
        WHERE ds = "{ds}"
        AND trade_date=ds
        )c
        ON a.symbol=c.symbol ; 

    ##-- 
    DROP TABLE IF EXISTS temp_stock_zh_a_analysis_b; 
    CREATE TABLE IF NOT EXISTS temp_stock_zh_a_analysis_b AS 
    SELECT *,
        (pb - pb_industry)/pb_industry AS dist_from_avg
    FROM 
        (SELECT *,
            avg(pb) OVER (partition by 行业) AS pb_industry, 
            rank()  OVER (partition by 行业 ORDER BY  pb asc) AS pb_rank_industry
        FROM temp_stock_zh_a_analysis_a
        )a ; 

    ##--
    DROP TABLE IF EXISTS temp_stock_zh_a_analysis_c; 
    CREATE TABLE IF NOT EXISTS temp_stock_zh_a_analysis_c AS 
    SELECT a.*,
             b.*,
             ESG等级,
             环境等级,
             社会等级,
             公司治理等级
    FROM temp_stock_zh_a_analysis_b a
    JOIN 
        (SELECT 代码,
             close-ma_20 AS ma20_diff,
             close,
             ma_20
        FROM ads_stock_zh_a_pred_data
        WHERE 日期="{ds}" 
        )b
        ON a.symbol=b.代码
    JOIN 
        (SELECT symbol,
             ESG等级,
             环境等级,
             社会等级,
             公司治理等级
        FROM dwd_stock_esg_hz_sina_df
        WHERE ds="{ds}" 
        )c
        ON a.symbol=c.symbol
    """

    db = StarrocksDbUtil()
    db.run_sql(temp_sql)

    table_name = "ads_stock_zh_a_analysis"
    if not db.table_exists(table_name):
        logger.info(f"create table {table_name}")
        temp_sql = f"""
        CREATE TABLE {table_name}
        PARTITION BY RANGE(ds)({generate_partition_spec(ds)})
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
        AS
        """
    else:
        logger.info(f"table {table_name} exists, insert data into a new partition")
        temp_sql = f"""
        INSERT OVERWRITE {table_name} PARTITION(p{ds})
        """
    temp_sql += "SELECT * FROM temp_stock_zh_a_analysis_c"

    db.run_sql(temp_sql)
    db.dqc_row_count(table_name, ds)


if __name__ == '__main__':
    ds = sys.argv[1]
    logger.info("execute task on ds {}".format(ds))
    if not is_trade_date(ds):
        logger.info(f"{ds} is not trade date. task exits.")
        exit(os.EX_OK)

    run_task(ds)
