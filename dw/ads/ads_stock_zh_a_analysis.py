
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
    DROP TABLE IF EXISTS temp_stock_zh_a_analysis_test;
    CREATE TABLE IF NOT EXISTS temp_stock_zh_a_analysis_test AS 
    SELECT b.*,
           a.ret_jbpv,
           a.ret_is_normal
    FROM (
      SELECT ticker,
               jarque_bera_pval AS ret_jbpv,
               IF(jarque_bera_pval < 0.05, 0, 1) AS ret_is_normal
        FROM dwd_stock_zh_a_test_df
        WHERE ds = "{ds}"
        AND variable = "pct_chg"
    ) a
    JOIN (
      SELECT ticker,
             jarque_bera_pval AS logret_jbpv,
             IF(jarque_bera_pval < 0.05, 0, 1) AS logret_is_normal,
             adfpv AS logret_adfpv,
             IF(adfpv < 0.05, 0, 1) AS is_non_stat,
             lb_pvalue AS logret_lbpv,
             IF(lb_pvalue < 0.05, 0, 1) AS is_white_noise
      FROM dwd_stock_zh_a_test_df
      WHERE ds = "{ds}"
      AND variable = "logret"
    ) AS b
    ON a.ticker = b.ticker;
    
    ##--
    DROP TABLE IF EXISTS temp_stock_zh_a_analysis_a;
    CREATE TABLE IF NOT EXISTS temp_stock_zh_a_analysis_a AS 
    SELECT a.ds,
             a.symbol,
             b.简称,
             a.sharpe,
             a.beta,
             a.alpha,
             a.correlation,
             a.volatility_ann,
             a.r_2,
             b.行业,
             c.pe,
             pe_ttm,
             pb,
             total_mv,
             d.alpha as ff3_alpha,
             logret_jbpv,
             logret_is_normal,
             logret_adfpv,
             is_non_stat,
             logret_lbpv,
             is_white_noise,
             ret_jbpv,
             ret_is_normal
    FROM 
    (SELECT *
        FROM dwd_stock_zh_a_stats_df
        WHERE ds = "{ds}"
        )a
    LEFT JOIN 
        (SELECT *,
             代码 AS symbol
        FROM dwd_stock_individual_info_em_df
        WHERE ds = "{ds}"
        )b
        ON a.symbol=b.symbol
    LEFT JOIN 
        (SELECT *, regexp_replace(ts_code,"[^0-9]","") AS symbol
        FROM dwd_tushare_daily_basic_df
        WHERE ds = "{ds}"
        AND trade_date=ds
        )c
        ON a.symbol=c.symbol
    LEFT JOIN (SELECT * FROM dwd_ff3_alpha_beta_df 
        WHERE ds = "{ds}"
        )d 
        ON a.symbol = d.ticker
    LEFT JOIN  temp_stock_zh_a_analysis_test f
        ON a.symbol = f.ticker;

    ##-- 
    DROP TABLE IF EXISTS temp_stock_zh_a_analysis_b; 
    CREATE TABLE IF NOT EXISTS temp_stock_zh_a_analysis_b AS 
    SELECT *,
        (pb - pb_industry)/pb_industry AS dist_from_avg
    FROM 
        (SELECT *,
            avg(pb) OVER (partition by 行业) AS pb_industry, 
            rank()  OVER (partition by 行业 ORDER BY  pb asc) AS pb_rank_industry,
            avg(pe) OVER (partition by 行业) AS pe_industry, 
            rank()  OVER (partition by 行业 ORDER BY  pe_ttm asc) AS pe_rank_industry,
            avg(total_mv) OVER (partition by 行业) AS total_mv_industry, 
            rank()  OVER (partition by 行业 ORDER BY total_mv asc) AS total_mv_rank_industry,
            rank() OVER (order by total_mv) AS total_mv_rank,
            count(*) over () as cnt, 
            count(*) over (partition by 行业) as cnt_industry 
        FROM temp_stock_zh_a_analysis_a
        )a ; 

    ##--
    DROP TABLE IF EXISTS temp_stock_zh_a_analysis_c; 
    CREATE TABLE IF NOT EXISTS temp_stock_zh_a_analysis_c AS 
    SELECT a.ds,
        a.symbol,
        简称,
        sharpe,
        beta,
        alpha,
        correlation,
        volatility_ann,
        r_2,
        行业,
        pe,
        pe_ttm,
        pb,
        total_mv,
        ff3_alpha,
        pb_industry,
        pb_rank_industry,
        pe_industry,
        pe_rank_industry,
        total_mv_industry,
        total_mv_rank_industry,
        total_mv_rank,
        cnt,
        cnt_industry,
        dist_from_avg,
        代码,
        ma20_diff,
        close,
        ma_20,
        ESG等级,
        环境等级,
        社会等级,
        公司治理等级,
        logret_jbpv,
        logret_is_normal,
        logret_adfpv,
        is_non_stat,
        logret_lbpv,
        is_white_noise,
        ret_jbpv,
        ret_is_normal
    FROM temp_stock_zh_a_analysis_b a
    LEFT JOIN 
        (SELECT 代码,
             (close-ma_20)/ma_20 AS ma20_diff,
             close,
             ma_20
        FROM ads_stock_zh_a_pred_data
        WHERE 日期="{ds}" 
        )b
        ON a.symbol=b.代码
    LEFT JOIN 
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
