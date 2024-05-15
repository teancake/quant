import sys, os


parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(parent_dir)

import sys

from utils.log_util import get_logger
from utils.starrocks_db_util import StarrocksDbUtil, generate_partition_spec


logger = get_logger(__name__)


def create_table_and_insert(table_name, ds, lifecycle=365):
    select_clause = get_select_clause(ds)
    if not StarrocksDbUtil().table_exists(table_name):
        logger.info(f"create table {table_name}")
        temp_sql = f"""
        CREATE TABLE {table_name}
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
        AS
        """
    else:
        logger.info(f"table {table_name} exists, insert data into a new partition")
        temp_sql = f"""
        INSERT OVERWRITE {table_name} PARTITION(p{ds})
        """
    temp_sql += select_clause
    StarrocksDbUtil().run_sql(temp_sql)
    StarrocksDbUtil().dqc_row_count(table_name, ds)
    logger.info(f"table {table_name} created or updated from selection.")


def get_select_clause(ds):
    sql = f"""
    select
        a.symbol,
        a.sentiment,
        a.post_count,
        a.click_count,
        a.user_count,
        a.per_post_click_count,
        b.post_count as total_post_count,
        b.click_count as total_click_count,
        b.user_count as total_user_count,
        b.per_post_click_count as total_per_post_click_count,
        a.post_count/b.post_count as post_ratio,
        a.click_count/b.click_count as click_ratio,
        2/(1/(a.post_count/b.post_count)+1/(a.click_count/b.click_count)) as avg_ratio,
        a.ds 
    from
        (select
            ds,
            symbol,
            sentiment,
            count(*) as post_count,
            sum(post_click_count) as click_count,
            count(DISTINCT user_id) as user_count,
            sum(post_click_count)/count(*) as per_post_click_count 
        from
        (
            select *, row_number() over (partition by symbol, post_id order by gmt_create desc) as rn
            from dwd_guba_em_di 
            where cast(post_last_time as date) = ds
            and ds = '{ds}'
        ) aa where rn=1
        group by
            ds,
            symbol,
            sentiment )a  
    join
        (
            select
                ds,
                symbol,
                count(*) as post_count,
                sum(post_click_count) as click_count,
                count(DISTINCT user_id) as user_count,
                sum(post_click_count)/count(*) as per_post_click_count 
            from
            (
                select *, row_number() over (partition by symbol, post_id order by gmt_create desc) as rn
                from dwd_guba_em_di  
                where cast(post_last_time as date) = ds
                and ds = '{ds}'
            ) aa where rn=1
            group by
                ds,
                symbol
        )b 
            on a.symbol = b.symbol
    """
    return sql




if __name__ == "__main__":
    ds = sys.argv[1]
    logger.info("execute task on ds {}".format(ds))
    create_table_and_insert(table_name="ads_guba_sentiment_di", ds=ds)

