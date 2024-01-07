import sys, os
parent_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parent_dir)


from utils.log_util import get_logger
from utils.starrocks_db_util import StarrocksDbUtil
from utils.stock_zh_a_util import is_trade_date
logger = get_logger(__name__)
import sys


if __name__ == '__main__':
    db = StarrocksDbUtil()
    ds = sys.argv[1]
    logger.info("execute task on ds {}".format(ds))
    if not is_trade_date(ds):
        logger.info(f"{ds} is not trade date. task exits.")
        exit(os.EX_OK)


    create_table_sql_str = '''
CREATE TABLE if not exists `dwd_stock_zh_a_feature_df` (
  `代码` varchar(20) NULL COMMENT "",
  `日期` date NULL COMMENT "",
    `简称` varchar(50) ,
  `行业` varchar(50) ,
  `上市时间` varchar(50) ,
  `总股本` varchar(50) ,
  `流通股` varchar(50) ,
  `总市值` varchar(50) ,
  `流通市值` varchar(50) ,
  `开盘` double NULL COMMENT "",
  `收盘` double NULL COMMENT "",
  `涨跌幅` double NULL COMMENT "",
  `开盘_max_3d` double NULL COMMENT "",
  `开盘_min_3d` double NULL COMMENT "",
  `开盘_avg_3d` double NULL COMMENT "",
  `收盘_max_3d` double NULL COMMENT "",
  `收盘_min_3d` double NULL COMMENT "",
  `收盘_avg_3d` double NULL COMMENT "",
  `涨跌幅_max_3d` double NULL COMMENT "",
  `涨跌幅_min_3d` double NULL COMMENT "",
  `涨跌幅_avg_3d` double NULL COMMENT "",
  `开盘_max_7d` double NULL COMMENT "",
  `开盘_min_7d` double NULL COMMENT "",
  `开盘_avg_7d` double NULL COMMENT "",
  `收盘_max_7d` double NULL COMMENT "",
  `收盘_min_7d` double NULL COMMENT "",
  `收盘_avg_7d` double NULL COMMENT "",
  `涨跌幅_max_7d` double NULL COMMENT "",
  `涨跌幅_min_7d` double NULL COMMENT "",
  `涨跌幅_avg_7d` double NULL COMMENT "",
  `开盘_max_30d` double NULL COMMENT "",
  `开盘_min_30d` double NULL COMMENT "",
  `开盘_avg_30d` double NULL COMMENT "",
  `收盘_max_30d` double NULL COMMENT "",
  `收盘_min_30d` double NULL COMMENT "",
  `收盘_avg_30d` double NULL COMMENT "",
  `涨跌幅_max_30d` double NULL COMMENT "",
  `涨跌幅_min_30d` double NULL COMMENT "",
  `涨跌幅_avg_30d` double NULL COMMENT "",
  `开盘_max_180d` double NULL COMMENT "",
  `开盘_min_180d` double NULL COMMENT "",
  `开盘_avg_180d` double NULL COMMENT "",
  `收盘_max_180d` double NULL COMMENT "",
  `收盘_min_180d` double NULL COMMENT "",
  `收盘_avg_180d` double NULL COMMENT "",
  `涨跌幅_max_180d` double NULL COMMENT "",
  `涨跌幅_min_180d` double NULL COMMENT "",
  `涨跌幅_avg_180d` double NULL COMMENT "",
  `换手率` double NULL COMMENT "",
`换手率_max_3d` double NULL COMMENT "",
`换手率_min_3d` double NULL COMMENT "",
`换手率_avg_3d` double NULL COMMENT "", 
`换手率_max_7d` double NULL COMMENT "",
`换手率_min_7d` double NULL COMMENT "",
`换手率_avg_7d` double NULL COMMENT "",
`换手率_max_30d` double NULL COMMENT "",
`换手率_min_30d` double NULL COMMENT "",
`换手率_avg_30d` double NULL COMMENT "", 
`换手率_max_180d` double NULL COMMENT "",
`换手率_min_180d` double NULL COMMENT "",
`换手率_avg_180d` double NULL COMMENT "",
  `ds` date 
)
PARTITION BY RANGE(ds)(
    START ("20230701") END ("20230720") EVERY (INTERVAL 1 day)
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
    temp_tables_sql_str = '''
drop view if exists dwd_stock_zh_a_hist_df_maxpt;
create view if not exists dwd_stock_zh_a_hist_df_maxpt as 
select * from dwd_stock_zh_a_hist_df where ds in (select max(ds) from dwd_stock_zh_a_hist_df)
and adjust = "hfq" and period = "daily";



drop table if exists temp_stock_feature_1d;
create table if not exists temp_stock_feature_1d as
select 代码, 日期, 开盘, 收盘, 涨跌幅, 换手率
from (
  select *, row_number()over (partition by 代码
  order by 日期 desc) as rn
  from dwd_stock_zh_a_hist_df_maxpt
) a
where rn = 1
;


drop table if exists temp_stock_feature_3d;
create table if not exists temp_stock_feature_3d as
select 代码, max(开盘) as 开盘_max_3d, min(开盘) as 开盘_min_3d, avg(开盘) as 开盘_avg_3d, max(收盘) as 收盘_max_3d,
       min(收盘) as 收盘_min_3d, avg(收盘) as 收盘_avg_3d, max(涨跌幅) as 涨跌幅_max_3d, min(涨跌幅) as 涨跌幅_min_3d,
       avg(涨跌幅) as 涨跌幅_avg_3d, max(换手率) as 换手率_max_3d, min(换手率) as 换手率_min_3d,
       avg(换手率) as 换手率_avg_3d
from (
  select *, row_number()over (partition by 代码
  order by 日期 desc) as rn
  from dwd_stock_zh_a_hist_df_maxpt
) a
where rn <= 3
group by 代码;




drop table if exists temp_stock_feature_7d;
create table if not exists temp_stock_feature_7d as
select 代码, max(开盘) as 开盘_max_7d, min(开盘) as 开盘_min_7d, avg(开盘) as 开盘_avg_7d, max(收盘) as 收盘_max_7d,
       min(收盘) as 收盘_min_7d, avg(收盘) as 收盘_avg_7d, max(涨跌幅) as 涨跌幅_max_7d, min(涨跌幅) as 涨跌幅_min_7d,
       avg(涨跌幅) as 涨跌幅_avg_7d, max(换手率) as 换手率_max_7d, min(换手率) as 换手率_min_7d,
       avg(换手率) as 换手率_avg_7d
from (
  select *, row_number()over (partition by 代码
  order by 日期 desc) as rn
  from dwd_stock_zh_a_hist_df_maxpt
) a
where rn <= 7
group by 代码;





drop table if exists temp_stock_feature_30d;
create table if not exists temp_stock_feature_30d as
select 代码, max(开盘) as 开盘_max_30d, min(开盘) as 开盘_min_30d, avg(开盘) as 开盘_avg_30d, max(收盘) as 收盘_max_30d,
       min(收盘) as 收盘_min_30d, avg(收盘) as 收盘_avg_30d, max(涨跌幅) as 涨跌幅_max_30d, min(涨跌幅) as 涨跌幅_min_30d,
       avg(涨跌幅) as 涨跌幅_avg_30d, max(换手率) as 换手率_max_30d, min(换手率) as 换手率_min_30d,
       avg(换手率) as 换手率_avg_30d
from (
  select *, row_number()over (partition by 代码
  order by 日期 desc) as rn
  from dwd_stock_zh_a_hist_df_maxpt
) a
where rn <= 30
group by 代码;




drop table if exists temp_stock_feature_180d;
create table if not exists temp_stock_feature_180d as
select 代码, max(开盘) as 开盘_max_180d, min(开盘) as 开盘_min_180d, avg(开盘) as 开盘_avg_180d, max(收盘) as 收盘_max_180d,
       min(收盘) as 收盘_min_180d, avg(收盘) as 收盘_avg_180d, max(涨跌幅) as 涨跌幅_max_180d, min(涨跌幅) as 涨跌幅_min_180d,
       avg(涨跌幅) as 涨跌幅_avg_180d, max(换手率) as 换手率_max_180d, min(换手率) as 换手率_min_180d,
       avg(换手率) as 换手率_avg_180d
from (
  select *, row_number()over (partition by 代码
  order by 日期 desc) as rn
  from dwd_stock_zh_a_hist_df_maxpt
) a
where rn <= 180
group by 代码;


drop table if exists temp_stock_individual_info_em;
create table if not exists temp_stock_individual_info_em as 
select * from dwd_stock_individual_info_em_df where ds in (select max(ds) from dwd_stock_individual_info_em_df);

'''

    insert_sql_str = '''
insert overwrite dwd_stock_zh_a_feature_df partition(p{})
select 
t1.代码,
日期,
`简称`,
`行业`,
`上市时间`,
`总股本`,
`流通股`,
`总市值`,
`流通市值`,
开盘,
收盘,
涨跌幅, 
开盘_max_3d,
开盘_min_3d,
开盘_avg_3d,
收盘_max_3d,
收盘_min_3d,
收盘_avg_3d,
涨跌幅_max_3d,
涨跌幅_min_3d,
涨跌幅_avg_3d,
开盘_max_7d,
开盘_min_7d,
开盘_avg_7d,
收盘_max_7d,
收盘_min_7d,
收盘_avg_7d,
涨跌幅_max_7d,
涨跌幅_min_7d,
涨跌幅_avg_7d,
开盘_max_30d,
开盘_min_30d,
开盘_avg_30d,
收盘_max_30d,
收盘_min_30d,
收盘_avg_30d,
涨跌幅_max_30d,
涨跌幅_min_30d,
涨跌幅_avg_30d,
开盘_max_180d,
开盘_min_180d,
开盘_avg_180d,
收盘_max_180d,
收盘_min_180d,
收盘_avg_180d,
涨跌幅_max_180d,
涨跌幅_min_180d,
涨跌幅_avg_180d,
`换手率`,
`换手率_max_3d`,
`换手率_min_3d`,
`换手率_avg_3d`,
`换手率_max_7d`,
`换手率_min_7d`,
`换手率_avg_7d`,
`换手率_max_30d`,
`换手率_min_30d`,
`换手率_avg_30d`,
`换手率_max_180d`,
`换手率_min_180d`,
`换手率_avg_180d`,
'{}' as ds
from 
temp_stock_feature_1d  t1
join 
temp_stock_feature_3d  t3
on t1.代码=t3.代码
join 
temp_stock_feature_7d  t7
on t1.代码=t7.代码
join 
temp_stock_feature_30d  t30
on t1.代码=t30.代码
join 
temp_stock_feature_180d  t180
on t1.代码=t180.代码
join 
temp_stock_individual_info_em info
on t1.代码=info.代码
;
    '''.format(ds, ds)
    db.run_sql(create_table_sql_str)
    logger.info("create table sql finished.")
    db.run_sql(temp_tables_sql_str)
    logger.info("temp table sql finished.")
    db.run_sql(insert_sql_str)
    logger.info("insert partition sql finished.")

