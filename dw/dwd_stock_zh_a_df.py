import sys, os
parent_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parent_dir)


from utils.log_util import get_logger
from utils.starrocks_db_util import StarrocksDbUtil

logger = get_logger(__name__)


if __name__ == '__main__':
    db = StarrocksDbUtil()
    sql_str = '''
    
create table if not exists dwd_stock_zh_a_df
(
  `gmt_create`   datetime,
  `gmt_modified` datetime,
  `代码`           varchar(20),
  `名称`           varchar(20),
  `ds` date  
)
PARTITION BY RANGE(ds)(
    START ("20230701") END ("20230710") EVERY (INTERVAL 1 day)
)
DISTRIBUTED BY HASH(ds) BUCKETS 32
PROPERTIES(
    "dynamic_partition.enable" = "true",
    "dynamic_partition.time_unit" = "DAY",
    "dynamic_partition.start" = "-365",
    "dynamic_partition.end" = "3",
    "dynamic_partition.prefix" = "p",
    "dynamic_partition.buckets" = "32"
)
;


INSERT OVERWRITE dwd_stock_zh_a_df PARTITION(p20230709)
select 
gmt_create,
gmt_modified,
代码,
名称,
'20230709' as ds from (
SELECT *, row_number() over ( partition by 代码 ORDER by gmt_create desc) as rn FROM external_stock_zh_a
)a 
where rn = 1 ORDER by 代码 asc ;


select * from dwd_stock_zh_a_df Partition(p20230709)
limit 100;

    
    '''
    db.run_sql(sql_str)




