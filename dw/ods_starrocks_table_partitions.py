import sys, os
parent_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parent_dir)


from utils.log_util import get_logger
from utils.starrocks_db_util import StarrocksDbUtil, generate_partition_spec
from utils.config_util import get_starrocks_config
import sys
import pandas as pd
from datetime import datetime

logger = get_logger(__name__)

if __name__ == '__main__':
    db = StarrocksDbUtil()
    ds = sys.argv[1]
    logger.info("execute task on ds {}".format(ds))

    partition_str = generate_partition_spec(ds)
    _, _, db_name, _, _ = get_starrocks_config()

    ods_ddl = '''
CREATE TABLE if not exists `ods_starrocks_table_partitions` ( 
`gmt_create` datetime,
`gmt_modified` datetime,
 `table_name` varchar(200) ,
 `partition_name` varchar(200) ,
 `row_count` bigint(20),
 `ds` date) 
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
); 
    '''.format(partition_str)
    logger.info("run ods ddl")
    db.run_sql(ods_ddl)

    table_sql = f"show tables from {db_name};"
    result = db.run_sql(table_sql)
    print(result)
    data_df_list = []
    for item in result:
        table_name = item[0]
        if table_name.startswith("external_"):
            continue
        partition_sql = "select DATE_FORMAT(ds, '%Y%m%d') as partition_name, count(*) as row_count from {} group by ds order by partition_name asc".format(table_name)
        try:
            data = db.run_sql(partition_sql)
        except Exception as e:
            data = []
            logger.error(e)
        data_df = pd.DataFrame(data)
        data_df["table_name"] = table_name
        data_df_list.append(data_df)
        print(data)
    pt_df = pd.concat(data_df_list, ignore_index=True, sort=False)
    print(pt_df.shape)
    del data_df_list
    pt_df["ds"] = ds
    pt_df["gmt_create"] = datetime.now()
    pt_df["gmt_modified"] = datetime.now()
    pt_df.to_sql(name="ods_starrocks_table_partitions", con=db.get_db_engine(), if_exists='append', index=False, method='multi', chunksize=1000)
    logger.info("partitions info saved to ods tables.")

    dwd_sql = '''
    CREATE TABLE if not exists dwd_starrocks_table_partitions_df LIKE ods_starrocks_table_partitions;
INSERT OVERWRITE dwd_starrocks_table_partitions_df PARTITION(p{})
select 
`gmt_create` ,
`gmt_modified` ,
 `table_name`,
 `partition_name` ,
 `row_count`,
'{}' as `ds`
from (
  select *, row_number()over (partition by table_name, partition_name order by gmt_create desc) as rn
  from ods_starrocks_table_partitions where ds in (select max(ds) from ods_starrocks_table_partitions)
) a
where rn = 1;
    '''.format(ds, ds)

    db.run_sql(dwd_sql)
    logger.info("dwd sql finished.")

    db.dqc_row_count("dwd_starrocks_table_partitions_df", ds)








