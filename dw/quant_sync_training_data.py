import sys, os
parent_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parent_dir)


from utils import quant_data_util
from utils.starrocks_db_util import StarrocksDbUtil, generate_partition_spec
from utils.stock_zh_a_util import is_trade_date

from utils.log_util import get_logger
logger = get_logger(__name__)
import sys


if __name__ == '__main__':
    ds = sys.argv[1]
    logger.info("execute task on ds {}".format(ds))
    if not is_trade_date(ds):
        logger.info(f"{ds} is not trade date. task exits.")
        exit(os.EX_OK)

    quant_data_util.sync_training_data(ds)
    quant_data_util.clean_up_old_training_data(ds)

    ## ods table for predictions
    logger.info("in case prediction table is not created. run ods ddl")
    partition_str = generate_partition_spec(ds)
    db = StarrocksDbUtil()
    ods_sql = '''
create table if not exists ods_stock_zh_a_prediction
(
`ds` date,
`gmt_create` datetime,
`gmt_modified` datetime,
`stage` varchar(200),
`model` varchar(200),
`run_id` varchar(200) comment 'run_id is to identify each run of the model',
`日期` date,
`代码` varchar(20),
`pred_name` varchar(200),
`pred_value` double,
`label` double
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
    '''.format(partition_str)
    db.run_sql(ods_sql)
    logger.info("ods sql finished.")
