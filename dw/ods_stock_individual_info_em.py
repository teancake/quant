import sys, os
parent_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parent_dir)


from utils.log_util import get_logger
from utils.starrocks_db_util import StarrocksDbUtil, generate_partition_spec, mysql_to_ods_dwd
import sys
logger = get_logger(__name__)
from utils.stock_zh_a_util import is_trade_date

from utils.db_util import get_mysql_config


if __name__ == '__main__':
    db = StarrocksDbUtil()
    ds = sys.argv[1]
    logger.info("execute task on ds {}".format(ds))
    if not is_trade_date(ds):
        logger.info(f"{ds} is not trade date. task exits.")
        exit(os.EX_OK)
    mysql_table_name = "stock_individual_info_em"
    mysql_to_ods_dwd(mysql_table_name, ds, di_df="df", unique_columns=["代码"],
                     rename_columns={"股票代码": "代码", "股票简称": "简称"})
