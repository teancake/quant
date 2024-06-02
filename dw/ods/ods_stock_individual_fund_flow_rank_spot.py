import sys, os

parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(parent_dir)

from utils.log_util import get_logger
from utils.starrocks_db_util import mysql_to_ods_dwd
from utils.stock_zh_a_util import is_trade_date

import sys

logger = get_logger(__name__)

if __name__ == '__main__':
    ds = sys.argv[1]
    logger.info("execute task on ds {}".format(ds))
    if not is_trade_date(ds):
        logger.info(f"{ds} is not trade date. task exits.")
        exit(os.EX_OK)

    mysql_table_name = "stock_individual_fund_flow_rank_spot"
    mysql_to_ods_dwd(mysql_table_name, ds)





