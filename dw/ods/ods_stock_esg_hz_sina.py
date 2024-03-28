import sys, os

parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(parent_dir)

from utils.log_util import get_logger
from utils.starrocks_db_util import mysql_to_ods_dwd, content_to_rag
from utils.stock_zh_a_util import is_trade_date

import sys

logger = get_logger(__name__)

if __name__ == '__main__':
    ds = sys.argv[1]
    logger.info("execute task on ds {}".format(ds))

    if not is_trade_date(ds):
        logger.info(f"{ds} is not trade date. task exits.")
        exit(os.EX_OK)

    mysql_table_name = "stock_esg_hz_sina"
    dwd_table_name = mysql_to_ods_dwd(mysql_table_name, ds, di_df="df", unique_columns=["symbol"])
