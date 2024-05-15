import sys, os

parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(parent_dir)

from utils.log_util import get_logger
from utils.starrocks_db_util import mysql_to_ods_dwd, content_to_rag

import sys
logger = get_logger(__name__)



if __name__ == '__main__':
    ds = sys.argv[1]
    logger.info("execute task on ds {}".format(ds))
    mysql_table_name = "guba_em"
    dwd_table_name = mysql_to_ods_dwd(mysql_table_name, ds, di_df="di", unique_columns=["symbol", "post_id"], lifecycle=3650)
