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
    # 天数据
    mysql_table_name = "tushare_daily"
    mysql_to_ods_dwd(mysql_table_name, ds, di_df="df", unique_columns=["ts_code", "trade_date"], lifecycle=31)

    mysql_table_name = "tushare_daily_basic"
    mysql_to_ods_dwd(mysql_table_name, ds, di_df="df", unique_columns=["ts_code", "trade_date"], lifecycle=31)

    mysql_table_name = "tushare_stk_factor"
    mysql_to_ods_dwd(mysql_table_name, ds, di_df="df", unique_columns=["ts_code", "trade_date"], lifecycle=31, days_ahead=15)

    mysql_table_name = "tushare_index_daily"
    mysql_to_ods_dwd(mysql_table_name, ds, di_df="df", unique_columns=["ts_code", "trade_date"], lifecycle=31)

    mysql_table_name = "tushare_stock_basic"
    mysql_to_ods_dwd(mysql_table_name, ds, di_df="df", unique_columns=["symbol"], lifecycle=31)

    mysql_table_name = "tushare_trade_cal"
    mysql_to_ods_dwd(mysql_table_name, ds, di_df="df", unique_columns=["tradedays"], lifecycle=31)

    mysql_table_name = "tushare_adj_factor"
    mysql_to_ods_dwd(mysql_table_name, ds, di_df="df", unique_columns=["ts_code", "trade_date"], lifecycle=31)


    # 季度数据
    mysql_table_name = "tushare_fina_indicator"
    mysql_to_ods_dwd(mysql_table_name, ds, di_df="df", unique_columns=["ts_code", "end_date"], lifecycle=31, days_ahead=0, mysql_where_cond="1 = 1")

    mysql_table_name = "tushare_cashflow"
    mysql_to_ods_dwd(mysql_table_name, ds, di_df="df", unique_columns=["ts_code", "end_date"], lifecycle=31, days_ahead=1, mysql_where_cond="1 = 1")

    mysql_table_name = "tushare_balancesheet"
    mysql_to_ods_dwd(mysql_table_name, ds, di_df="df", unique_columns=["ts_code", "end_date"], lifecycle=31, days_ahead=1, mysql_where_cond="1 = 1")

    mysql_table_name = "tushare_income"
    mysql_to_ods_dwd(mysql_table_name, ds, di_df="df", unique_columns=["ts_code", "end_date"], lifecycle=31, days_ahead=1, mysql_where_cond="1 = 1")