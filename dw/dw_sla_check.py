import sys, os

parent_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parent_dir)

from utils.log_util import get_logger
from utils.starrocks_db_util import StarrocksDbUtil
import sys
import pandas as pd
from datetime import datetime

logger = get_logger(__name__)
import pytz


def split_date(iso_date):
    tz = pytz.timezone('Asia/Shanghai')
    dt = datetime.fromisoformat(iso_date).astimezone(tz)
    print(f"dt {dt}")
    ds = dt.date().strftime("%Y%m%d")
    ts = dt.time().strftime("%H:%M")
    print(f"ds {ds}, ts {ts}")
    return ds, ts


def dqc_row_count_with_sla(table_name, iso_date, deadline):
    ds, ts = split_date(iso_date)
    dqc_sql = "select count(*) from {} where ds = '{}'".format(table_name, ds)
    row_count = StarrocksDbUtil().run_sql(dqc_sql)[0][0]
    logger.info(f"table {table_name}, ds {ds}, ts {ts}, deadline {deadline}, row count {row_count}.")
    if row_count == 0 and ts > deadline:
        raise Exception(f"SLA check failed: table {table_name}")


if __name__ == '__main__':
    logical_date = sys.argv[1]
    print(f"logical_date {logical_date}")
    split_date(logical_date)
    confs = [("ads_stock_zh_a_training_data", "20:00")]
    for conf in confs:
        table_name, deadline = conf
        dqc_row_count_with_sla(table_name, logical_date, deadline)
