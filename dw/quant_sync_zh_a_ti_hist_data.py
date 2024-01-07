import sys, os
parent_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parent_dir)


import sys

from datax_processor import DataxProcessor
from utils.log_util import get_logger

logger = get_logger(__name__)


if __name__ == '__main__':
    ds = sys.argv[1]
    logger.info("execute task on ds {}".format(ds))
    if not is_trade_date(ds):
        logger.info(f"{ds} is not trade date. task exits.")
        exit(os.EX_OK)

    columns = ["代码", "period", "adjust", "日期", "开盘", "收盘", "最高", "最低", "成交量", "成交额", "振幅", "涨跌幅", "涨跌额", "换手率"]
    table_name = "dwd_stock_zh_a_hist_df"
    where = f"ds='{ds}' and period='daily' and adjust='hfq'"
    data_file_name = "zha_a_hist_data.csv"

    processor = DataxProcessor(columns=columns, table=table_name, where=where, data_file_name=data_file_name)
    processor.process()
