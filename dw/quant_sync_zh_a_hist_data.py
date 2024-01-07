
import sys, os
parent_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parent_dir)


import sys

from utils import quant_data_util

from utils.log_util import get_logger

logger = get_logger(__name__)


if __name__ == '__main__':
    ds = sys.argv[1]
    logger.info("execute task on ds {}".format(ds))
    if not is_trade_date(ds):
        logger.info(f"{ds} is not trade date. task exits.")
        exit(os.EX_OK)

    quant_data_util.sync_zh_a_hist_data(ds)
    quant_data_util.clean_up_old_zh_a_hist_data(ds)
    logger.info("data sync task finished.")