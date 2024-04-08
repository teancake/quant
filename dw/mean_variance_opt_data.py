import sys, os
parent_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parent_dir)



from datetime import datetime, timedelta

from utils.log_util import get_logger
logger = get_logger(__name__)

from utils.stock_zh_a_util import is_trade_date,  is_backfill

from base_data import BaseData
from model.mean_variance_opt import MeanVarianceOptimizer


class MeanVarianceOptData(BaseData):
    def __init__(self, ds, start_date):
        super().__init__()
        self.ds = ds
        self.start_date = start_date

    def get_table_name(self):
        return "mean_variance_opt_data"

    def get_df_schema(self):
        model = MeanVarianceOptimizer(self.ds, self.start_date)
        df = model.opt()
        df.reset_index(inplace=True, drop=True)
        return df

    def before_retrieve_data(self):
        pass


if __name__ == '__main__':
    ds = sys.argv[1]
    logger.info("execute task on ds {}".format(ds))
    if not is_trade_date(ds):
        logger.info(f"{ds} is not trade date. task exits.")
        exit(os.EX_OK)
    start_date = datetime.strptime(ds, '%Y%m%d') - timedelta(days=500)
    data = MeanVarianceOptData(ds, start_date)
    data.retrieve_data()

