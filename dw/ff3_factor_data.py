import sys, os
parent_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parent_dir)



from datetime import datetime, timedelta

from utils.log_util import get_logger
logger = get_logger(__name__)

from utils.stock_zh_a_util import is_trade_date,  is_backfill

from base_data import BaseData
from model.factor_model import FamaFrenchThree


class FF3FactorData(BaseData):
    def __init__(self, ds, start_date):
        super().__init__()
        self.ds = ds
        self.start_date = start_date

    def get_table_name(self):
        return "ff3_factor_data"

    def get_df_schema(self):
        model = FamaFrenchThree(self.ds, self.start_date)
        df, rf = model.get_df_rf()
        factors = model.get_ff3_factors(df)
        factors = factors.rename(columns={"MKT_G1/BM_G1": "SL", "MKT_G1/BM_G2": "SM", "MKT_G1/BM_G3": "SH",
                           "MKT_G2/BM_G1": "BL", "MKT_G2/BM_G2": "BM", "MKT_G2/BM_G3": "BH"})
        factors["model"] = "ff3"
        factors["period"] = "daily"
        factors.reset_index(inplace=True)
        return factors

    def before_retrieve_data(self):
        pass


if __name__ == '__main__':
    ds = sys.argv[1]
    logger.info("execute task on ds {}".format(ds))
    if not is_trade_date(ds):
        logger.info(f"{ds} is not trade date. task exits.")
        exit(os.EX_OK)

    backfill = is_backfill(ds)
    logger.info(f"ds {ds}, backfill {backfill}")

    if backfill:
        start_date = datetime.strptime(ds, '%Y%m%d') - timedelta(days=500)
    else:
        start_date = datetime.strptime(ds, '%Y%m%d') - timedelta(days=15)

    data = FF3FactorData(ds, start_date)
    data.retrieve_data()

