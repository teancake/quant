import sys, os

parent_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parent_dir)

import numpy as np
from datetime import datetime, timedelta

from utils.log_util import get_logger

logger = get_logger(__name__)

from utils.stock_zh_a_util import is_trade_date, is_backfill

from base_data import BaseData
from model.factor_model import FamaFrenchThree


class FF3FactorData(BaseData):
    def __init__(self, ds):
        super().__init__()
        self.ds = ds

    def get_table_name(self):
        return "ff3_factor_data"

    def set_df(self, df):
        self.df = df.copy()

    def get_df_schema(self):
        df = self.df
        df["model"] = "ff3"
        df["period"] = "daily"
        df.reset_index(inplace=True)
        return df

    def before_retrieve_data(self):
        pass


class FF3AlphaBeta(BaseData):
    def __init__(self, ds):
        super().__init__()
        self.ds = ds

    def get_table_name(self):
        return "ff3_alpha_beta"

    def set_df(self, df):
        self.df = df.copy()

    def get_df_schema(self):
        df = self.df
        df["model"] = "ff3"
        df["period"] = "daily"
        df.reset_index(inplace=True)
        return df

    def before_retrieve_data(self):
        pass


class DataHelper:
    def __init__(self, ds, start_date, window=180):
        self.ds = ds
        self.start_date = start_date
        self.window = window
        self.ff3_factors = None
        self.ff3_alpha_beta = None

    def exec(self):
        model = FamaFrenchThree(self.ds, self.start_date)
        factors_df = model.get_ff3_factors()
        factors_data = FF3FactorData(self.ds)
        factors_data.set_df(factors_df)
        factors_data.retrieve_data()

        ols_df = model.get_alpha_beta(days_ahead=self.window)
        ols_df["window"] = self.window
        ols_df.replace([np.inf, -np.inf], np.nan, inplace=True)

        ols_data = FF3AlphaBeta(self.ds)
        ols_data.set_df(ols_df)
        ols_data.retrieve_data()


if __name__ == '__main__':
    ds = sys.argv[1]
    logger.info("execute task on ds {}".format(ds))
    if not is_trade_date(ds):
        logger.info(f"{ds} is not trade date. task exits.")
        exit(os.EX_OK)

    start_date = datetime.strptime(ds, "%Y%m%d") - timedelta(days=365)
    DataHelper(ds=ds, start_date=start_date).exec()
