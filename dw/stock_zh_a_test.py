import sys, os
parent_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parent_dir)

import numpy as np
import pandas as pd


from datetime import datetime, timedelta
from statsmodels.stats.stattools import jarque_bera, durbin_watson
from statsmodels.stats.weightstats import ztest
from statsmodels.stats.descriptivestats import describe
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.stats.diagnostic import acorr_ljungbox


from utils.stock_zh_a_util import get_stock_data, is_trade_date
from dw.base_data import BaseData

from utils.log_util import get_logger
logger = get_logger(__name__)

def get_return(ds, days_ahead=365):
    start_date = datetime.strptime(ds, "%Y%m%d") - timedelta(days=days_ahead)
    df = get_stock_data("all", ds=ds, start_date=start_date, yf_compatible=True)
    levels = [df.index, df.symbol]
    df.index = pd.MultiIndex.from_arrays(levels, names=["date", "ticker"])
    df = df.sort_index(level=["date", "ticker"], ascending=[True, True])
    df["pct_chg"] = df["close"].groupby(level="ticker").pct_change().dropna()
    df["logret"] = np.log(df["close"] / df["close"].groupby(level="ticker").shift(1))
    df.dropna(inplace=True)
    df = df[["close", "pct_chg", "logret"]]
    df = df.unstack(level="ticker")
    df.fillna(0, inplace=True)
    logger.info(f"data df {df}")
    return df


def white_noise_test(series):
    ljdf = acorr_ljungbox(series, boxpierce=True, auto_lag=True)
    ljdf_minp = ljdf[ljdf["bp_pvalue"] == ljdf["bp_pvalue"].min()]
    lb_minp_lag = ljdf_minp.index[0]
    lb_stat = ljdf_minp.iloc[0, 0]
    lb_pvalue = ljdf_minp.iloc[0, 1]
    bp_stat = ljdf_minp.iloc[0, 2]
    bp_pvalue = ljdf_minp.iloc[0, 3]
    return [lb_minp_lag, lb_stat, lb_pvalue, bp_stat, bp_pvalue]


def compute_metrics(data):
    # basic
    df = describe(data)
    df.drop(df.iloc[-9:, :].index, inplace=True)
    df.rename(index={"range": "min_max_range"}, inplace=True)
    # normal distribution test
    # jb, jbpv, skew, kurtosis = jarque_bera(x)
    tstat, tpv = ztest(data)
    dw = durbin_watson(data)
    df.loc["tstat"] = tstat
    df.loc["tpv"] = tpv
    df.loc["dw"] = dw

    # stationary test
    adf_df = data.apply(adfuller, axis=0).loc[:1]
    adf_df.index = ["adf", "adfpv"]
    # 注意ADF和KPSS的原假设是反着的
    kpss_df = data.apply(kpss, axis=0).loc[:1]
    kpss_df.index = ["kpss", "kpsspv"]

    # auto-correlation / white noise test
    lb_df = data.apply(white_noise_test, axis=0)
    lb_df.index = ["lb_minp_lag", "lb_stat", "lb_pvalue", "bp_stat", "bp_pvalue"]

    df = pd.concat([df, adf_df, kpss_df, lb_df])

    df = df.transpose()
    df.index.names = ["variable", "ticker"]
    logger.info(f"metrics df {df}")
    df.reset_index(inplace=True)
    return df




class StockZhATest(BaseData):
    def get_df_schema(self):
        data = get_return(self.ds)
        df = compute_metrics(data)
        return df

    def get_table_name(self):
        return "stock_zh_a_test"

    def before_retrieve_data(self):
        pass


if __name__ == '__main__':
    ds = sys.argv[1]
    logger.info("execute task on ds {}".format(ds))
    if not is_trade_date(ds):
        logger.info(f"{ds} is not trade date. task exits.")
        exit(os.EX_OK)
    data = StockZhATest()
    data.set_ds(ds)
    data.retrieve_data()


