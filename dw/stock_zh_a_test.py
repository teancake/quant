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


from utils.stock_zh_a_util import get_stock_data, is_trade_date, get_zh_securities_data
from dw.base_data import BaseData

from utils.log_util import get_logger
logger = get_logger(__name__)

def get_return(ds, security_type, days_ahead=365):
    start_date = datetime.strptime(ds, "%Y%m%d") - timedelta(days=days_ahead)
    df = get_zh_securities_data("all", ds=ds, start_date=start_date, yf_compatible=True, security_type=security_type)
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


def func_wrapper(func):
    def wrapped_func(x):
        try:
            return func(x)
        except Exception as e:
            logger.warning(f"exception in {func}, row value {x}")
            raise e
    return wrapped_func




def compute_metrics(data):
    # basic
    df = describe(data)
    df.drop(df.iloc[-9:, :].index, inplace=True)
    df.rename(index={"range": "min_max_range"}, inplace=True)
    logger.info(f"descriptive metrics {df}")
    # normal distribution test
    # jb, jbpv, skew, kurtosis = jarque_bera(x)
    tstat, tpv = ztest(data)
    dw = durbin_watson(data)
    df.loc["tstat"] = tstat
    df.loc["tpv"] = tpv
    df.loc["dw"] = dw

    # stationary test
    adf_df = data.apply(func_wrapper(adfuller), axis=0).loc[:1]
    adf_df.index = ["adf", "adfpv"]
    # 注意ADF和KPSS的原假设是反着的
    kpss_df = data.apply(func_wrapper(kpss), axis=0).loc[:1]
    kpss_df.index = ["kpss", "kpsspv"]

    # auto-correlation / white noise test
    lb_df = data.apply(func_wrapper(white_noise_test), axis=0)
    lb_df.index = ["lb_minp_lag", "lb_stat", "lb_pvalue", "bp_stat", "bp_pvalue"]

    df = pd.concat([df, adf_df, kpss_df, lb_df])

    df = df.transpose()
    df.index.names = ["variable", "ticker"]
    logger.info(f"metrics df {df}")
    df.reset_index(inplace=True)
    return df




class StockZhATest(BaseData):
    def __init__(self, ds, security_type="stock_zh_a"):
        super().__init__(ds)
        self.security_type = security_type
    def get_df_schema(self):
        data = get_return(self.ds, security_type = self.security_type)
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
    logger.info("computing stock_zh_a")
    StockZhATest(ds, security_type="stock_zh_a").retrieve_data()
    logger.info("computing fund_etf")
    StockZhATest(ds, security_type="fund_etf").retrieve_data()
    # logger.info("computing fund_lof")
    # StockZhATest(ds, security_type="fund_lof").retrieve_data()


