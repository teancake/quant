import sys, os

import numpy as np

parent_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parent_dir)

from datetime import datetime, timedelta
import traceback

import pandas as pd
import quantstats as qs
import re
from joblib import Parallel, delayed

from utils.stock_zh_a_util import get_stock_data, get_index_list, get_benchmark_data, get_stock_list, is_trade_date
from base_data import BaseData

from utils.log_util import get_logger
logger = get_logger(__name__)


def get_underscore_names(cols):
    res = {}
    for col in cols:
        temp = re.sub(r"[ %./\\()^√﹪-]", "_", col)
        temp = re.sub(r"_+", "_", temp)
        temp = re.sub(r"_$", "", temp)
        res[col] = temp.lower()
    return res


def metrics_bootstrap(ds, start_date, benchmark_symbol):
    benchmark = get_benchmark_data(benchmark_symbol, ds, start_date)
    benchmark = benchmark["涨跌幅"] / 100
    benchmark.name = benchmark_symbol
    sorted_index = np.argsort(benchmark.index)
    benchmark = benchmark.iloc[sorted_index]

    metrics = qs.reports.metrics(benchmark, benchmark, mode="full", display=False)

    met_wide = metrics.transpose().reset_index()
    met_wide.rename(columns={"index": "symbol"}, inplace=True)
    met_wide.rename(columns=get_underscore_names(met_wide.columns), inplace=True)
    met_wide.loc[0, "symbol"] = benchmark_symbol
    met_wide = met_wide.drop(1)
    return met_wide, benchmark


def get_one_stock_metrics(stock_symbol, stock, benchmark):
    row = []
    logger.info(f"computing metrics for {stock_symbol}, stock data size {len(stock)}, benchmark size {len(benchmark)}")
    if stock.empty:
        logger.info(f"stock {stock_symbol} empty data.")
        return row
    stock = stock["涨跌幅"] / 100
    stock.name = stock_symbol
    sorted_index = np.argsort(stock.index)
    stock = stock.iloc[sorted_index]

    if len(stock) < 90:
        logger.info(f"stock {stock_symbol} size {len(stock)}, not enough data.")
        return row
    try:
        metrics = qs.reports.metrics(stock, benchmark, mode="full", display=False)
        row = [stock_symbol]
        row.extend(metrics["Strategy"].tolist())
    except:
        logger.error(f"Exception occurred, stock {stock_symbol}")
        logger.error(traceback.format_exc())
    logger.info(f"metrics computation finished, values {row}")
    return row


def compute_metrics(ds, stock_symbols=None):
    start_date = (datetime.strptime(ds, '%Y%m%d') - timedelta(days=190)).strftime("%Y-%m-%d")
    benchmark_symbol = "sh000300"
    met_wide, benchmark = metrics_bootstrap(ds, start_date, benchmark_symbol)
    stocks = get_stock_data("all", ds, start_date)
    if stock_symbols is None or len(stock_symbols) == 0:
        stock_symbols = get_stock_list()
    results = Parallel(n_jobs=4, backend="multiprocessing")(delayed(get_one_stock_metrics)(symbol, stocks[stocks["代码"]==symbol], benchmark) for symbol in stock_symbols)
    results = [row for row in results if len(row) > 0]
    met_wide = pd.concat([met_wide, pd.DataFrame(data=results, columns=met_wide.columns)])
    met_wide = handle_special_values(met_wide)
    return met_wide


def handle_special_values(df):
    for col in df.columns:
        df[col] = df[col].replace("-", np.NaN)
    return df


def compute_report(ds="20240228", stock_symbol="600066"):
    start_date = (datetime.strptime(ds, '%Y%m%d') - timedelta(days=190)).strftime("%Y-%m-%d")
    benchmark_symbol = "sh000300"
    benchmark = get_benchmark_data(benchmark_symbol, ds, start_date)
    benchmark = benchmark["涨跌幅"] / 100
    benchmark.name = benchmark_symbol
    stock = get_stock_data(stock_symbol, ds, start_date)
    stock = stock["涨跌幅"] / 100
    stock.name = stock_symbol
    logger.info(f"computing metrics for {stock_symbol}, stock data size {len(stock)}, value {stock}")
    qs.reports.html(stock, benchmark, output="test.html")
    logger.info("report generated.")


class StockZhAStats(BaseData):
    def get_df_schema(self):
        return compute_metrics(self.ds)

    def get_table_name(self):
        return "stock_zh_a_stats"

    def before_retrieve_data(self):
        pass


if __name__ == '__main__':
    ds = sys.argv[1]
    logger.info("execute task on ds {}".format(ds))
    if not is_trade_date(ds):
        logger.info(f"{ds} is not trade date. task exits.")
        exit(os.EX_OK)
    data = StockZhAStats()
    data.set_ds(ds)
    data.retrieve_data()

    # compute_report("20240229", "301338")
    # compute_metrics("20240229", ["300234","301337","301338"])
