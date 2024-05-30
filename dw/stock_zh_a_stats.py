import sys, os
parent_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parent_dir)

from datetime import datetime, timedelta
import traceback

import numpy as np
import pandas as pd
import quantstats as qs
import re
from joblib import Parallel, delayed
from tqdm import tqdm

from utils.stock_zh_a_util import get_zh_securities_data, get_securities_list, get_index_list, get_benchmark_data, get_stock_list, is_trade_date, get_fund_etf_data, get_fund_etf_list, get_fund_lof_list
from base_data import BaseData, BaseDataHelper

from utils.log_util import get_logger
logger = get_logger(__name__)


class DataHelper(BaseDataHelper):
    def __init__(self, ds):
        self.ds = ds
        self.start_date = (datetime.strptime(self.ds, '%Y%m%d') - timedelta(days=365)).strftime("%Y-%m-%d")
        self.met_wide, self.benchmark = metrics_bootstrap(self.ds, self.start_date, benchmark_symbol="sh000300")
        super().__init__(parallel=1, loops_per_second_max=None)

    def _get_all_symbols(self):
        return get_securities_list()

    def _get_downloaded_symbols(self):
        data = StockZhAStats(ds=self.ds)
        return data.get_downloaded_symbols()

    def _fetch_symbol_data(self, symbol):
        logger.info(f"symbol {symbol}")
        input_data = get_zh_securities_data(symbol=symbol, ds=self.ds, start_date=self.start_date, security_type="all")

        if input_data is None or len(input_data) == 0:
            logger.info("empty dataset, no computation required.")
            return

        input_data = input_data.reset_index(level="代码")
        data = StockZhAStats(ds=self.ds, symbol=symbol, start_date=self.start_date, input_data=input_data, benchmark=self.benchmark, met_wide=self.met_wide)
        data.retrieve_data()

    def _clean_up_history(self):
        data = StockZhAStats(ds=self.ds)
        data.clean_up_history(lifecycle=15)




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


def get_one_stock_metrics(stock_symbol, stock, benchmark, met_wide):
    df = pd.DataFrame(columns=met_wide.columns)
    logger.info(f"computing metrics for {stock_symbol}, stock data size {len(stock)}, benchmark size {len(benchmark)}")
    if stock.empty:
        logger.info(f"stock {stock_symbol} empty data.")
        return df
    stock = stock["涨跌幅"] / 100
    stock.name = stock_symbol
    sorted_index = np.argsort(stock.index)
    stock = stock.iloc[sorted_index]

    if len(stock) < 90:
        logger.info(f"stock {stock_symbol} size {len(stock)}, not enough data.")
        return df
    try:
        metrics = qs.reports.metrics(stock, benchmark, mode="full", display=False)
        row = [stock_symbol]
        row.extend(metrics["Strategy"].tolist())
        df = pd.DataFrame(data=[row], columns=met_wide.columns)
        df = handle_special_values(df)
        logger.info(f"metrics computation finished, values {df}")
    except:
        logger.error(f"Exception occurred, stock {stock_symbol}")
        logger.error(traceback.format_exc())
    return df


def compute_metrics(ds, stock_symbols=None):
    start_date = (datetime.strptime(ds, '%Y%m%d') - timedelta(days=190)).strftime("%Y-%m-%d")
    benchmark_symbol = "sh000300"
    met_wide, benchmark = metrics_bootstrap(ds, start_date, benchmark_symbol)
    stocks = get_zh_securities_data(symbol="all", ds=ds, start_date=start_date, security_type="all")
    stocks = stocks.reset_index(level="代码")
    if stock_symbols is None or len(stock_symbols) == 0:
        stock_symbols = get_securities_list()
    results = Parallel(n_jobs=1, backend="multiprocessing")(delayed(get_one_stock_metrics)(symbol, stocks[stocks["代码"]==symbol], benchmark) for symbol in tqdm(stock_symbols))
    results = [row for row in results if len(row) > 0]
    met_wide = pd.concat([met_wide, pd.DataFrame(data=results, columns=met_wide.columns)])
    met_wide = handle_special_values(met_wide)
    return met_wide


def handle_special_values(df):
    for col in df.columns:
        df[col] = df[col].replace("-", np.NaN)
    return df


def compute_report(ds="20240228", stock_symbol="600066", days_ahead=400):
    start_date = (datetime.strptime(ds, '%Y%m%d') - timedelta(days=days_ahead)).strftime("%Y-%m-%d")
    benchmark_symbol = "sh000300"
    benchmark = get_benchmark_data(benchmark_symbol, ds, start_date)
    benchmark = benchmark["涨跌幅"] / 100
    benchmark.name = benchmark_symbol
    stock = get_zh_securities_data(stock_symbol, ds, start_date)
    stock = stock["涨跌幅"] / 100
    stock.name = stock_symbol
    logger.info(f"computing metrics for {stock_symbol}, stock data size {len(stock)}, benchmark size {len(benchmark)}")
    file_name = f"qs_{ds}_{stock_symbol}.html"
    qs.reports.html(stock, benchmark, output=file_name)
    logger.info(f"report generated, file name {file_name}")


class StockZhAStats(BaseData):
    def __init__(self, ds, symbol=None, start_date=None, input_data=None, benchmark=None, met_wide=None):
        super().__init__(ds)
        self.symbol = symbol
        self.start_date = start_date
        self.input_data = input_data
        self.benchmark = benchmark
        self.met_wide = met_wide
    def get_df_schema(self):
        df = get_one_stock_metrics(self.symbol, self.input_data, self.benchmark, self.met_wide)
        return df

    def get_table_name(self):
        return "stock_zh_a_stats"

    def before_retrieve_data(self):
        pass

    def get_downloaded_symbols(self):
        if not self.table_exists():
            logger.info(f"table {self.table_name} does not exist, no downloaded symbols")
            return []
        sql = f"select distinct symbol from {self.table_name} where ds = '{self.ds}'"
        results = self.db.run_sql(sql)
        return [result[0] for result in results]

if __name__ == '__main__':
    ds = sys.argv[1]
    logger.info("execute task on ds {}".format(ds))
    if not is_trade_date(ds):
        logger.info(f"{ds} is not trade date. task exits.")
        exit(os.EX_OK)
    DataHelper(ds).fetch_all_data()


    # compute_report("20240401", "562500", etf=True)
    # compute_metrics("20240229", ["300234","301337","301338"])
