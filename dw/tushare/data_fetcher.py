# -*- coding: utf-8 -*-
"""
阿尔法收割者

Project: alphasickle
Author: Moses
E-mail: 8342537@qq.com
"""
import os, sys

parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(parent_dir)

import time
import traceback
from datetime import datetime
import pandas.tseries.offsets as toffsets


import numpy as np
import pandas as pd
import tushare as ts
from retrying import retry
from functools import wraps

from abc import ABC, abstractmethod

try:
    basestring
except NameError:
    basestring = str

from tqdm import tqdm

from utils.db_util import DbUtil
from utils.log_util import get_logger
from utils.starrocks_db_util import StarrocksDbUtil

logger = get_logger(__name__)

# 打印能完整显示
# pd.set_option('display.max_columns', 100)
# pd.set_option('display.max_rows', None)
# pd.set_option('display.width', 50000)
# pd.set_option('max_colwidth', 1000)

from utils.config_util import get_data_config

import pytz


def segment_op(limit, _max):
    """ 分段获取数据
    """

    def segment_op_(f):
        #
        @wraps(f)
        def wrapper(*args, **kwargs):
            dfs = []
            for i in range(0, _max, limit):
                kwargs['offset'] = i
                df = f(*args, **kwargs)
                if len(df) < limit:
                    if len(df) > 0:
                        dfs.append(df)
                    break
                df = df.iloc[0:limit]
                dfs.append(df)
            if len(dfs) > 0:
                return pd.concat(dfs, ignore_index=True)
            else:
                return pd.DataFrame()

        #
        return wrapper

    #
    return segment_op_


class BaseData(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def open_file(self, name):
        pass

    def close_file(self, df, name, **kwargs):
        self._close_file(df, name, **kwargs)

    @abstractmethod
    def _close_file(self, df, name, **kwargs):
        pass

    def get_last_month_end(self, date):
        index_weight_start_dt = pd.to_datetime(date).replace(day=1) - pd.Timedelta("1 day")
        return index_weight_start_dt.strftime('%Y%m%d')

class LocalData(BaseData):

    metafile = 'all_stocks.csv'
    mmapfile = 'month_map.csv'
    month_group_file = 'month_group.csv'
    tradedays_file = 'tradedays.csv'
    tdays_be_m_file = 'trade_days_begin_end_of_month.csv'


    def __init__(self, start_dt=None, end_dt=None, root_dir="raw_data"):

        super().__init__()
        work_path = os.path.join(os.path.dirname(__file__), root_dir)
        src_path = os.path.join(work_path, "src")
        os.umask(0)
        os.makedirs(work_path, exist_ok=True)
        os.makedirs(src_path, exist_ok=True)
        logger.info(f"{work_path} and {src_path} created.")

        self.start_dt = start_dt
        self.end_dt = end_dt

        self.root = work_path
        self.src = src_path
        self.freqmap = {}
        self.__update_frepmap()
        logger.info("LocalData initialized.")

    def __update_frepmap(self):
        logger.info(f"update frepmap {self.freqmap}")
        self.freqmap.update({name.split(".")[0]: self.root for name in os.listdir(self.root)})

    def __update_attr(self, name):
        logger.info(f"update attr {name}")
        if name in self.__dict__:
            del self.__dict__[name]
        self.__dict__[name] = getattr(self, name, None)

    def __getattr__(self, name):
        logger.info(f"get attr {name}")
        if name not in self.__dict__:
            self.__dict__[name] = self.open_file(name)
            logger.info(f"{name} file opened, attr value is {str(self.__dict__[name])[:100]} ...")
        return self.__dict__[name]

    def open_file(self, name:str):
        logger.info(f"open file for {name}")
        if name == 'meta':
            fn = os.path.join(self.root, self.src, self.metafile)
            logger.info(f"file name is {fn}")
            df = pd.read_csv(fn, index_col=[0], parse_dates=['ipo_date', "delist_date"])
            logger.info(f"meta data {df}")
            return df
        elif name == 'month_map':
            return pd.read_csv(os.path.join(self.root, self.src, self.mmapfile), index_col=[0], parse_dates=[0, 1])[
                'calendar_date']
        elif name == 'trade_days_begin_end_of_month':
            return pd.read_csv(os.path.join(self.root, self.src, self.tdays_be_m_file), index_col=[1],
                               parse_dates=[0, 1])
        elif name == 'month_group':
            return pd.read_csv(os.path.join(self.root, self.src, self.month_group_file), index_col=[0],
                               parse_dates=True)
        elif name == 'tradedays':
            return pd.read_csv(os.path.join(self.root, self.src, self.tradedays_file), index_col=[0],
                               parse_dates=True).index.tolist()

        path = self.freqmap.get(name, None)
        if path is None:
            path = self.root
        try:
            fn = os.path.join(str(path), f"{name}.csv")
            logger.info(f"open file {fn}")
            dat = pd.read_csv(fn, index_col=[0], engine='python', parse_dates=True)
            # dat = pd.DataFrame(data=dat, index=dat.index.union(self.meta.index), columns=dat.columns)
        except TypeError:
            logger.info("exception occurred.")
            print(name, path)
            raise
        # dat.columns = pd.to_datetime(dat.columns)
        #if name in ('stm_issuingdate', 'applied_rpt_date_M'):
        #    dat = dat.replace('0', np.nan)
        #    dat = dat.applymap(pd.to_datetime)
        return dat

    def _close_file(self, df, name, **kwargs):
        logger.info(f"saving data to local file {name}")
        if name == 'meta':
            file_name = os.path.join(self.root, self.src, self.metafile)
            df.to_csv(file_name, **kwargs)
            logger.info(f"file saved to {file_name}")
        elif name == 'month_map':
            file_name = os.path.join(self.root, self.src, self.mmapfile)
            df.to_csv(file_name, **kwargs)
            logger.info(f"file saved to {file_name}")
        elif name == 'trade_days_begin_end_of_month':
            file_name = os.path.join(self.root, self.src, self.tdays_be_m_file)
            df.to_csv(file_name, **kwargs)
            logger.info(f"file saved to {file_name}")
        elif name == 'month_group':
            file_name = os.path.join(self.root, self.src, self.month_group_file)
            df.to_csv(file_name, **kwargs)
            logger.info(f"file saved to {file_name}")
        elif name == 'tradedays':
            file_name = os.path.join(self.root, self.src, self.tradedays_file)
            df.to_csv(file_name, **kwargs)
            logger.info(f"file saved to {file_name}")
        else:
            if "/" in name:
                temp_names = name.rsplit("/", 1)
                dir_name = temp_names[0]
                file_name = temp_names[1]
                tmp_dir = os.path.join(self.root, dir_name)
                os.makedirs(tmp_dir, exist_ok=True)
                logger.info(f"dir {tmp_dir} created.")

            path = self.freqmap.get(name, None)
            if path is None:
                path = self.root
            #if name in ['stm_issuingdate', 'applied_rpt_date_M']:
            #    df = df.replace(0, pd.NaT)
            file_name = os.path.join(path, name + '.csv')
            df.to_csv(file_name, **kwargs)
            logger.info(f"file saved to {file_name}")
            self.__update_frepmap()

    #    @staticmethod
    #    def _fill_nan(series, value=0, ffill=False):
    #        if ffill:
    #            series = series.fillna(method='ffill')
    #        else:
    #            if value:
    #                start_valid_idx = np.where(pd.notna(series))[0][0]
    #                series.loc[start_valid_idx:] = series.loc[start_valid_idx:].fillna(0)
    #        return series


    def _generate_month_group(self, start_year=None, end_year=None):

        start_idx = 0
        for i, m in enumerate(self.month_map.index):
            if m.month == 1:
                start_idx = i
                break
        ori_syear, ori_eyear = self.month_map.index[start_idx].year, self.month_map.index[-1].year

        if start_year is None and end_year is None:
            start_year, end_year = ori_syear, ori_eyear

        month_group = pd.DataFrame(index=self.month_map.index[start_idx:])

        group1 = [[None] * 2 + [i + 1] * 5 + [None] * 5 for i in range(len(range(ori_syear, ori_eyear + 1)))]
        group1 = [i for group in group1 for i in group]

        group2 = [[None] * 5 + [i + 1] + [None] + [i + 1] * 2 + [None] * 3 for i in
                  range(len(range(ori_syear, ori_eyear + 1)))]
        group2 = [i for group in group2 for i in group]

        group3 = [[i] * 3 + [None] * 5 + [i + 1] * 4 for i in range(len(range(ori_syear, ori_eyear + 1)))]
        group3 = [i for group in group3 for i in group]

        month_group['Q1'] = group1
        month_group['Q2'] = group2
        month_group['Q3'] = group3

        month_group = pd.concat([month_group, self.month_map], axis=1)
        month_group = month_group.loc[datetime(start_year,1,1):datetime(end_year,12,31)]
        self.close_file(month_group, "month_group")


class DbData(BaseData):

    def __init__(self, ds):
        super().__init__()
        self.ds = ds
        self.db = DbUtil()
        self.dw = StarrocksDbUtil()

    def df_from_sql(self, sql, index_columns=None, date_columns=None):
        res = self.dw.run_sql(sql)
        df = pd.DataFrame(res)
        if date_columns is not None:
            for item in date_columns:
                df[item] = pd.to_datetime(df[item])
        if index_columns is not None:
            df = df.set_index(index_columns)
        return df

    def get_meta(self, parse_dates):
        sql = f"select * from dwd_tushare_stock_basic_df where ds = '{self.ds}'"
        res = self.dw.run_sql(sql)
        df = pd.DataFrame(res)
        df.set_index(["code"])
        for item in parse_dates:
            df[item] = pd.to_datetime(df[item])
        return df

    def get_month_map(self, parse_dates):
        sql = f"select * from dwd_tushare_month_map_df where ds = '{self.ds}'"
        res = self.dw.run_sql(sql)
        df = pd.DataFrame(res)
        df.set_index(["trade_date"])
        for item in parse_dates:
            df[item] = pd.to_datetime(df[item])
        return df[["calendar_date"]]

    def get_tradedays(self):
        sql = f"select * from dwd_tushare_trade_cal_df where ds = '{self.ds}'"
        res = self.dw.run_sql(sql)
        df = pd.DataFrame(res)
        df["tradedays"] = pd.to_datetime(df["tradedays"])
        return df["tradedays"].tolist()

    def get_meta_data(self):
        sql = f"select * from dwd_tushare_daily_basic_df where ds = '{self.ds}'"
        res = self.dw.run_sql(sql)
        df = pd.DataFrame(res)
        df["trade_date"] = pd.to_datetime(df["trade_date"])
        df.set_index(["ts_code", "trade_date"])
        return df

    def get_factor_data(self, name):
        sql = f"select ts_code, trade_date, {name} from dwd_tushare_factor_df where ds = '{self.ds}'"
        res = self.dw.run_sql(sql)
        df = pd.DataFrame(res)
        df["trade_date"] = pd.to_datetime(df["trade_date"])
        df.set_index(["ts_code", "trade_date"])
        return df

    def get_cashflow(self):
        sql = f"select * from dwd_tushare_cashflow_df where ds = '{self.ds}'"
        res = self.dw.run_sql(sql)
        df = pd.DataFrame(res)
        df["ann_date"] = pd.to_datetime(df["ann_date"])
        df["end_date"] = pd.to_datetime(df["end_date"])
        df.set_index(["ts_code", "end_date"], inplace=True)
        return df

    def get_balancesheet(self):
        sql = f"select * from dwd_tushare_balancesheet_df where ds = '{self.ds}'"
        res = self.dw.run_sql(sql)
        df = pd.DataFrame(res)
        df["ann_date"] = pd.to_datetime(df["ann_date"])
        df["end_date"] = pd.to_datetime(df["end_date"])
        df.set_index(["ts_code", "end_date"], inplace=True)
        return df

    def get_income(self):
        sql = f"select * from dwd_tushare_income_df where ds = '{self.ds}'"
        res = self.dw.run_sql(sql)
        df = pd.DataFrame(res)
        df["ann_date"] = pd.to_datetime(df["ann_date"])
        df["end_date"] = pd.to_datetime(df["end_date"])
        df.set_index(["ts_code", "end_date"], inplace=True)
        return df

    def open_file(self, name):
        logger.warning("not implemented")
        return
        if name == "meta":
            return self.get_meta(parse_dates=["ipo_date", "delist_date"])
        elif name == "month_map":
            return self.get_month_map(parse_dates=["trade_date", "calendar_date"])
        elif name == "tradedays":
            return self.get_tradedays()
        elif name == 'trade_days_begin_end_of_month':
            logger.error("trade_days_begin_end_of_month does not exist.")
            return
        elif name == 'month_group':
            logger.error("month_group does not exist.")
            return


    def _close_file(self, df, name, **kwargs):
        self.df_to_sql(df, table_name=name, ds=kwargs["ds"])

    def df_to_sql(self, df: pd.DataFrame, table_name, ds):
        dt = datetime.now().astimezone(pytz.timezone("Asia/Shanghai"))
        df.insert(0, "gmt_create", dt)
        df.insert(1, "gmt_modified", dt)
        df.insert(len(df.columns), "ds", ds)
        logger.info(f"df index name is {df.index.name}")
        save_index = False if df.index.name is None else True
        df.to_sql(name=table_name, con=self.db.get_db_engine(), if_exists='append', index=save_index, method="multi",
                  chunksize=5000)
        logger.info(f"{len(df)} records saved to db, table name {table_name}")


    def get_downloaded_dates(self, table_name, date_column, daily):
        if not self.db.table_exists(table_name):
            return []
        where_cond = f"where ds = {self.ds}" if daily else ""
        sql = f"select distinct {date_column} from {table_name} {where_cond}"
        recs = self.db.run_sql(sql)
        return [rec[0] for rec in recs]

class DataAccessor:
    def __init__(self, ds, start_dt, end_dt):
        self.data = LocalData(start_dt=start_dt, end_dt=end_dt)
        self.db = DbData(ds)
        self.ds = ds

    def close_file(self, df, file_name, table_name=None, **kwargs):
        self.data.close_file(df, file_name, **kwargs)
        if table_name is not None:
            logger.info(f"saving data to db, table name {table_name}, ds {self.ds}")
            self.db.close_file(df, table_name, ds=self.ds)

    def __getattr__(self, name):
        return getattr(self.data, name, None)

    def get_downloaded_dates(self, table_name, date_column, daily=True):
        return self.db.get_downloaded_dates(table_name, date_column, daily)

class BaseDataFetcher(DataAccessor):

    def __init__(self, start_dt, end_dt, ds):
        super().__init__(ds=ds, start_dt=start_dt, end_dt=end_dt)

    def sleep_if_too_fast(self, timer_start, minimum_loop_time_ms=150):
        # 每秒最多7次，每分钟420次， TUSHARE接口限制每分钟500次
        dt = time.time() - timer_start
        if dt < minimum_loop_time_ms / 1000.0:
            logger.info(f"dt is {dt}, sleep {dt} seconds")
            time.sleep(dt)


    @retry(stop_max_attempt_number=3, wait_random_min=1000, wait_random_max=60000)
    def ensure_data(self, func, save_dir, table_name=None):
        """ 确保按交易日获取数据
        """
        logger.info(f"ensure_data {func.__name__}")
        dl = self.get_downloaded_dates(table_name, date_column="trade_date")
        logger.info(f"downloaded  dates {dl}")
        dl = sorted(dl)
        s = pd.to_datetime(self.start_dt)
        e = pd.to_datetime(self.end_dt)
        tdays = pd.Series(self.tradedays, index=self.tradedays)
        tdays = tdays[(tdays >= s) & (tdays <= e)]
        tdays = tdays.index.tolist()
        for tday in tdays:
            logger.info(f"download trade day {tday}")
            t = tday.strftime("%Y%m%d")
            if t in dl:
                logger.info(f"{tday} data already downloaded, skip.")
                continue
            timer_start = time.time()
            datdf = func(t)
            # path = os.path.join(tmp_dir, t + ".csv")
            self.close_file(datdf, file_name=f"{save_dir}/{t}", table_name=table_name)
            self.sleep_if_too_fast(timer_start)

    @retry(stop_max_attempt_number=3, wait_random_min=1000, wait_random_max=2000)
    def ensure_data_by_q(self, func, save_dir, table_name=None):
        """ 确保按季度获取数据
        """
        dl = self.get_downloaded_dates(table_name, date_column="end_date", daily=False)
        dl = sorted(dl)
        logger.info(f"downloaded  dates {dl}")
        if len(dl) > 3:
            dl = dl[0:len(dl) - 3]  # 已经存在的最后三个季度数据重新下载

        logger.info(f"downloaded  dates new {dl}")
        s = pd.to_datetime(self.start_dt)
        e = pd.to_datetime(self.end_dt)
        qdates = pd.date_range(start=s, end=e, freq='Q')
        qdates = qdates.tolist()
        for tday in qdates:
            logger.info(f"download trade day {tday}")
            t = tday.strftime("%Y%m%d")
            if t in dl:
                logger.info(f"{tday} data already exists, skip.")
                continue

            timer_start = time.time()
            datdf = func(period=t)
            self.close_file(datdf, file_name=f"{save_dir}/{t}", table_name=table_name)
            self.sleep_if_too_fast(timer_start)


class TushareFetcher(BaseDataFetcher):

    def __init__(self, start_dt, end_dt, ds):
        self.pro = ts.pro_api(get_data_config()["tushare_token"])
        super().__init__(start_dt, end_dt, ds)

    def fetch_meta_data(self):
        """ 股票基础信息
        """
        logger.info("obtaining meta data")
        df_list = []
        fields = "ts_code,symbol,name,area,industry,fullname,enname,cnspell,market,exchange,curr_type,list_status,list_date,delist_date,is_hs,act_name,act_ent_type"
        df = self.pro.stock_basic(exchange='', fields=fields)
        df_list.append(df)
        df = self.pro.stock_basic(exchange='', fields=fields, list_status='D')
        df_list.append(df)
        df = self.pro.stock_basic(exchange='', fields=fields, list_status='P')
        df_list.append(df)
        df = pd.concat(df_list)
        df = df.rename(columns={"list_date": "ipo_date"})
        df = df.rename(columns={'name': 'sec_name'})
        df = df.rename(columns={"ts_code": "code"})
        df.drop_duplicates(subset=['code'], keep='first', inplace=True)
        df.sort_values(by=['ipo_date'], inplace=True)
        # print(pd.to_datetime(df['ipo_date']))
        # df.reset_index(drop=True, inplace=True)
        df.set_index(['code'], inplace=True)
        self.close_file(df, 'meta', table_name="tushare_stock_basic")
        logger.info("meta data obtained.")

    def fetch_trade_day(self):
        """ 交易日列表
        """
        logger.info("obtaining trade day")
        df = self.pro.trade_cal(is_open='1')
        df = df[['cal_date', 'is_open']]
        df = df.rename(columns={"cal_date": "tradedays"})
        df.set_index(['tradedays'], inplace=True)
        self.close_file(df, 'tradedays', table_name="tushare_trade_cal")
        logger.info("trade day obtained.")

    def fetch_month_map(self):
        """ 每月最后一个交易日和每月最后一个日历日的映射表
        """
        logger.info("obtaining month map")
        tdays = self.tradedays
        s_dates = pd.Series(tdays, index=tdays)
        func_last = lambda ser: ser.iat[-1]
        new_dates = s_dates.resample('M').apply(func_last)
        month_map = new_dates.to_frame(name='trade_date')
        month_map.index.name = 'calendar_date'
        month_map.reset_index(inplace=True)
        month_map.set_index(['trade_date'], inplace=True)
        self.close_file(month_map, 'month_map')
        logger.info("month map obtained.")

    # ------------------------------------------------------------------------------------
    # 日数据
    def daily(self, t):
        logger.info(f"obtaining daily {t}")
        return self.pro.daily(trade_date=t)

    def stk_factor(self, t):
        logger.info(f"obtaining stk_factor {t}")
        return self.pro.stk_factor(trade_date=t)

    def suspend_d(self, t):
        logger.info(f"obtaining suspend_d {t}")
        return self.pro.suspend_d(trade_date=t)

    def limit_list(self, t):
        logger.info(f"obtaining limit_list {t}")
        return self.pro.limit_list(trade_date=t)

    def adj_factor(self, t):
        logger.info(f"obtaining adj_factor {t}")
        return self.pro.adj_factor(trade_date=t)

    def daily_basic(self, t):
        logger.info(f"obtaining daily_basic {t}")
        return self.pro.daily_basic(trade_date=t)

    def moneyflow(self, t):
        logger.info(f"obtaining moneyflow {t}")
        return self.pro.moneyflow(trade_date=t)

    # ------------------------------------------------------------------------------------
    # 季度数据
    @segment_op(limit=5000, _max=100000)
    def fina_indicator(self, *args, **kwargs):
        logger.info(f"obtaining fina_indicator, args {args}, kwargs {kwargs}")
        fields = '''ts_code,
        ann_date,
        end_date,
        eps,
        dt_eps,
        total_revenue_ps,
        revenue_ps,
        capital_rese_ps,
        surplus_rese_ps,
        undist_profit_ps,
        extra_item,
        profit_dedt,
        gross_margin,
        current_ratio,
        quick_ratio,
        cash_ratio,
        invturn_days,
        arturn_days,
        inv_turn,
        ar_turn,
        ca_turn,
        fa_turn,
        assets_turn,
        op_income,
        valuechange_income,
        interst_income,
        daa,
        ebit,
        ebitda,
        fcff,
        fcfe,
        current_exint,
        noncurrent_exint,
        interestdebt,
        netdebt,
        tangible_asset,
        working_capital,
        networking_capital,
        invest_capital,
        retained_earnings,
        diluted2_eps,
        bps,
        ocfps,
        retainedps,
        cfps,
        ebit_ps,
        fcff_ps,
        fcfe_ps,
        netprofit_margin,
        grossprofit_margin,
        cogs_of_sales,
        expense_of_sales,
        profit_to_gr,
        saleexp_to_gr,
        adminexp_of_gr,
        finaexp_of_gr,
        impai_ttm,
        gc_of_gr,
        op_of_gr,
        ebit_of_gr,
        roe,
        roe_waa,
        roe_dt,
        roa,
        npta,
        roic,
        roe_yearly,
        roa2_yearly,
        roe_avg,
        opincome_of_ebt,
        investincome_of_ebt,
        n_op_profit_of_ebt,
        tax_to_ebt,
        dtprofit_to_profit,
        salescash_to_or,
        ocf_to_or,
        ocf_to_opincome,
        capitalized_to_da,
        debt_to_assets,
        assets_to_eqt,
        dp_assets_to_eqt,
        ca_to_assets,
        nca_to_assets,
        tbassets_to_totalassets,
        int_to_talcap,
        eqt_to_talcapital,
        currentdebt_to_debt,
        longdeb_to_debt,
        ocf_to_shortdebt,
        debt_to_eqt,
        eqt_to_debt,
        eqt_to_interestdebt,
        tangibleasset_to_debt,
        tangasset_to_intdebt,
        tangibleasset_to_netdebt,
        ocf_to_debt,
        ocf_to_interestdebt,
        ocf_to_netdebt,
        ebit_to_interest,
        longdebt_to_workingcapital,
        ebitda_to_debt,
        turn_days,
        roa_yearly,
        roa_dp,
        fixed_assets,
        profit_prefin_exp,
        non_op_profit,
        op_to_ebt,
        nop_to_ebt,
        ocf_to_profit,
        cash_to_liqdebt,
        cash_to_liqdebt_withinterest,
        op_to_liqdebt,
        op_to_debt,
        roic_yearly,
        total_fa_trun,
        profit_to_op,
        q_opincome,
        q_investincome,
        q_dtprofit,
        q_eps,
        q_netprofit_margin,
        q_gsprofit_margin,
        q_exp_to_sales,
        q_profit_to_gr,
        q_saleexp_to_gr,
        q_adminexp_to_gr,
        q_finaexp_to_gr,
        q_impair_to_gr_ttm,
        q_gc_to_gr,
        q_op_to_gr,
        q_roe,
        q_dt_roe,
        q_npta,
        q_opincome_to_ebt,
        q_investincome_to_ebt,
        q_dtprofit_to_profit,
        q_salescash_to_or,
        q_ocf_to_sales,
        q_ocf_to_or,
        basic_eps_yoy,
        dt_eps_yoy,
        cfps_yoy,
        op_yoy,
        ebt_yoy,
        netprofit_yoy,
        dt_netprofit_yoy,
        ocf_yoy,
        roe_yoy,
        bps_yoy,
        assets_yoy,
        eqt_yoy,
        tr_yoy,
        or_yoy,
        q_gr_yoy,
        q_gr_qoq,
        q_sales_yoy,
        q_sales_qoq,
        q_op_yoy,
        q_op_qoq,
        q_profit_yoy,
        q_profit_qoq,
        q_netprofit_yoy,
        q_netprofit_qoq,
        equity_yoy,
        rd_exp,
        update_flag'''
        kwargs['fields'] = fields
        return self.pro.fina_indicator_vip(*args, **kwargs)

    @segment_op(limit=5000, _max=100000)
    def income(self, *args, **kwargs):
        logger.info(f"obtaining income, args {args}, kwargs {kwargs}")
        return self.pro.income_vip(*args, **kwargs)

    @segment_op(limit=5000, _max=100000)
    def balancesheet(self, *args, **kwargs):
        logger.info(f"obtaining balancesheet, args {args}, kwargs {kwargs}")
        return self.pro.balancesheet_vip(*args, **kwargs)

    @segment_op(limit=5000, _max=100000)
    def cashflow(self, *args, **kwargs):
        logger.info(f"obtaining cashflow, args {args}, kwargs {kwargs}")
        return self.pro.cashflow_vip(*args, **kwargs)

    # ------------------------------------------------------------------------------------
    # 指数日行情
    def index_daily(self, table_name=None):
        logger.info("obtaining index_daily")
        index_list = ['000001.SH', '000300.SH', '000905.SH']
        tmp_dir = os.path.join(self.root, "__temp_index_daily__")
        os.umask(0)
        os.makedirs(tmp_dir, exist_ok=True)
        logger.info(f"dir {tmp_dir} created.")
        for i in index_list:
            logger.info(f"obtain index {i}")
            df = self.pro.index_daily(ts_code=i)
            self.close_file(df, file_name=f"__temp_index_daily__/{i}", table_name=table_name)
        logger.info("index_daily obtained.")

    def index_weight(self, table_name=None):
        logger.info("obtaining index_weight")
        index_list = ['000001.SH', '000300.SH', '000905.SH']
        tmp_dir = "__temp_index_weight__"
        for i in index_list:
            logger.info(f"obtain index weight {i}")
            last_month_end = self.data.get_last_month_end(self.data.start_dt)
            df = self.pro.index_weight(index_code=i, start_date=last_month_end, end_date=self.data.end_dt)
            self.close_file(df, file_name=f"{tmp_dir}/{i}", table_name=table_name)
        logger.info("index_daily obtained.")