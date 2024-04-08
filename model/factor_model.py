import sys, os
parent_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parent_dir)


import re
import pandas as pd
import numpy as np

from datetime import datetime, timedelta

from utils.log_util import get_logger
logger = get_logger(__name__)

from utils.starrocks_db_util import StarrocksDbUtil
from utils.stock_zh_a_util import get_stock_data, get_stock_list, get_benchmark_data, get_list_date
import statsmodels.api as sm



# adapted from https://github.com/Young-Mann/Fama-5-Factors/blob/main/FAMA.ipynb

class BaseFactorModel:

    def __init__(self, ds, start_date):
        self.ds = ds
        self.start_date = start_date
        self.period = "daily"

    def get_benchmark(self):
        symbol = "sh000300"
        df = get_benchmark_data(symbol, ds=self.ds, start_date=self.start_date, yf_compatible=True)
        df["pct_chg"] = df["close"].pct_change()
        df.dropna(inplace=True)
        return df[["close", "pct_chg"]]

    def get_return(self):
        df = get_stock_data("all", ds=self.ds, start_date=self.start_date, yf_compatible=True)
        levels = [df.index, df.symbol]
        df.index = pd.MultiIndex.from_arrays(levels, names=["date", "ticker"])
        df = df.sort_index(level=["date", "ticker"], ascending=[True, True])
        df["pct_chg"] = df["close"].groupby(level="ticker").pct_change()
        df.dropna(inplace=True)
        return df[["close", "pct_chg"]]

    def get_basic_data(self):
        # tushare daily_basic 接口返回的close收盘价是不复权的，不使用它计算pct_chg，使用历史行情
        sql = f"select * from dwd_tushare_daily_basic_df where ds='{self.ds}' and trade_date > '{self.start_date}';"
        results = StarrocksDbUtil().run_sql(sql)
        df = pd.DataFrame(results)
        df.set_index("trade_date", inplace=True)
        df.index = pd.to_datetime(df.index)
        df.index.name = "date"
        logger.info(f"basic data columns {df.columns}")
        df["symbol"] = df["ts_code"].apply(lambda x: re.sub(r"[^0-9]", "", x))
        levels = [df.index, df.symbol]
        df.index = pd.MultiIndex.from_arrays(levels, names=["date", "ticker"])
        df["bm"] = 1 / df["pb"]
        df.drop(columns=["close", "ds"], inplace=True)
        return df

    def get_rf(self):
        # 应该用3个月
        sql = f"select * from dwd_rate_interbank_df where ds= '{self.ds}' and symbol='Shibor人民币' and indicator='1周'"
        results = StarrocksDbUtil().run_sql(sql)
        df = pd.DataFrame(results)
        df.set_index("日期", inplace=True)
        df.index = pd.to_datetime(df.index)
        df.index.name = "date"
        df = df.sort_index(level=["date"], ascending=[True])

        # r_f = df_shibor.resample('M', on='trade_date').last()
        df["rf"] = df["利率"] / 100
        return df[["rf"]]


class CapmModel(BaseFactorModel):

    def __init__(self, ds, start_date):
        super().__init__(ds, start_date)
        symbols = self.get_stock_position()
        self.RISKY_ASSET = symbols[0]
        self.MARKET_BENCHMARK = "sh000300"

    def get_stock_position(self):
        results = StarrocksDbUtil().run_sql(f"""select 代码 from ads_stock_zh_a_position where position > 0""")
        results = [item[0] for item in results]
        stock_list = list(set(get_stock_list()) & set(results))
        return stock_list



    def adapt_stock_data(self):
        df_list = []
        for stock in [self.RISKY_ASSET]:
            df = get_stock_data(stock, ds=self.ds, start_date=self.start_date, yf_compatible=True)
            levels = [df.index, [stock] * len(df)]
            df.index = pd.MultiIndex.from_arrays(levels, names=["date", "ticker"])
            df_list.append(df)
        data = pd.concat(df_list)
        data = data.unstack()
        return data

    def adapt_benchmark_data(self):
        symbol = self.MARKET_BENCHMARK
        df = get_benchmark_data(symbol="sh000300", ds=self.ds, start_date=self.start_date, yf_compatible=True)
        levels = [df.index, [symbol] * len(df)]
        df.index = pd.MultiIndex.from_arrays(levels, names=["date", "benchmark"])
        data = df.unstack()
        return data

    def get_model(self):
        stock_data = self.adapt_stock_data()
        market_data = self.adapt_benchmark_data()
        data = stock_data.join(market_data, how='inner')
        print(data)
        X = (
            data["close"]
                .rename(columns={self.RISKY_ASSET: "asset",
                                 self.MARKET_BENCHMARK: "market"})
                .resample("M")
                .last()  # Resample the DataFrame to a monthly frequency and get the last value within each week
                .pct_change()
                .dropna()
        )
        covariance = X.cov().iloc[0, 1]
        benchmark_variance = X.market.var()
        beta = covariance / benchmark_variance
        print(f"beta {beta}")

        # separate target
        y = X.pop("asset")

        # add constant
        X = sm.add_constant(X)
        logger.info(f"before OLS, X {X}, y {y}")
        # define and fit the regression model
        capm_model = sm.OLS(y, X).fit()

        # print results
        print(capm_model.summary())
        return capm_model


class FamaFrenchThree(BaseFactorModel):
    def __init__(self, ds, start_date):
        super().__init__(ds, start_date)

    # 因子分组
    def add_group_columns(self, df, val_col, group_col, group_cnt):
        if group_cnt == 3:
            df[group_col] = 2
            df.loc[df.groupby("date", group_keys=False)[val_col].apply(lambda x: x > x.quantile(0.7)), group_col] = 3
            df.loc[df.groupby("date", group_keys=False)[val_col].apply(lambda x: x <= x.quantile(0.3)), group_col] = 1
        else:
            df[group_col] = 1
            df.loc[df.groupby("date", group_keys=False)[val_col].apply(lambda x: x > x.quantile(0.5)), group_col] = 2
        df[group_col] = df[group_col].apply(lambda x: group_col + str(int(x)))
        return df

    def get_df_rf(self):
        data = self.get_basic_data()
        ret = self.get_return()
        df = pd.merge(data, ret,  left_index=True, right_index=True)
        df.rename(columns={"pct_chg": "RETURN", "circ_mv": "MKT", "bm": "BM"}, inplace=True)
        logger.info(f"after merge, df columns {df.columns}, value {df}")
        # 缩尾，剔除极值
        df["RETURN"] = self.winsorize(df["RETURN"])
        df = self.delete_smallsize(df)
        df = self.delete_new_listed(df)
        df = self.handle_nan(df)
        rf = self.get_rf()
        return df, rf

    def handle_nan(self, df):
        logger.info("processing NAN values")
        df = df.groupby('ticker', group_keys=False).apply(lambda x: x.sort_values('date').ffill(axis=0))
        df = df.dropna(subset=["RETURN", "MKT", "BM"])
        return df

    def delete_new_listed(self, df):
        logger.info(f"delete new listed company, before df length {len(df)}")
        list_date = get_list_date()
        df = pd.merge(df, list_date, left_index=True, right_index=True)
        df['days_dif'] = (df.index.get_level_values(level="date") - df['list_date'])
        df = df[df['days_dif'].apply(lambda x: x.days > 180)]
        logger.info(f"delete new listed company, after df length {len(df)}")
        return df

    def delete_smallsize(self, df, q=0.3):
        logger.info(f"delete small size company, before df length {len(df)}")
        df["is_small_size"] = df.groupby("date", group_keys=False)["MKT"].apply(lambda x: x < x.quantile(0.3))
        df = df[df['is_small_size'] == False]
        return df

    def winsorize(self, s):
        up_q = s.quantile(0.99)
        low_q = s.quantile(0.01)
        s = np.where(s > up_q, up_q, s)
        s = np.where(s < low_q, low_q, s)
        upper = s.mean() + s.std() * 3
        low = s.mean() - s.std() * 3
        s = np.where(s > upper, upper, s)
        s = np.where(s < low, low, s)
        return s

    def get_model(self,df, rf):
        df_factor = self.get_ff3_factors(df)

        import statsmodels.api as sm
        temp_df = pd.merge(pd.DataFrame(df.groupby("date")["RETURN"].mean()), rf["rf"], left_index=True, right_index=True)
        lhs = pd.DataFrame(temp_df["RETURN"] - temp_df["rf"], columns=["lhs"])
        X = pd.merge(df_factor, lhs, left_index=True, right_index=True)
        X.dropna(inplace=True)
        logger.info(f"ols data {X}")
        y = X.pop("lhs")
        X = sm.add_constant(X)
        logger.info(f"before OLS, X {X}, y {y}")
        result = sm.OLS(y, X).fit()
        logger.info(result.summary())
        return result


    def get_ff3_factors(self, df):
        size, size_group_col, size_group_cnt = "MKT", "MKT_G", 2
        value, value_group_col, value_group_cnt = "BM", "BM_G", 3
        df = self.add_group_columns(df, val_col=size, group_col=size_group_col, group_cnt=size_group_cnt)
        df = self.add_group_columns(df, val_col=value, group_col=value_group_col, group_cnt=value_group_cnt)

        # logger.info(f"group market and size {df[(df.index.get_level_values(level='date')=='2024-04-02')&(df['MKT_G']=='MKT_G1') & (df['BM_G']=='BM_G1')].index.get_level_values(level='ticker').drop_duplicates().sort_values().tolist()}")
        # logger.info(f"group market {df[(df.index.get_level_values(level='date')=='2024-04-02')&(df['MKT_G']=='MKT_G1')].index.get_level_values(level='ticker').drop_duplicates().sort_values().tolist()}")
        # logger.info(f"group size {df[(df.index.get_level_values(level='date')=='2024-04-02')&(df['BM_G']=='BM_G1')].index.get_level_values(level='ticker').drop_duplicates().sort_values().tolist()}")

        ret_type = "mkt"
        factor_lst = []
        # for double_label, label in zip(['value_double'], [value_label]):
        tmp_col = "tmp_col"
        df[tmp_col] = df[[size_group_col, value_group_col]].apply(lambda x: x[size_group_col] + '/' + x[value_group_col], axis=1)
        if ret_type == 'mkt':
            ret_series = df.groupby(["date", tmp_col]).apply(lambda x: (x['RETURN'] * x[size] / x[size].sum()).sum()).unstack(tmp_col)
        elif ret_type == 'equal':
            ret_series = df.groupby(["date", tmp_col]).apply(lambda x: x['RETURN'].mean()).unstack(tmp_col)
        else:
            raise ValueError("Parameter 'ret_type' is not in ['mkt', 'equal']")
        factor_lst.append(ret_series)
        factor_ret = pd.concat(factor_lst, axis=1)

        # 计算价值因子
        factor_ret['HML'] = (factor_ret['MKT_G1/BM_G3'] + factor_ret['MKT_G2/BM_G3']) / 2 - (factor_ret['MKT_G1/BM_G1'] + factor_ret['MKT_G2/BM_G1']) / 2
        # 计算规模因子
        factor_ret['SMB'] = (factor_ret['MKT_G1/BM_G1'] + factor_ret['MKT_G1/BM_G2'] + factor_ret['MKT_G1/BM_G3']) / 3 - \
                            (factor_ret['MKT_G2/BM_G1'] + factor_ret['MKT_G2/BM_G2'] + factor_ret['MKT_G2/BM_G3']) / 3

        # 使用指数数据做为市场因子
        benchmark = self.get_benchmark()
        benchmark = benchmark[["pct_chg"]].rename(columns={"pct_chg": "MARKET"})
        factor_ret = pd.merge(benchmark, factor_ret, how="inner", on="date")
        logger.info(f"FF3 factor computation done, factor columns {factor_ret.columns}, value {factor_ret}")
        return factor_ret


    def mark_FF_set(self, df):
        tmp_df = df
        tmp_df['MKT_G'] = ((tmp_df['MKT'] > tmp_df['MKT'].quantile(0.5)).astype(int)).replace({1: 'B', 0: 'S'})
        tmp_df['BM_G'] = ((tmp_df['BM'] > tmp_df['BM'].quantile(0.7)).astype(int) + (tmp_df['BM'] > tmp_df['BM'].quantile(0.3)).astype(int)).replace({2: 'H', 1: 'M', 0: 'L'})
        return tmp_df

    def get_ff3_factors_alt(self, df, rf):

        # time_list = list(df.index.get_level_values(level="date").drop_duplicates())
        # change_d = 1  # 换仓间隔
        # changeing_lst = time_list[0:len(time_list):change_d]  # 换仓时点
        # changeing_df = df[df.index.get_level_values(level="date").isin(changeing_lst)]
        # changeing_df = changeing_df.groupby("date", group_keys=False).apply(self.mark_FF_set)
        # df = pd.merge(df, changeing_df[['MKT_G', 'BM_G']], how='left', on=['date', 'ticker'])

        df = df.groupby("date", group_keys=False).apply(self.mark_FF_set)
        df['class'] = df['MKT_G'] + df['BM_G']
        logger.info(f"group size value {df[(df.index.get_level_values(level='date')=='2024-04-02')&(df['MKT_G']=='S') & (df['BM_G']=='L')].index.get_level_values(level='ticker').drop_duplicates().sort_values().tolist()}")
        logger.info(f"group size {df[(df.index.get_level_values(level='date')=='2024-04-02')&(df['MKT_G']=='S')].index.get_level_values(level='ticker').drop_duplicates().sort_values().tolist()}")
        logger.info(f"group value {df[(df.index.get_level_values(level='date')=='2024-04-02')&(df['BM_G']=='L')].index.get_level_values(level='ticker').drop_duplicates().sort_values().tolist()}")

        # （1）市值加权法：
        returns = df[['class', 'RETURN', 'MKT']].groupby(['date', 'class']) \
            .apply(lambda x: (x['RETURN'] * x['MKT'] / x['MKT'].sum()).sum()).unstack()
        # （2）也可以采用等权平均法：
        # returns=df[['时间','class', '涨跌幅(%)','总市值']].groupby(['时间','class']).mean()['涨跌幅(%)'].unstack()

        returns['SMB'] = (returns['SH'] + returns['SM'] + returns['SL']) / 3 - (returns['BH'] + returns['BM'] + returns['BL']) / 3
        returns['HML'] = (returns['SH'] + returns['BH']) / 2 - (returns['SL'] + returns['BL']) / 2


        # 合并上证指数，算出因子净值：
        benchmark = self.get_benchmark()
        benchmark = benchmark[["pct_chg"]].rename(columns={"pct_chg": "MARKET"})
        returns = pd.merge(benchmark, returns, how="inner", on="date")

        # （1）计算每日收益乘数R=1+r
        returns['SMB-R'] = 1 + returns['SMB'] / 100
        returns['HML-R'] = 1 + returns['HML'] / 100
        returns['MKT-R'] = 1 + returns['MARKET'] / 100
        # （2）计算因子净值net value
        smb_nv = [1]
        hml_nv = [1]
        mkt_nv = [1]
        for i in returns.index:
            smb_nv.append(smb_nv[-1] * returns.loc[i, 'SMB-R'])
            hml_nv.append(hml_nv[-1] * returns.loc[i, 'HML-R'])
            mkt_nv.append(mkt_nv[-1] * returns.loc[i, 'MKT-R'])
        returns['SMB-NV'] = smb_nv[1:]
        returns['HML-NV'] = hml_nv[1:]
        returns['MKT-NV'] = mkt_nv[1:]
        logger.info(f"FF3 factor computation done, factor columns {returns.columns}, value {returns}")
        return returns

