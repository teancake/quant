# -*- coding: utf-8 -*-
"""
阿尔法收割者

Project: alphasickle
Author: Moses
E-mail: 8342537@qq.com
"""

# 数据预处理及截面回归的过程参考 https://qsdoc.readthedocs.io/zh-cn/latest/%E9%A3%8E%E9%99%A9%E6%A8%A1%E5%9E%8B.html
# QuantStudio的文档


import os
import numpy as np
import pandas as pd
from datetime import datetime



from utils.log_util import get_logger
logger = get_logger(__name__)
from dw.tushare.data_fetcher import DataAccessor

from dw.tushare.barra_cne6_factor import DividendYield, EarningsQuality, EarningsVariablity, EarningsYield, Growth, \
    InvestmentQuality, Leverage, Liquidity, LongTermReversal, Momentum, Profitability, Size, Value, Volatility


class BarraCne6DataAdaptor(DataAccessor):

    def __init__(self, ds, start_dt, end_dt):
        super().__init__(ds, start_dt, end_dt)

    def create_indicator_from_db(self, raw_data_dir: str, raw_data_field, indicator_name):
        table_name_map = {"__temp_fina_indicator__": "dwd_tushare_fina_indicator_df",
                          "__temp_daily_basic__": "dwd_tushare_daily_basic_df",
                          "__temp_balancesheet__": "dwd_tushare_balancesheet_df",
                          "__temp_cashflow__": "dwd_tushare_cashflow_df",
                          "__temp_income__": "dwd_tushare_income_df",
                          "__temp_daily__": "dwd_tushare_daily_df",
                          "__temp_index_daily__":"dwd_tushare_index_daily_df",
                          "__temp_adj_factor__": "dwd_tushare_adj_factor_df",
                          }
        index_columns_map = {"__temp_fina_indicator__": ["ts_code", "end_date"],
                             "__temp_daily_basic__": ["ts_code", "trade_date"],
                             "__temp_balancesheet__": ["ts_code", "end_date"],
                             "__temp_cashflow__": ["ts_code", "end_date"],
                             "__temp_income__": ["ts_code", "end_date"],
                             "__temp_daily__": ["ts_code", "trade_date"],
                             "__temp_index_daily__": ["ts_code", "trade_date"],
                             "__temp_adj_factor__": ["ts_code", "trade_date"],

                             }
        date_columns_map = {"__temp_fina_indicator__": ["end_date"],
                            "__temp_daily_basic__": ["trade_date"],
                            "__temp_balancesheet__": ["end_date"],
                            "__temp_cashflow__": ["end_date"],
                            "__temp_income__": ["end_date"],
                            "__temp_daily__": ["trade_date"],
                            "__temp_index_daily__": ["trade_date"],
                            "__temp_adj_factor__": ["trade_date"],
                            }
        table_name = table_name_map.get(raw_data_dir)
        index_columns = index_columns_map.get(raw_data_dir)
        date_columns = date_columns_map.get(raw_data_dir)
        fields = set(index_columns + date_columns + [raw_data_field])
        if raw_data_dir != "__temp_index_daily__":
            sql = f"select {','.join(fields)} from {table_name} where ds='{self.ds}' and ts_code in ('603127.SH', '601288.SH', '601006.SH', '600221.SH', '603206.SH')"
        else:
            sql = f"select {','.join(fields)} from {table_name} where ds='{self.ds}'"

        df = self.db.df_from_sql(sql, index_columns, date_columns)
        df = df.unstack(level=0)
        df.columns = df.columns.levels[1]
        self.close_file(df, indicator_name)


    def _align_element(self, df1, df2):
        ''' 对齐股票和时间
        '''
        row_index = sorted(df1.index.intersection(df2.index))
        col_index = sorted(df1.columns.intersection(df2.columns))
        return df1.loc[row_index, col_index], df2.loc[row_index, col_index]

    def create_all_stocks_code(self):
        logger.info("create_all_stocks_code")
        all_stocks_code = self.meta
        all_stocks_code["wind_code"] = all_stocks_code["symbol"]
        self.close_file(all_stocks_code, "all_stocks_code")

    def create_hfq_close(self):
        self.create_indicator_from_db("__temp_daily__", "close", "close")
        self.create_indicator_from_db("__temp_adj_factor__", "adj_factor", "adjfactor")
        logger.info("adjusting close price")
        close, adjfactor = self._align_element(self.close, self.adjfactor)
        hfq_close = close * adjfactor
        self.close_file(hfq_close, 'hfq_close')  # 后复权收盘价

    def create_firstind(self):
        # assume no change of industry since ipo, because could not get historical industry data.
        logger.info("create_firstind")
        import itertools
        ind = self.meta[["industry"]]
        ind = ind[ind.index.isin(['603127.SH', '601288.SH', '601006.SH', '600221.SH', '603206.SH'])]
        tuples = list(itertools.product(self.tradedays, ind.index))
        multi_index = pd.MultiIndex.from_tuples(tuples, names=('date', 'code'))
        df = pd.DataFrame(index=multi_index)
        df = pd.merge(df, ind, left_index=True, right_index=True)
        df = df.unstack(level="code")
        df.columns = df.columns.droplevel()
        self.close_file(df, "firstind")

        # sql = f"""
        #     SELECT tradedays, code, industry FROM
        #         (SELECT code, industry FROM dwd_tushare_stock_basic_df WHERE ds="{self.ds}")a
        #     JOIN
        #         (SELECT tradedays FROM dwd_tushare_trade_cal_df WHERE ds="{self.ds}")b
        #     """
        # df = self.db.df_from_sql(sql, index_columns=["tradedays", "code"], date_columns=["tradedays"])
        # df = df.unstack(level="code")
        # df.columns = df.columns.levels[1]


class BarraCne6FactorGenerater:
    def __init__(self, ds, start_dt, end_dt):
        self.adaptor = BarraCne6DataAdaptor(ds, start_dt, end_dt)

    def generate_base_indicators(self):

        self.adaptor.create_all_stocks_code()
        # 一级行业
        self.adaptor.create_firstind()

        self.adaptor.create_hfq_close()
        self.adaptor.create_indicator_from_db("__temp_daily_basic__", "circ_mv", "negotiablemv")
        self.adaptor.create_indicator_from_db("__temp_daily_basic__", "total_mv", "totalmv")
        self.adaptor.create_indicator_from_db("__temp_daily_basic__", "turnover_rate", "turnoverrate")
        self.adaptor.create_indicator_from_db("__temp_daily__", "pct_chg", "changepct")
        self.adaptor.create_indicator_from_db("__temp_daily__", "amount", "turnovervalue")

        self.adaptor.create_indicator_from_db("__temp_index_daily__", "pct_chg", "indexquote_changepct")

        self.adaptor.create_indicator_from_db("__temp_balancesheet__", "total_ncl", "totalnoncurrentliability")
        self.adaptor.create_indicator_from_db("__temp_balancesheet__", "oth_eqt_tools_p_shr", "preferedequity")
        self.adaptor.create_indicator_from_db("__temp_balancesheet__", "total_hldr_eqy_inc_min_int", "totalshareholderequity")
        self.adaptor.create_indicator_from_db("__temp_balancesheet__", "total_assets", "totalassets")
        self.adaptor.create_indicator_from_db("__temp_balancesheet__", "total_liab", "totalliability")
        self.adaptor.create_indicator_from_db("__temp_balancesheet__", "money_cap", "cashequialents")
        self.adaptor.create_indicator_from_db("__temp_balancesheet__", "total_hldr_eqy_exc_min_int", "sewithoutmi")


        self.adaptor.create_indicator_from_db("__temp_fina_indicator__", "eqt_to_interestdebt", "sewmitointerestbeardebt")
        self.adaptor.create_indicator_from_db("__temp_fina_indicator__", "eqt_to_debt", "sewithoutmitotl")
        self.adaptor.create_indicator_from_db("__temp_fina_indicator__", "ebit", "ebit")
        self.adaptor.create_indicator_from_db("__temp_fina_indicator__", "eps", "eps")
        self.adaptor.create_indicator_from_db("__temp_fina_indicator__", "capitalized_to_da", "capitalexpendituretodm")

        self.adaptor.create_indicator_from_db("__temp_income__", "n_income", "netprofit")
        self.adaptor.create_indicator_from_db("__temp_income__", "revenue", "operatingreenue")
        self.adaptor.create_indicator_from_db("__temp_income__", "total_cogs", "cogs_q")

        self.adaptor.create_indicator_from_db("__temp_cashflow__", "n_incr_cash_cash_equ", "cashequialentincrease")
        self.adaptor.create_indicator_from_db("__temp_cashflow__", "c_pay_acq_const_fiolta", "capital_expenditure")
        self.adaptor.create_indicator_from_db("__temp_cashflow__", "n_cashflow_act", "netoperatecashflow")
        self.adaptor.create_indicator_from_db("__temp_cashflow__", "n_cashflow_inv_act", "netinvestcashflow")


    def generate_barra_cne6_factors(self):
        logger.info(Size().LNCAP)
        logger.info(Size().MIDCAP)
        logger.info(Volatility().BETA)
        logger.info(Volatility().CMRA)
        logger.info(Volatility().DASTD)
        logger.info(Volatility().HALPHA)
        logger.info(Volatility().HSIGMA)
        logger.info(Liquidity().ATVR)
        logger.info(Liquidity().STOA)
        logger.info(Liquidity().STOM)
        logger.info(Liquidity().STOQ)
        logger.info(Momentum().INDMOM)
        logger.info(Momentum().RSTR)
        logger.info(Momentum().SEASON)
        logger.info(Momentum().STREV)
        logger.info(Leverage().BLEV)
        logger.info(Leverage().DTOA)
        logger.info(Leverage().MLEV)

        logger.info(EarningsVariablity().VERN)
        logger.info(EarningsVariablity().VFLO)
        logger.info(EarningsVariablity().VSAL)

        logger.info(EarningsQuality().ABS)
        logger.info(EarningsQuality().ACF)

        logger.info(Profitability().ATO)
        logger.info(Profitability().GP)
        logger.info(Profitability().GPM)
        logger.info(Profitability().ROA)

        # 因为 applied_lyr_date_d 木有搞定
        # InvestmentQuality().AGRO
        # InvestmentQuality().CXGRO
        # InvestmentQuality().IGRO

        logger.info(Value().BTOP)
        logger.info(EarningsYield().BTOP)
        logger.info(EarningsYield().CETOP)
        logger.info(EarningsYield().EM)
        logger.info(EarningsYield().ETOP)

        logger.info(LongTermReversal().BTOP)

        # 窗口太长，报错
        # logger.info(LongTermReversal().LTHALPHA)
        # logger.info(LongTermReversal().LTRSTR)

        # 因为 applied_lyr_date_d 木有搞定
        # Growth().EGRO
        # Growth().SGRO

        # dividendps 数据不知道从哪里拿，搞不定
        # DividendYield().DTOP


if __name__ == '__main__':
    fg = BarraCne6FactorGenerater(ds="20240528", start_dt="20240501", end_dt="20240523")
    fg.generate_base_indicators()
    fg.generate_barra_cne6_factors()
