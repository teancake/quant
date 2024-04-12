import sys, os

parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(parent_dir)

from utils.log_util import get_logger
logger = get_logger(__name__)

from utils.stock_zh_a_util import is_backfill

from raw_data_fetch import TushareFetcher
from datetime import datetime, timedelta

if __name__ == '__main__':
    ds = sys.argv[1]
    backfill = is_backfill(ds)
    logger.info("execute task on ds {}".format(ds))
    days_ahead = 756 if backfill else 7
    start_date = (datetime.strptime(ds, '%Y%m%d') - timedelta(days=days_ahead)).strftime("%Y%m%d")
    fetcher = TushareFetcher(start_dt=start_date, end_dt=ds)
    fetcher.set_ds(ds)
    fetcher.set_to_sql(True)
    # ---------------------------------------------------------------
    # 先下载数据到本地
    # ---------------------------------------------------------------
    fetcher.fetch_meta_data()
    fetcher.fetch_trade_day()
    fetcher.fetch_month_map()
    # 日数据
    fetcher.ensure_data(fetcher.daily, "__temp_daily__", table_name="tushare_daily")  # 日行情表
    fetcher.ensure_data(fetcher.suspend_d, "__temp_suspend_d__", table_name="tushare_suspend_d")  # 停牌表
    fetcher.ensure_data(fetcher.limit_list, "__temp_limit_list__", table_name="tushare_limit_list")  # 涨跌停表
    fetcher.ensure_data(fetcher.adj_factor, "__temp_adj_factor__", table_name="tushare_adj_factor")  # 复权因子表
    fetcher.ensure_data(fetcher.daily_basic, "__temp_daily_basic__", table_name="tushare_daily_basic")  # 每日指标表
    fetcher.ensure_data(fetcher.moneyflow, "__temp_moneyflow__", table_name="tushare_moneyflow")  # 资金流表
    fetcher.ensure_data(fetcher.stk_factor, "__temp_stk_factor__", table_name="tushare_stk_factor")  # 每日技术面因子

    # 季度数据
    fetcher.ensure_data_by_q(fetcher.fina_indicator, "__temp_fina_indicator__", table_name="tushare_fina_indicator")  # 财务指标表
    fetcher.ensure_data_by_q(fetcher.income, "__temp_income__", table_name="tushare_income")  # 利润表
    fetcher.ensure_data_by_q(fetcher.balancesheet, "__temp_balancesheet__", table_name="tushare_balancesheet")  # 资产负债表
    fetcher.ensure_data_by_q(fetcher.cashflow, "__temp_cashflow__", table_name="tushare_cashflow")  # 现金流表
    fetcher.index_daily(table_name="tushare_index_daily")
    fetcher.index_weight(table_name="tushare_index_weight")

    # ---------------------------------------------------------------
    # # 然后从本地数据生成指标
    # # ---------------------------------------------------------------
    # if generate:
    #     fetcher.create_index_wt()
    #     fetcher.create_listday_matrix()
    #     fetcher.create_month_tdays_begin_end()
    #     fetcher.create_trade_status()
    #     fetcher.create_turn_d()
    #     fetcher.create_maxupordown()
    #     fetcher.create_indicator("__temp_adj_factor__", "adj_factor", "adjfactor")
    #     fetcher.create_industry_citic()
    #
    #     fetcher.create_mkt_cap_float_m()
    #     fetcher.create_pe_ttm_m()
    #     fetcher.create_val_pe_deducted_ttm_m()
    #     fetcher.create_pb_lf_m()
    #     fetcher.create_ps_ttm_m()
    #     fetcher.create_pcf_ncf_ttm_m()
    #     fetcher.create_pcf_ocf_ttm_m()
    #     fetcher.create_dividendyield2_m()
    #     fetcher.create_profit_ttm_G_m()
    #     fetcher.create_indicator_m_by_q("__temp_fina_indicator__", "q_sales_yoy", "qfa_yoysales_m")
    #     fetcher.create_indicator_m_by_q("__temp_fina_indicator__", "q_profit_yoy", "qfa_yoyprofit_m")
    #     fetcher.create_indicator_m_by_q("__temp_fina_indicator__", "ocf_yoy", "qfa_yoyocf_m")  # 临时替代
    #     fetcher.create_indicator_m_by_q("__temp_fina_indicator__", "roe_yoy", "qfa_roe_G_m")  # 临时替代
    #     fetcher.create_indicator_m_by_q("__temp_fina_indicator__", "q_roe", "qfa_roe_m")
    #     fetcher.create_indicator_m_by_q("__temp_fina_indicator__", "roe_yearly", "roe_ttm2_m")  # 临时替代
    #     fetcher.create_indicator_m_by_q("__temp_fina_indicator__", "roa", "qfa_roa_m")  # 临时替代
    #     fetcher.create_indicator_m_by_q("__temp_fina_indicator__", "roa_yearly", "roa2_ttm2_m")  # 临时替代
    #     fetcher.create_indicator_m_by_q("__temp_fina_indicator__", "q_gsprofit_margin", "qfa_grossprofitmargin_m")
    #     fetcher.create_indicator_m_by_q("__temp_fina_indicator__", "grossprofit_margin",
    #                                     "grossprofitmargin_ttm2_m")  # 临时替代
    #     fetcher.create_indicator_m_by_q("__temp_fina_indicator__", "assets_turn", "turnover_ttm_m")  # 临时替代
    #     fetcher.create_indicator_m_by_q("__temp_fina_indicator__", "assets_to_eqt", "assetstoequity_m")
    #     fetcher.create_indicator_m_by_q("__temp_fina_indicator__", "debt_to_eqt", "longdebttoequity_m")  # 临时替代
    #     fetcher.create_indicator_m_by_q("__temp_fina_indicator__", "cash_to_liqdebt", "cashtocurrentdebt_m")
    #     fetcher.create_indicator_m_by_q("__temp_fina_indicator__", "current_ratio", "current_m")
    #     fetcher.create_daily_quote_indicators()
    #     fetcher.create_indicator("__temp_daily_basic__", "circ_mv", "mkt_cap_float")
    #     fetcher.create_indicator("__temp_daily_basic__", "total_mv", "mkt_cap_ard")
    #     fetcher.create_indicator_m_by_q("__temp_fina_indicator__", "longdeb_to_debt", "longdebttodebt_lyr_m")
    #     fetcher.create_indicator_m_by_q("__temp_balancesheet__", "total_liab", "tot_liab_lyr_m")
    #     fetcher.create_indicator_m_by_q("__temp_balancesheet__", "oth_eqt_tools_p_shr",
    #                                     "other_equity_instruments_PRE_lyr_m")
    #     fetcher.create_indicator_m_by_q("__temp_balancesheet__", "total_hldr_eqy_inc_min_int", "tot_equity_lyr_m")
    #     fetcher.create_indicator_m_by_q("__temp_balancesheet__", "total_assets", "tot_assets_lyr_m")
    #
