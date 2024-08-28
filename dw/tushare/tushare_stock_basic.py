import sys, os

parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(parent_dir)

from utils.log_util import get_logger
logger = get_logger(__name__)

from utils.stock_zh_a_util import is_backfill

from data_fetcher import TushareFetcher
from datetime import datetime, timedelta

if __name__ == '__main__':
    ds = sys.argv[1]
    backfill = is_backfill(ds)
    logger.info("execute task on ds {}".format(ds))
    days_ahead = 3650 if backfill else 7
    start_date = (datetime.strptime(ds, '%Y%m%d') - timedelta(days=days_ahead)).strftime("%Y%m%d")
    fetcher = TushareFetcher(start_dt=start_date, end_dt=ds, ds=ds)

    # ---------------------------------------------------------------
    # 先下载数据到本地
    # ---------------------------------------------------------------
    fetcher.fetch_meta_data()
    fetcher.fetch_trade_day()
    fetcher.fetch_month_map()
    fetcher.fetch_index_member_all()
    # 日数据
    fetcher.ensure_data(fetcher.daily, "__temp_daily__", table_name="tushare_daily")  # 日行情表
    fetcher.ensure_data(fetcher.daily_basic, "__temp_daily_basic__", table_name="tushare_daily_basic")  # 每日指标表
    fetcher.ensure_data(fetcher.stk_factor, "__temp_stk_factor__", table_name="tushare_stk_factor")  # 每日技术面因子
    fetcher.ensure_data(fetcher.suspend_d, "__temp_suspend_d__", table_name="tushare_suspend_d")  # 停牌表
    fetcher.ensure_data(fetcher.limit_list, "__temp_limit_list__", table_name="tushare_limit_list")  # 涨跌停表
    fetcher.ensure_data(fetcher.adj_factor, "__temp_adj_factor__", table_name="tushare_adj_factor")  # 复权因子表
    fetcher.ensure_data(fetcher.moneyflow, "__temp_moneyflow__", table_name="tushare_moneyflow")  # 资金流表

    # 季度数据
    fetcher.ensure_data_by_q(fetcher.fina_indicator, "__temp_fina_indicator__", table_name="tushare_fina_indicator")  # 财务指标表
    fetcher.ensure_data_by_q(fetcher.income, "__temp_income__", table_name="tushare_income")  # 利润表
    fetcher.ensure_data_by_q(fetcher.balancesheet, "__temp_balancesheet__", table_name="tushare_balancesheet")  # 资产负债表
    fetcher.ensure_data_by_q(fetcher.cashflow, "__temp_cashflow__", table_name="tushare_cashflow")  # 现金流表
    #
    # 日数据
    fetcher.index_daily(table_name="tushare_index_daily")
    # 月度数据
    fetcher.index_weight(table_name="tushare_index_weight")
