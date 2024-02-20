import sys, os
parent_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parent_dir)


from utils.log_util import get_logger
from utils.starrocks_db_util import StarrocksDbUtil, generate_partition_spec
from utils import stock_zh_a_util, ta_util
import sys
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from utils.stock_zh_a_util import is_trade_date, is_backfill

logger = get_logger(__name__)


def compute_technical_indicators(db_util, symbol, ds, backfill):
    logger.info("compute technical indicators for symbol {} ".format(symbol))
    sql = "select * from dwd_stock_zh_a_hist_df where ds = '{}' and 代码='{}' and adjust='hfq' and period='daily' order by 日期 asc;".format(ds, symbol)
    data = db_util.run_sql(sql)
    data_df = pd.DataFrame(data)
    logger.info("data obtained")
    if len(data_df) == 0:
        logger.info("empty dataset, no computation required.")
        return
    ti_df = ta_util.get_ta_indicator_map(data_df)
    logger.info("indicators computed.")

    if backfill:
        days_ahead = 365 * 3
        logger.info("do backfill, save indicators of last {} days. This will take some time.".format(days_ahead))
    else:
        days_ahead = 7
        logger.info("no backfill, only save indicators of last {} days".format(days_ahead))
    start_date = (datetime.strptime(ds, '%Y%m%d') - timedelta(days=days_ahead)).date()
    ti_df = ti_df[ti_df.日期 >= start_date]
    ti_df = ti_df.replace([np.inf, -np.inf], np.nan)
    ti_df["ds"] = ds
    ti_df["gmt_create"] = datetime.now()
    ti_df["gmt_modified"] = datetime.now()
    ti_df["代码"] = symbol
    ti_df.to_sql(name="ods_stock_zh_a_ti_hist", con=db_util.get_db_engine(), if_exists='append', index=False, method='multi', chunksize=2000)
    logger.info("indicators saved to db.")


if __name__ == '__main__':
    ds = sys.argv[1]

    logger.info("execute task on ds {}".format(ds))
    if not is_trade_date(ds):
        logger.info(f"{ds} is not trade date. task exits.")
        exit(os.EX_OK)

    backfill = is_backfill(ds)
    logger.info("execute task on ds {}, backfill {}".format(ds, backfill))

    partition_str = generate_partition_spec(ds)

    db = StarrocksDbUtil()
    ods_sql_str = '''
create table if not exists ods_stock_zh_a_ti_hist
(
`ds` date,
`gmt_create` datetime,
`gmt_modified` datetime,
`日期` date,
`代码` varchar(20),
`ma_5` double,
`ma_10` double,
`ma_20` double,
`ma_60` double,
`ma_120` double,
`ma_240` double,
`hhv_5` double,
`hhv_10` double,
`hhv_20` double,
`hhv_60` double,
`hhv_120` double,
`hhv_240` double,
`llv_5` double,
`llv_10` double,
`llv_20` double,
`llv_60` double,
`llv_120` double,
`llv_240` double,
`bias_6` double,
`bias_12` double,
`bias_24` double,
`boll_upper_20` double,
`boll_mid_20` double,
`boll_lower_20` double,
`rsi_6` double,
`rsi_12` double,
`rsi_24` double,
`wr_10` double,
`wr_6` double,
`mtm_12` double,
`mtm_12_ma_6` double,
`k_9` double,
`d_9` double,
`j_9` double,
`macd_dif` double,
`macd_dea` double,
`macd` double,
`dmi_pdi` double,
`dmi_mdi` double,
`dmi_adx` double,
`dmi_adxr` double,
`obv` double,
`cci` double,
`roc_12` double,
`ma_6_roc_12` double,
`bbi` double,
`expma_12` double,
`expma_50` double,
`ar` double,
`br` double,
`atr` double,
`dma_dif` double,
`dma` double,
`emv` double,
`maemv` double,
`psy` double,
`psyma` double,
`asi` double,
`asit` double,
`mfi` double,
`mass` double,
`mamass` double,
`dpo` double,
`madpo` double,
`vr` double,
`trix` double,
`trma` double,
`kc_upper` double,
`kc_mid` double,
`kc_lower` double,
`dc_upper` double,
`dc_mid` double,
`dc_lower` double
)
PARTITION BY RANGE(ds)(
{}
)
DISTRIBUTED BY HASH(ds) BUCKETS 32
PROPERTIES(
    "replication_num" = "1",
    "dynamic_partition.enable" = "true",
    "dynamic_partition.time_unit" = "DAY",
    "dynamic_partition.start" = "-10",
    "dynamic_partition.end" = "7",
    "dynamic_partition.prefix" = "p",
    "dynamic_partition.buckets" = "32"
)
;
'''.format(partition_str)

    db.run_sql(ods_sql_str)
    logger.info("ods sql finished.")

    symbol_list = stock_zh_a_util.get_stock_list()
    for symbol in symbol_list:
        compute_technical_indicators(db, symbol, ds, backfill)


