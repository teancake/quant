import sys, os
parent_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parent_dir)


from utils.log_util import get_logger
from utils.starrocks_db_util import StarrocksDbUtil, get_days_ahead_ds
from utils.stock_zh_a_util import is_trade_date
import sys

logger = get_logger(__name__)


if __name__ == '__main__':
    ds = sys.argv[1]
    logger.info("execute task on ds {}".format(ds))

    if not is_trade_date(ds):
        logger.info(f"{ds} is not trade date. task exits.")
        exit(os.EX_OK)


    db = StarrocksDbUtil()
    dwd_sql_str = '''
CREATE TABLE if not exists dwd_stock_zh_a_ti_hist_df LIKE ods_stock_zh_a_ti_hist;
INSERT OVERWRITE dwd_stock_zh_a_ti_hist_df PARTITION(p{})
select 
'{}' as `ds`,
`gmt_create` ,
`gmt_modified` ,
`日期`,
`代码`,
`ma_5`,
`ma_10`,
`ma_20`,
`ma_60`,
`ma_120`,
`ma_240`,
`hhv_5`,
`hhv_10`,
`hhv_20`,
`hhv_60`,
`hhv_120`,
`hhv_240`,
`llv_5`,
`llv_10`,
`llv_20`,
`llv_60`,
`llv_120`,
`llv_240`,
`bias_6`,
`bias_12`,
`bias_24`,
`boll_upper_20`,
`boll_mid_20`,
`boll_lower_20`,
`rsi_6`,
`rsi_12`,
`rsi_24`,
`wr_10`,
`wr_6`,
`mtm_12`,
`mtm_12_ma_6`,
`k_9`,
`d_9`,
`j_9`,
`macd_dif`,
`macd_dea`,
`macd`,
`dmi_pdi`,
`dmi_mdi`,
`dmi_adx`,
`dmi_adxr`,
`obv`,
`cci`,
`roc_12`,
`ma_6_roc_12`,
`bbi`,
`expma_12`,
`expma_50`,
`ar`,
`br`,
`atr`,
`dma_dif`,
`dma`,
`emv`,
`maemv`,
`psy`,
`psyma`,
`asi`,
`asit`,
`mfi`,
`mass`,
`mamass`,
`dpo`,
`madpo`,
`vr`,
`trix`,
`trma`,
`kc_upper`,
`kc_mid`,
`kc_lower`,
`dc_upper`,
`dc_mid`,
`dc_lower`
from (
  select *, row_number()over (partition by 日期, 代码 order by gmt_create desc) as rn
  from ods_stock_zh_a_ti_hist
  where ds >= '{}'
) a
where rn = 1;
    '''.format(ds, ds, get_days_ahead_ds(ds, 7))

    db.run_sql(dwd_sql_str)
    logger.info("dwd sql finished.")

    db.dqc_row_count("dwd_stock_zh_a_ti_hist_df", ds)


