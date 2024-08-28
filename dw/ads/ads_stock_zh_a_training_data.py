import sys, os
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(parent_dir)


from utils.log_util import get_logger
from utils.starrocks_db_util import StarrocksDbUtil
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
    temp_sql_str = '''
    drop table if exists temp_stock_zh_a_hist_filter_a;
    create table if not exists temp_stock_zh_a_hist_filter_a as
    select *,
           row_number() over (partition by 代码  order by 日期 asc) as date_rn
    from dwd_stock_zh_a_hist_df
    where ds in (select max(ds)  from dwd_stock_zh_a_hist_df)
      and period = 'daily'
      and adjust = 'hfq'
      and substr(代码, 1, 3) regexp '600|601|603|000'
      and 代码 in
        (select 代码
         from dwd_stock_individual_info_em_df
         where ds in  (select max(ds) from dwd_stock_individual_info_em_df)
           and 简称 not regexp 'ST|退|PT')
      and 日期>= date_add(current_timestamp(), interval -2 year) ;
'''
    db.run_sql(temp_sql_str)
    logger.info("sql finished.")

    temp_sql_str = '''
    drop table if exists temp_stock_zh_a_hist_filter;
    create table if not exists temp_stock_zh_a_hist_filter as
        select a.*
        ,a.开盘 as open
        ,a.收盘 as close
        ,a.最高 as high
        ,a.最低 as low
        ,a.涨跌幅 as roc
        ,a.成交量 as volume
        ,a.换手率 as turnover_rate
        ,substr(a.代码, 1, 3) as sym
        ,a.代码 as symbol
        ,month(a.日期) as date_mon
        ,dayofweek(a.日期) as date_dow
        ,d.涨跌幅 as roc_m1d
        ,f.涨跌幅 as roc_m2d
        ,g.涨跌幅 as roc_m3d
    from temp_stock_zh_a_hist_filter_a a
    join temp_stock_zh_a_hist_filter_a d on a.代码 = d.代码
        and a.date_rn - d.date_rn = 1
    join temp_stock_zh_a_hist_filter_a f on a.代码 = f.代码
        and a.date_rn - f.date_rn = 2
    join temp_stock_zh_a_hist_filter_a g on a.代码 = g.代码
        and a.date_rn - g.date_rn = 3
        ;
    '''
    db.run_sql(temp_sql_str)
    logger.info("sql finished.")


    temp_sql_str = '''
    
drop table if exists temp_stock_zh_a_index_wide;
create table if not exists temp_stock_zh_a_index_wide as
select a.代码,
if(b.成分券代码 is null, 0, 1) as is_index_000001,
if(c.成分券代码 is null, 0, 1) as is_index_000016,
if(d.成分券代码 is null, 0, 1) as is_index_000300
from (
  select *
  from dwd_stock_individual_info_em_df
  where ds in (select max(ds) from dwd_stock_individual_info_em_df)
  )a
left join (
  select *
  from dwd_index_stock_cons_weight_csindex_df
  where ds in (select max(ds) from dwd_index_stock_cons_weight_csindex_df ) and 指数代码 = '000001'
) b
on a.代码 = b.成分券代码
left join (
  select *
  from dwd_index_stock_cons_weight_csindex_df
  where ds in (select max(ds) from dwd_index_stock_cons_weight_csindex_df ) and 指数代码 = '000016'
) c
on a.代码 = c.成分券代码
left join (
  select *
  from dwd_index_stock_cons_weight_csindex_df
  where ds in (select max(ds) from dwd_index_stock_cons_weight_csindex_df ) and 指数代码 = '000300'
) d
on a.代码 = d.成分券代码
;

select * from temp_stock_zh_a_index_wide limit 10;


drop table if exists temp_stock_zh_a_ti_hist_ext;
create table if not exists temp_stock_zh_a_ti_hist_ext as
select a.*
    ,b.open
    ,b.close
    ,b.high
    ,b.low
    ,b.roc
    ,b.volume
    ,b.turnover_rate
    ,b.sym
    ,b.symbol
    ,b.date_mon
    ,b.date_dow
    ,b.date_rn
    ,b.roc_m1d
    ,b.roc_m2d
    ,b.roc_m3d
    ,coalesce(c.行业, '-') as industry
    ,d.is_index_000001
    ,d.is_index_000016
    ,d.is_index_000300
from (
    select ds, 日期, 代码, adjust, period, ma_5, ma_10, ma_20, ma_60, ma_120, ma_240, hhv_5, hhv_10, hhv_20, hhv_60, hhv_120, hhv_240, llv_5, llv_10, llv_20, llv_60, llv_120, llv_240, bias_6, bias_12, bias_24, boll_upper_20, boll_mid_20, boll_lower_20, rsi_6, rsi_12, rsi_24, wr_10, wr_6, mtm_12, mtm_12_ma_6, k_9, d_9, j_9, macd_dif, macd_dea, macd, dmi_pdi, dmi_mdi, dmi_adx, dmi_adxr, obv, cci, roc_12, ma_6_roc_12, bbi, expma_12, expma_50, ar, br, atr, dma_dif, dma, emv, maemv, psy, psyma, asi, asit, mfi, mass, mamass, dpo, madpo, vr, trix, trma, kc_upper, kc_mid, kc_lower, dc_upper, dc_mid, dc_lower
    from dwd_stock_zh_a_ti_hist_df
    where ds in (select max(ds) from dwd_stock_zh_a_ti_hist_df)
    and adjust='hfq' and period='daily'
    ) a
join temp_stock_zh_a_hist_filter b 
on a.代码 = b.代码 and a.日期 = b.日期
left join (
    select *
    from dwd_stock_individual_info_em_df
    where ds in (select max(ds)  from dwd_stock_individual_info_em_df)
    ) c 
on a.代码 = c.代码
left join temp_stock_zh_a_index_wide d 
on a.代码 = d.代码
;
    '''
    db.run_sql(temp_sql_str)
    logger.info("sql finished.")

    temp_sql_str = '''
    drop table if exists ads_stock_zh_a_training_data;
    create table if not exists ads_stock_zh_a_training_data as
    select
      a.*,
      b.涨跌幅,
      b.roc as label_roc,
      if(b.roc >= 1, 1, 0) as label_cls,
      b.close as label_close,
      b.close as label,
      (b.close - a.close)/a.close*100 as label_roi_1d,
      (c.close - a.close)/a.close*100 as label_roi_2d,
      (d.close - a.close)/a.close*100 as label_roi_3d,
      (f.close - a.close)/a.close*100 as label_roi_5d,
      (g.close - a.close)/a.close*100 as label_roi_10d
    from temp_stock_zh_a_ti_hist_ext a
    join temp_stock_zh_a_hist_filter b 
    on a.代码 = b.代码 and b.date_rn - a.date_rn = 1
    left join temp_stock_zh_a_hist_filter c 
    on a.代码 = c.代码 and c.date_rn - a.date_rn = 2
    left join temp_stock_zh_a_hist_filter d 
    on a.代码 = d.代码 and d.date_rn - a.date_rn = 3
    left join temp_stock_zh_a_hist_filter f 
    on a.代码 = f.代码 and f.date_rn - a.date_rn = 5
    left join temp_stock_zh_a_hist_filter g 
    on a.代码 = g.代码 and g.date_rn - a.date_rn = 10;
        '''
    db.run_sql(temp_sql_str)
    logger.info("sql finished.")

    temp_sql_str = '''
    drop table if exists ads_stock_zh_a_pred_data;
    create table if not exists ads_stock_zh_a_pred_data as
    SELECT a. *,
         0 AS label_roc,
         0 AS label_cls,
         0 AS label_close,
         0 AS label,
         0 AS label_roi_1d,
         0 AS label_roi_2d,
         0 AS label_roi_3d,
         0 AS label_roi_5d,
         0 AS label_roi_10d
    from temp_stock_zh_a_ti_hist_ext a
    join (
      select max(日期) as 日期
      from temp_stock_zh_a_ti_hist_ext
    ) b
    on datediff(b.日期, a.日期) <= 30;
        '''
    db.run_sql(temp_sql_str)
    logger.info("sql finished.")

    db.dqc_row_count("ads_stock_zh_a_training_data", ds)
    db.dqc_row_count("ads_stock_zh_a_pred_data", ds)


