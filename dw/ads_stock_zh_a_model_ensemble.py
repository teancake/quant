import sys, os
parent_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
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
    ads_sql_str = '''
drop table if exists ads_stock_zh_a_model_ensemble;
SET SESSION query_timeout=600;
create table if not exists ads_stock_zh_a_model_ensemble as
SELECT
  ds,
  日期,
  代码,
  pred_name,
  SUM(if (pred_cls='up', 1, 0)) AS up_cnt,
  SUM(if (pred_cls='neutral', 1, 0)) AS neutral_cnt,
  SUM(if (pred_cls='down', 1, 0)) AS down_cnt,
  group_concat (if (pred_cls='up', model, NULL)) AS up_models,
  group_concat (if (pred_cls='neutral', model, NULL)) AS neutral_models,
  group_concat (if (pred_cls='down', model, NULL)) AS down_models,
  AVG(pred_value) AS pred_value_mean,
  variance (pred_value) AS pred_value_var,
  MAX(pred_value) AS pred_max,
  MIN(pred_value) AS pred_min
FROM
  (
    SELECT
      ds,
      日期,
      model,
      run_id,
      代码,
      pred_name,
      pred_value,
      CASE
        WHEN pred_value>=1 THEN 'up'
        WHEN pred_value<-1 THEN 'down'
        ELSE 'neutral'
      END AS pred_cls
    FROM
      ods_stock_zh_a_prediction
    WHERE
      ds>=date_add (CURDATE (), INTERVAL -365 DAY)
      AND stage='prediction'
  ) a
GROUP BY
  ds,
  日期,
  代码,
  pred_name
'''

    db.run_sql(ads_sql_str)
    logger.info("sql finished.")
    db.dqc_row_count("ads_stock_zh_a_model_ensemble", ds)


