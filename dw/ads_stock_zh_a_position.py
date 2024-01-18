import sys, os
parent_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parent_dir)


from utils.log_util import get_logger
from utils.starrocks_db_util import StarrocksDbUtil
import sys

logger = get_logger(__name__)



if __name__ == '__main__':
    ds = sys.argv[1]
    logger.info("execute task on ds {}".format(ds))
    db = StarrocksDbUtil()
    ddl_sql = '''
CREATE TABLE IF NOT EXISTS `stock_zh_a_transaction`
(
  `gmt_create`   datetime COMMENT "",
  `gmt_modified` datetime COMMENT "",
  `代码`           varchar(20) COMMENT "",
  `名称`           varchar(20) COMMENT "",
  `action`       int comment "0 buy, 1 sell",
  `price`        double COMMENT "",
  `amount`          int COMMENT ""
)
DISTRIBUTED BY HASH (`代码`) BUCKETS 32 PROPERTIES
(
"replication_num" = "1"
);


CREATE TABLE IF NOT EXISTS `ads_stock_zh_a_position`
(
  `代码`              varchar(20) COMMENT "",
  `名称`              varchar(20) COMMENT "",
  `buy_first_date`  datetime COMMENT "",
  `buy_recent_date` datetime COMMENT "",
  `buy_amount` int COMMENT "",
  `buy_price_avg` decimal(10,2) COMMENT "",
  `buy_cost` decimal(10,2) COMMENT "",
  `buy_price_recent` decimal(10,2) COMMENT "",
  `sell_first_date` datetime NULL COMMENT "",
  `sell_recent_date` datetime NULL COMMENT "",
  `sell_amount` int COMMENT "",
  `sell_price_avg` decimal(10,2) COMMENT "",
  `sell_cost` decimal(10,2) COMMENT "",
  `position` int COMMENT "",
  `return` decimal(10,2) COMMENT ""
)
DISTRIBUTED BY HASH (`代码`) BUCKETS 32 PROPERTIES
(
"replication_num" = "1"
);
    '''

    ads_sql_str = '''
insert overwrite ads_stock_zh_a_position 
select buy.代码,
        buy.名称,
        buy_first_date,
        buy_recent_date,
        buy_amount,
        buy_price_avg,
        buy_cost,
        buy_price_recent,
        sell_first_date, 
        sell_recent_date, 
        sell_amount,
        sell_price_avg, 
        sell_cost,
        buy_amount - coalesce(sell_amount, 0) as position,
        sell_price_avg * sell_amount - buy_price_avg * buy_amount - buy_cost - sell_cost as return
from (
  select a.*, buy_amount, buy_cost, buy_first_date, buy_recent_date, buy_price_avg
  from (
      select 代码, 名称, 
      price as buy_price_recent,
      row_number()over (partition by 代码  order by gmt_create desc) as rn
      from stock_zh_a_transaction
      where action = 0
  ) a
  join (
    select 代码, 
           min(gmt_create) as buy_first_date, 
           max(gmt_create) as buy_recent_date, 
           sum(amount) as buy_amount,
           sum(price * amount) / sum(amount) as buy_price_avg,
           count(*) * 5 as buy_cost
    from stock_zh_a_transaction
    where action = 0
    group by 代码
  ) b
  on a.代码 = b.代码
  and a.rn = 1
) buy
left join (
  select min(gmt_create) as sell_first_date, max(gmt_create) as sell_recent_date, 代码, sum(amount) as sell_amount,
         sum(price * amount) / sum(amount) as sell_price_avg, count(*) * 5 as sell_cost
  from stock_zh_a_transaction
  where action = 1
  group by 代码
) sell
on buy.代码 = sell.代码;
'''
    db.run_sql(ddl_sql)
    logger.info("ddl sql finished.")
    db.run_sql(ads_sql_str)
    logger.info("ads sql finished.")


