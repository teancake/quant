import sys, os

parent_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parent_dir)

from utils.log_util import get_logger
from utils.starrocks_db_util import mysql_to_ods_dwd, content_to_rag
from utils.stock_zh_a_util import is_trade_date

import sys
logger = get_logger(__name__)



def content_sql(dwd_table_name, ds):
    source = "stock_zyjs_ths"
    sql = f"""
    drop table if exists temp_{dwd_table_name}_a;
    create table if not exists temp_{dwd_table_name}_a as 
    select b.简称, b.行业, a.* from (
    select * from {dwd_table_name} where ds="{ds}"
    )a 
    join
    (
    select * from dwd_stock_individual_info_em_df where ds in (select max(ds) from dwd_stock_individual_info_em_df)
    )b 
    on a.股票代码=b.代码;
        
    drop table if exists temp_{dwd_table_name}_b;
    create table if not exists  temp_{dwd_table_name}_b as 
    select gmt_create, gmt_modified, 
        concat(source, "_", sha2(content, 256)) as id,  
        source,
        pub_time,
        content,
        ds 
        from (
            select *,
            "{source}" as source,
            "" as pub_time,
            concat("股票：", coalesce(简称,""), "，代码：", 股票代码, "，行业：", coalesce(行业,""), "，主营业务：", coalesce(主营业务,""), "，产品类型：", coalesce(产品类型,""), "，产品名称：", coalesce(产品名称,""), "，经营范围：", coalesce(经营范围,"")) as content
            from temp_{dwd_table_name}_a
        ) a 
        ;
    """
    return sql, f"temp_{dwd_table_name}_b"

if __name__ == '__main__':
    ds = sys.argv[1]
    logger.info("execute task on ds {}".format(ds))
    if not is_trade_date(ds):
        logger.info(f"{ds} is not trade date. task exits.")
        exit(os.EX_OK)

    mysql_table_name = "stock_zyjs_ths"
    dwd_table_name = mysql_to_ods_dwd(mysql_table_name, ds, di_df="df", unique_columns=["股票代码"], ods_dqc=False)
    sql, table_name = content_sql(dwd_table_name, ds)
    content_to_rag(sql, table_name, ds)

