import sys, os

parent_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parent_dir)

from utils.log_util import get_logger
from utils.starrocks_db_util import mysql_to_ods_dwd, content_to_rag
from utils.stock_zh_a_util import is_trade_date

import sys

logger = get_logger(__name__)


def content_sql(dwd_table_name, ds):
    source = "stock_telegraph_cls"
    content_table_name = f"temp_{dwd_table_name}_content"
    sql = f"""
    drop table if exists {content_table_name};
    create table if not exists {content_table_name} as 
    select gmt_create, gmt_modified, 
        concat(source, "_", sha2(content, 256)) as id,  
        source,
        pub_time,
        content,
        ds 
    from (
        select *,
            "{source}" as source,
            发布时间 as pub_time,
            内容 as content
        from {dwd_table_name}  where ds="{ds}"
        and 内容 is not null
    ) a 
    """
    return sql, content_table_name

if __name__ == '__main__':
    ds = sys.argv[1]
    logger.info("execute task on ds {}".format(ds))
    if not is_trade_date(ds):
        logger.info(f"{ds} is not trade date. task exits.")
        exit(os.EX_OK)

    mysql_table_name = "stock_telegraph_cls"
    dwd_table_name = mysql_to_ods_dwd(mysql_table_name, ds, di_df="di", unique_columns=["发布日期", "内容"], lifecycle=3650)
    sql, table_name = content_sql(dwd_table_name, ds)
    content_to_rag(sql, table_name, ds)
