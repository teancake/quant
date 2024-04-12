import sys, os

parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(parent_dir)

from utils.log_util import get_logger
from utils.starrocks_db_util import mysql_to_ods_dwd, content_to_rag

import sys
logger = get_logger(__name__)



def content_sql(dwd_table_name, ds):
    source = "news_cctv"
    sql = f"""
    drop table if exists temp_{dwd_table_name}_b;
    create table if not exists  temp_{dwd_table_name}_b as 
    select gmt_create, gmt_modified, 
        concat(source, "_", sha2(content, 256)) as id,  
        source,
        pub_time,
        content,
        ds 
        from (
            select ds, gmt_create, gmt_modified, 
            "{source}" as source,
            date as pub_time,
            concat(coalesce(title,""), " ",  coalesce(content,"")) as content
            from {dwd_table_name} where ds="{ds}"
        ) a 
        ;
    """
    return sql, f"temp_{dwd_table_name}_b"

if __name__ == '__main__':
    ds = sys.argv[1]
    logger.info("execute task on ds {}".format(ds))

    mysql_table_name = "news_cctv"
    dwd_table_name = mysql_to_ods_dwd(mysql_table_name, ds, di_df="di", lifecycle=3650)
    sql, table_name = content_sql(dwd_table_name, ds)
    content_to_rag(sql, table_name, ds)

