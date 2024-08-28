import sqlalchemy

from utils.log_util import get_logger

logger = get_logger(__name__)

from utils.config_util import get_mysql_config


def mysql_datatypes_to_starrocks(mysql_type):
    map = {"text": "string",
           "timestamp": "datetime"}
    if mysql_type in map.keys():
        return map[mysql_type]
    else:
        return mysql_type


def get_table_columns(table_name):
    res = DbUtil().run_sql(f"show columns from {table_name}")
    cols = dict()
    for item in res:
        col_name, col_type = item[0], mysql_datatypes_to_starrocks(item[1])
        cols[col_name] = col_type
    return cols


# db
# create database akshare_data;
# CREATE USER 'quant'@'localhost' IDENTIFIED BY 'quant';
# GRANT ALL PRIVILEGES ON akshare_data.* TO 'quant'@'localhost';
#
# GRANT ALL PRIVILEGES ON *.* TO 'quant'@'%'
#     IDENTIFIED BY 'quant'
#     WITH GRANT OPTION;
# FLUSH PRIVILEGES;
class DbUtil:
    # The typical usage of create_engine() is once per particular database URL,
    # held globally for the lifetime of a single application process.
    # A single Engine manages many individual DBAPI connections on behalf of the process
    # and is intended to be called upon in a concurrent fashion.
    # The Engine is not synonymous to the DBAPI connect() function,
    # which represents just one connection resource -
    # the Engine is most efficient when created just once at the module level of an application,
    # not per-object or per-function call.
    # create_engine 会建立一个连接池，一个 engine就够了，否则mysqld创建很多连接之后可能会OOM
    #
    server_address, port, db_name, user, password = get_mysql_config()
    engine = sqlalchemy.create_engine(
        f"mysql+pymysql://{user}:{password}@{server_address}:{port}/{db_name}?charset=utf8")

    def __init__(self):
        pass

    def get_db_engine(self):
        return self.engine

    def table_exists(self, table_name):
        return sqlalchemy.inspect(self.get_db_engine()).has_table(table_name)

    def run_sql(self, sql):
        logger.info("run sql: {}".format(sql))
        sql = sqlalchemy.text(sql)
        result = []
        with self.get_db_engine().connect() as con:
        # When the Connection is closed at the end of the with: block,
        # the referenced DBAPI connection is released to the connection pool.
        # From the perspective of the database itself, the connection pool will not
        # actually “close” the connection assuming the pool has room to store this connection for the next use.
        # When the connection is returned to the pool for re-use,
        # the pooling mechanism issues a rollback() call on the DBAPI connection
        # so that any transactional state or locks are removed
            trans = con.begin()
            try:
                cursor_result = con.execute(sql)
                trans.commit()
            except:
                trans.rollback()
                logger.warn("sql execution failed. transaction rolled back.")
                raise
            try:
                result = list(cursor_result)
            except:
                # if there are no results returned, exception will be ignored.
                pass
        return result

if __name__ == '__main__':
    print((mysql_datatypes_to_starrocks("abc")))
    print((mysql_datatypes_to_starrocks("timestamp")))