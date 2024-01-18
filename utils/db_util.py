import sqlalchemy

from utils.log_util import get_logger

logger = get_logger(__name__)

from utils.config_util import get_mysql_config


def get_table_columns(table_name, ignore_ds):
    res = DbUtil().run_sql(f"show columns from {table_name}")
    if ignore_ds:
        res = [item for item in res if item[0] != "ds"]
    cols = [f"{item[0]} {item[1] if item[1] != 'text' else 'varchar(255)'}" for item in res]
    col_names = [item[0] for item in res]
    return cols, col_names



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
    def __init__(self):
        self.engine = None

    def get_db_engine(self):
        if self.engine is None:
            self.engine = self._create_db_engine()
        return self.engine

    def _create_db_engine(self):
        server_address, port, db_name, user, password = get_mysql_config()
        return sqlalchemy.create_engine(f"mysql+pymysql://{user}:{password}@{server_address}:{port}/{db_name}?charset=utf8")

    def table_exists(self, table_name):
        return sqlalchemy.inspect(self.get_db_engine()).has_table(table_name)

    def run_sql(self, sql):
        logger.info("run sql: {}".format(sql))
        sql = sqlalchemy.text(sql)
        result = []
        with self.get_db_engine().connect() as con:
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
