import sys, os
parent_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parent_dir)


import traceback
import subprocess
import os
import re
import uuid

from utils.log_util import get_logger

from utils.starrocks_db_util import get_starrocks_config

logger = get_logger(__name__)


class DataxProcessor:

    def __init__(self, columns, table, where, data_file_name, datax_path="/home/lotus/sandbox/datax/bin/datax.py"):
        self.columns = columns
        self.table = table
        self.where = where
        self.data_file_name = data_file_name
        self.conf_file_name = self.get_conf_file_name(data_file_name)
        self.temp_file_prefix = str(uuid.uuid4().hex)

        self.datax_path = datax_path

        self.conf_str = self._generate_conf_str(self.columns, self.table, self.where, self.temp_file_prefix)
        self._write_conf_to_file(self.conf_str, self.conf_file_name)

    @classmethod
    def get_conf_file_name(cls, data_file_name):
        return "{}_conf.json".format(data_file_name.split(".")[0])


    def _clean_up_temp_file(self, prefix):
        fnames = [f for f in os.listdir(".") if re.match(f"^{prefix}.*", f)]
        for fname in fnames:
            logger.info(f"clean up temp file {fname}")
            os.remove(fname)

    def _generate_conf_str(self, columns, table, where, temp_file_prefix):
        server_address, port, db_name, user, password = get_starrocks_config()
        template = '''
            {
                "job": {
                    "setting": {
                        "speed": {
                             "channel":1
                        },
                        "errorLimit": {
                            "record": 0,
                            "percentage": 0.02
                        }
                    },
                    "content": [
                        {
                            "reader": {
                                "name": "starrocksreader",
                                "parameter": {
                                    "username": "$username",
                                    "password": "$password",
                                    "column": $columns,
                                    "where": "$where",
                                    "connection": [
                                        {   "table": ["$table"],
                                            "jdbcUrl": [
                                                 "jdbc:mysql://$server_address:$port/$db_name"
                                            ]
                                        }
                                    ]
                                }
                            },
                           "writer": {
                                "name": "txtfilewriter",
                                "parameter": {
                                    "path": "./",
                                    "fileName": "$file_name",
                                    "fileFormat": "csv",
                                    "writeMode": "truncate",
                                    "dateFormat": "yyyy-MM-dd",
                                    "header": $columns
                                }
                            }
                        }
                    ]
                }
            }
        '''
        template = template.replace("$username", user).replace("$password", password).replace("$server_address",server_address).replace("$port", port).replace("$db_name", db_name)
        columns_str = "[" + ', '.join(f'"{c}"' for c in columns) + "]"
        conf_str = template.replace("$columns", columns_str).replace("$table", table).replace("$where", where)
        conf_str = conf_str.replace("$file_name", temp_file_prefix)
        return conf_str

    def _write_conf_to_file(self, conf_str, conf_file_name):
        with open(conf_file_name, "w") as text_file:
            text_file.write(conf_str)
        logger.info(f"conf written to file {conf_file_name}")

    def _sync_data(self, datax_path, conf_file_name):
        logger.info(f"start data sync configured by {conf_file_name}")
        try:
            # datax_path = "/Users/encore/bin/datax/bin/datax.py"
            res = subprocess.run(["python", datax_path, conf_file_name])
            res.check_returncode()
        except:
            logger.warning("exception occurred while running datax.")
            logger.warning(traceback.format_exc())
            return
        logger.info("data sync finished.")

    def _merge_files(self, prefix, out_file_name):
        fnames = [f for f in os.listdir(".") if re.match(f"^{prefix}.*", f)]
        logger.info(f"merging files {fnames} into {out_file_name}")
        with open(out_file_name, "w") as outfile:
            for fname in fnames:
                with open(fname) as infile:
                    for line in infile:
                        outfile.write(line)
        logger.info("merge done")

    # def read_csv(file_name):
    #     logger.info("read data")
    #     df = pd.read_csv(file_name, dtype=str, header=None)
    #     print(df)

    def process(self):
        self._sync_data(self.datax_path, self.conf_file_name)
        self._merge_files(self.temp_file_prefix, self.data_file_name)
        self._clean_up_temp_file(self.temp_file_prefix)

    def clean_up_data_and_conf_files(self):
        logger.info("delete conf file {}".format(self.conf_file_name))
        os.remove(self.conf_file_name)
        logger.info("delete data file {}".format(self.data_file_name))
        os.remove(self.data_file_name)