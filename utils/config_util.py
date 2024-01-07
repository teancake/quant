import nacos
import yaml

NAMESPACE = "86eb73b4-0a4e-420f-9363-238ab6ac9471"
SERVER_ADDRESSES = "http://192.168.50.228:31232"
client = nacos.NacosClient(SERVER_ADDRESSES, namespace=NAMESPACE)

group = "quant"


def get_mysql_config():
    data_id = "quant-common-mysql.yml"
    conf = client.get_config(data_id, group)
    conf = yaml.safe_load(conf)
    return conf["server_address"], conf["port"], conf["db_name"], conf["user_name"], conf["password"]


def get_starrocks_config():
    data_id = "quant-common-starrocks.yml"
    conf = client.get_config(data_id, group)
    conf = yaml.safe_load(conf)
    return conf["server_address"], conf["port"], conf["db_name"], conf["user_name"], conf["password"]
