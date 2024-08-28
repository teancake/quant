import redis

from utils.config_util import get_redis_config

class RedisUtil:
    conf = get_redis_config()
    pool = redis.ConnectionPool(host=conf["server_address"], port=conf["port"], decode_responses=True,
                                password=conf["password"])
    r = redis.Redis(connection_pool=pool)

    def put(self, key, val, ttl=7*86400):
        self.r.set(key, val, ex=ttl)

    def filter_existing_records(self, keys):
        vals = self.r.mget(keys)
        return [keys[idx] for idx, val in enumerate(vals) if val is None]

    def record_exists(self, key):
        val = self.r.get(key)
        return val is not None

    def set_records(self, keys):
        mapping = {key: "" for key in keys}
        self.r.mset(mapping)

    def get(self, key):
        return self.r.get(key)

    def delete_records(self, keys):
        self.r.delete(keys)

if __name__ == '__main__':

    red = RedisUtil()
    red.put("key3", "")

    print(red.filter_existing_records(["key3", "key5"]))
    print(red.record_exists("key5"))
    print(red.get("key3"))
