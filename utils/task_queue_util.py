import random
import threading
import queue

from concurrent.futures import ThreadPoolExecutor

import time

from utils.log_util import get_logger

logger = get_logger(__name__)

from utils.config_util import get_ollama_config, get_moonshot_config, get_zhipu_config
from utils.redis_util import RedisUtil
import ollama
import openai
import hashlib
import json
class TaskQueue:

    def __init__(self, thread_count=1):
        self.q = queue.Queue()
        self.counter = 0
        self.executor = self.start_workers(thread_count=thread_count)

    def submit_task(self, task_func, task_kwargs):
        self.q.put((task_func, task_kwargs))
        self.counter += 1
        logger.info(f"submit_task, length of queue {self.q.qsize()}, {self.q}")

    def start_workers(self, thread_count):
        logger.info(f"start_workers, thread_count {thread_count}")
        executor = ThreadPoolExecutor(max_workers=thread_count)
        for _ in range(thread_count):
            executor.submit(self._run_tasks)
        return executor

        # threads = []
        # for _ in range(thread_count):
        #     thread = threading.Thread(target=self._run_tasks)
        #     thread.start()
        #     threads.append(thread)
        # return threads

    def stop_workers(self):
        logger.info("stop_workers")
        for _ in range(threading.active_count() - 1):
            self.submit_task(None, None)
        logger.info("wait threads to finish.")
        # for worker in self.workers:
        #     worker.join()
        self.executor.shutdown(wait=True)
        logger.info("workers stopped.")

    def _run_tasks(self):
        while True:
            if self.q.not_empty:
                # always pop from left
                func, kwargs = self.q.get()
                if func is None:
                    logger.info("stop signal received, task thread exit")
                    break
                try:
                    func(**kwargs)
                except:
                    raise
                finally:
                    self.counter -= 1


def test_task_queue():
    tq = TaskQueue()

    def task_func(a, b, c, call_back):
        res = a + b + c
        print(f"res {res}")
        call_back()
        return res

    def call_back_func():
        print("in callback func")

    for i in range(1, 100):
        kwargs = {"a": i, "b": 2, "c": 3, "call_back": call_back_func}
        tq.submit_task(task_func, kwargs)

    time.sleep(10)
    tq.stop_workers()


class Connection:
    def __init__(self, name, qps):
        self.name = name
        self.is_active = False
        self.is_idle = True
        self.qps = qps
        self.last_fetch_time = time.time()

    def connect(self):
        self._conn_func()
        self.is_active = True

    def disconnect(self):
        self._disconn_func()
        self.is_active = False

    def _disconn_func(self):
        pass

    def _conn_func(self):
        pass

    def _fetch_func(self, req):
        pass

    def fetch(self, req):
        logger.info(f"fetch {req} by {self.name}")
        if not self.is_active:
            logger.warning(f"connection {self.name} not initialized.")
            return None
        return self.fetch_thread(req)

    def fetch_thread(self, req):
        self.is_idle = False
        cur_time = time.time()
        dt = cur_time - self.last_fetch_time
        interval = 1 / self.qps + 0.01
        logger.info(
            f"{self.name} current time {cur_time}, last fetch time {self.last_fetch_time}, qps limit {self.qps}")
        if dt < interval:
            logger.info(f"{self.name} sleep {interval - dt} seconds")
            time.sleep(interval - dt)
        else:
            logger.info(f"{self.name} no sleep needed.")
        try:
            resp = self._fetch_func(req)
        except Exception as e:
            logger.error(f"exception occurred {e}")
            resp = {}
        self.last_fetch_time = time.time()
        logger.info(f"{self.name} last fetch time updated {self.last_fetch_time}")
        self.is_idle = True
        return resp


class LocalConnection(Connection):
    def __init__(self, name="local", model="qwen2:7b"):
        super().__init__(name=name, qps=100)
        self.model = model
        ollama_conf = get_ollama_config()
        self.client = ollama.Client(host=f"http://{ollama_conf['server_address']}:{ollama_conf['port']}", timeout=60)
    def _conn_func(self):

        logger.info(f"local conn init.")

    def _fetch_func(self, req):
        logger.info(f"local fetch {req}")
        response = self.client.chat(model=self.model, messages=req["messages"])
        return response


class KimiConnection(Connection):
    def __init__(self):
        super().__init__(name="kimi", qps=2 / 60)
        conf = get_moonshot_config()
        self.model = "moonshot-v1-8k"
        self.client = openai.OpenAI(api_key=conf["api_key"], base_url=conf["base_url"])

    def _conn_func(self):
        logger.info(f"kimi conn init.")

    def _fetch_func(self, req):
        logger.info(f"kimi fetch {req}")
        response = self.client.chat.completions.create(model=self.model, messages=req["messages"], stream=False)
        response = response.choices[0].to_dict()
        return response


class ZhipuConnection(Connection):
    def __init__(self):
        super().__init__(name="zhipu", qps=10 / 60)
        conf = get_zhipu_config()
        self.model = "glm-4-flash"
        self.client = openai.OpenAI(api_key=conf["api_key"], base_url=conf["base_url"])

    def _conn_func(self):
        logger.info(f"{self.name} conn init.")

    def _fetch_func(self, req):
        logger.info(f"{self.name} fetch {req}")
        response = self.client.chat.completions.create(model=self.model, messages=req["messages"], stream=False)
        response = response.choices[0].to_dict()
        return response


class ConnectionPool:
    def __init__(self):
        self.conns = list()
        self.lock = threading.Lock()

    def add(self, conn: Connection):
        logger.info(f"add conn {conn.name} to pool")
        if not conn.is_active:
            conn.connect()
        self.conns.append(conn)

    def delete(self, conn_name):
        logger.info(f"remove conn {conn_name} from pool")
        for conn in self.conns:
            if conn.name == conn_name:
                self.conns.remove(conn)
            else:
                logger.warning(f"{conn_name} not found in pool")


    def get(self):
        self.lock.acquire()
        logger.info(f"get from conn pool")
        if len(self.conns) == 0:
            logger.info("connection pool size is 0, please add connections first")
            self.lock.release()
            return None
        i = random.randint(0, len(self.conns) - 1)
        while True:
            if i > len(self.conns) - 1:
                i = 0
            conn = self.conns[i]
            # logger.info(f"conn {conn.name} status, active {conn.is_active}, idle {conn.is_idle}")
            if conn.is_active and conn.is_idle:
                logger.info(f"get {conn.name}")
                self.lock.release()
                return conn
            time.sleep(0.1)


class LLMTaskQueue:
    def __init__(self, name=""):
        self.name = name
        self.tq = TaskQueue(thread_count=3)
        self.rq = queue.Queue()
        self.red = RedisUtil()
        pool = ConnectionPool()
        pool.add(LocalConnection(name="ollama", model="qwen2:7b"))
        pool.add(LocalConnection(name="ollama", model="gemma2:9b"))
        pool.add(KimiConnection())
        pool.add(ZhipuConnection())
        self.pool = pool

    def task_func(self, req, call_back):
        print(f"req {req}")
        request_id = req.pop("request_id")
        json_bytes = json.dumps(sorted(req.items()), sort_keys=True).encode('utf-8')
        key = f"llmtask_{self.name}_req_{hashlib.sha256(json_bytes).hexdigest()}"
        logger.info(f"get cache for key {key}")
        resp = self.red.get(key)
        req["request_id"] = request_id
        if resp is not None:
            resp = json.loads(resp)
            logger.info(f"cache hit for key {key}, use cached value {resp}")
        else:
            conn = self.pool.get()
            resp = conn.fetch(req)
            resp["request_id"] = req["request_id"]
            logger.info(f"put key and value for {key} into cache")
            self.red.put(key, json.dumps(resp))
        call_back(req, resp)

    def call_back_func(self, req, resp):
        print("in callback func")
        self.rq.put(resp)
        print(f"response queue size {self.rq.qsize()}, content {self.rq.queue}")

    def enqueue(self, req):
        kwargs = {"req": req, "call_back": self.call_back_func}
        self.tq.submit_task(self.task_func, kwargs)

    def wait_tasks(self):
        while self.tq.counter > 0:
            logger.info(f"wait tasks in the queue to finish. {self.tq.counter} tasks remain.")
            time.sleep(5)
        logger.info(f"wait tasks in the queue to finish. {self.tq.counter} tasks remain.")

    def get_tasks_outputs(self):
        res = []
        while not self.rq.empty():
            item = self.rq.get()
            res.append(item)
        return res

    def stop(self):
        self.tq.stop_workers()


def test_connection_pool():
    pool = ConnectionPool()
    pool.add(LocalConnection(name="ollama", model="qwen2:7b"))
    pool.add(LocalConnection(name="ollama", model="gemma2:9b"))

    reqs = [f"a{i}" for i in range(50)]
    resps = []
    for req in reqs:
        conn = pool.get()
        resp = conn.fetch(req)
        resps.append(resp)
    print(resps)


def test_task_queue_with_connection_pool():
    tq = TaskQueue(thread_count=3)

    pool = ConnectionPool()
    pool.add(KimiConnection())
    pool.add(LocalConnection())

    def task_func(req, call_back):
        print(f"req {req}")
        pool.get().fetch(req)
        call_back()

    def call_back_func():
        print("in callback func")

    for i in range(1, 100):
        kwargs = {"req": i, "call_back": call_back_func}
        tq.submit_task(task_func, kwargs)

    time.sleep(10)
    tq.stop_workers()


def test_llm_queue():
    q = LLMTaskQueue()
    for i in range(1, 50):
        q.enqueue(i)
    q.wait_tasks()
    print(f"tasks output {q.get_tasks_outputs()}")
    q.stop()

if __name__ == '__main__':
    # test_task_queue()
    # test_connection_pool()
    # test_task_queue_with_connection_pool()
    test_llm_queue()