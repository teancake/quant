from airflow.utils.email import send_email
from datetime import timedelta
from airflow.sensors.date_time import DateTimeSensor

def get_remote_ssh_conf():
    return "cheese", "cheese", "192.168.50.20"

def get_default_args():
    return {'email_on_failure': False, "email": ["bowmore.alert@outlook.com"],
            'on_failure_callback': failure_callback, 'retries': 1, 'retry_delay': timedelta(minutes=1)}


def failure_callback(context: dict):
    send_email_with_failure_info(context)


def send_email_with_failure_info(context: dict):
    dag_id = context['dag'].dag_id
    email = context['dag'].default_args['email']
    schedule_interval = context['dag'].schedule_interval
    task_id = context['task_instance'].task_id
    run_id = context['run_id']
    operator = context['task_instance'].operator
    state = context['task_instance'].state
    duration = '%.1f' % context['task_instance'].duration
    max_tries = context['task_instance'].max_tries
    hostname = context['task_instance'].hostname
    start_date = context['task_instance'].start_date.strftime('%Y-%m-%d %H:%M:%S')
    end_date = context['task_instance'].end_date.strftime('%Y-%m-%d %H:%M:%S')
    params = context['params']
    var = context['var']
    test_mode = context['test_mode']
    exception = context['exception']
    execution_date = context['logical_date'].strftime('%Y-%m-%d %H:%M:%S')
    next_execution_date = context['data_interval_end'].strftime('%Y-%m-%d %H:%M:%S')
    msg = f"""<table width='100%' border='1' cellpadding='2' style='border-collapse:collapse'>
            <h3 style='color:red;'>Airflow {task_id} 任务报警</h3>
            <tr><td width='40%'>DAG名称</td><td>{dag_id}</td></tr>
            <tr><td width='40%'>任务名称</td><td>{task_id}</td></tr>
            <tr><td width='40%'>运行周期</td><td>{schedule_interval}</td></tr>
            <tr><td width='40%'>运行ID</td><td>{run_id}</td></tr>
            <tr><td width='40%'>任务类型</td><td>{operator}</td></tr>
            <tr><td width='40%' style='color:red;'>任务状态</td><td style='color:red;'>{state}</td></tr>
            <tr><td width='40%'>重试次数</td><td>{max_tries}</td></tr>
            <tr><td width='40%'>持续时长</td><td>{duration}s</td></tr>
            <tr><td width='40%'>运行主机</td><td>{hostname}</td></tr>
            <tr><td width='40%'>计划执行时间</td><td>{execution_date}</td></tr>
            <tr><td width='40%'>实际开始时间</td><td>{start_date}</td></tr>
            <tr><td width='40%'>实际结束时间</td><td>{end_date}</td></tr>
            <tr><td width='40%'>下次执行时间</td><td>{next_execution_date}</td></tr>
            <tr><td width='40%'>参数</td><td>{params}</td></tr>
            <tr><td width='40%'>变量</td><td>{var}</td></tr>
            <tr><td width='40%'>是否测试模式</td><td>{test_mode}</td></tr>
            <tr><td width='40%' style='color:red;'>错误信息</td><td style='color:red;'>{exception}</td></tr>
            <tr><td width='40%'>上下文</td><td>{str(context)}</td><tr>
            </table>"""
    subject = f'Airflow {task_id} 任务报警'
    send_email(email, subject, msg)


def wait_till(hour: int, minute: int, second: int):
    target_time_str = f'next_execution_date.in_tz("Asia/Shanghai").replace(hour={hour}, minute={minute}, second={second})'
    task_id_str = f"wait_till_{hour:02d}{minute:02d}{second:02d}"
    return DateTimeSensor(
        task_id=task_id_str, target_time="{{ " + target_time_str + " }}", poke_interval=60
    )