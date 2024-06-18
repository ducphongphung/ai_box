from src.global_def import *
from threading import Thread

import requests
import pickle as pk

logger = dbg.get_logger(__name__)


def queue_requests_task(requests_data):
    """ 
    queues request so that it will be sent by a forwarder thread later, 
    returns immediately 
    """
    ut.mkdir_if_not_existed(PATH_TASK_DIR)
    path = PATH_TASK_DIR + 'f_r_{}'.format(ut.get_timestamp())
    if not os.path.exists(path):
        pk.dump(requests_data, open(path, 'wb'))


def send_request(requests_data):
    url = requests_data['url']  # mandatory
    params = requests_data.get('params', None)  # optional
    data = requests_data.get('data', None)  # optional
    files = requests_data.get('files', None)  # optional
    timeout = requests_data.get('timeout', None)  # optional
    if timeout is None:
        timeout = INTERNET_SERVICE_TIMEOUT_SEC

    response = None
    send_ok = False
    try:
        if (params is not None) and (data is None) and (files is None):
            response = requests.get(url, params=params, timeout=timeout, verify=False)
        else:
            response = requests.post(url, data=data, files=files, timeout=timeout, verify=False)
        send_ok = True
    except Exception as ex:
        logger.error("requests: error %s", ex)

    if response is not None:
        try:
            response = response.json()
        except:
            pass

    return response, send_ok


def forward_requests_task(requests_data, queue_task=None):
    """ 
    send with autoretry
    queue_task: path to request file 
    """
    response, send_ok = send_request(requests_data)

    if send_ok:
        if queue_task:  # if this is a queued task, remove it when it is done
            os.remove(queue_task)
    else:
        if queue_task is None:  # queue the request so it is repeated later
            queue_requests_task(requests_data)

    return response, send_ok


def forward_to_ws(ws_url, json_data=None, timeout=STD_SERVICE_TIMEOUT_SEC):
    def invoke_ws():
        try:
            if ws_url.startswith('GET'):
                requests.get(ws_url[3:], timeout=timeout)
            else:
                requests.post(
                    ws_url, json=json_data,
                    timeout=timeout, verify=False)
        except requests.ConnectionError:
            logger.error('ConnectionError: {}'.format(ws_url))
        except Exception as ex:
            logger.exception(ex)

    Thread(target=invoke_ws).start()
