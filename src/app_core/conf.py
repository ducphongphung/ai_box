import json
import logging
from urllib.parse import urlparse


def read_json_conf(file='setup/sc/source/camera_zone.json'):
    try:
        with open(file, "r") as json_file:
            json_data = json.load(json_file)
            return json_data
    except Exception as ex:
        logging.exception(ex)
        return {}


def write_cam_info_json(file='setup/sc/source/camera_zone.json', info=None):
    try:
        if info is None:
            info = {}
        with open(file, 'w') as fp:
            json.dump(info, fp)
            return True
    except Exception as ex:
        logging.exception(ex)
        return False


def cam_url_to_info(url, cam_type):
    parsed_url = urlparse(url)
    username = parsed_url.username
    password = parsed_url.password
    ip_address = parsed_url.hostname
    if cam_type.lower() == 'dahua':
        if "subtype=0" in url:
            url = url.replace("subtype=0", "subtype=1")
    elif cam_type.lower() == 'hkvision':
        if "/Channels/101" in url:
            url = url.replace("Channels/101", 'Channels/102')
    return username, password, ip_address, url
