import time

import paho.mqtt.client as mqtt
import uuid
import hashlib
import netifaces
from src.utils.debug import get_logger
from src.app_core.conf import read_json_conf

logger = get_logger("rdhc_api")

TOPIC_V1 = "RD_STATUS"
TOPIC_V2 = "device/HC"
TOPIC_V3 = "RD_CONTROL"
TOPIC_V4 = "HC/device"


def hash_name(name):
    hash_value = 0
    for c in name:
        hash_value = (hash_value * 31 + ord(c)) % 2 ** 64
    return hash_value


def get_unique_number(name):
    hash_value = hash_name(str(name))
    unique_number = hash_value % 1000
    return unique_number


def gen_id_uuid_device(string):
    combined_string = string
    md5_hash = hashlib.md5(combined_string.encode()).digest()
    encoded_uuid = uuid.UUID(bytes=md5_hash)
    return str(encoded_uuid)


def get_id_aihub():
    mac_aihub = ':'.join(
        ['{:02x}'.format((uuid.getnode() >> elements) & 0xff) for elements in range(0, 8 * 6, 8)][::-1])
    return gen_id_uuid_device(mac_aihub)


def set_hc(hc_pass, hc_ip, hc_ver):
    global g_HC_IP
    global g_HC_PASS
    global g_HC_VERSION
    g_HC_IP = hc_ip
    g_HC_PASS = hc_pass
    g_HC_VERSION = hc_ver
    return True


# HC config
g_HC_USER = 'RD'
g_HC_VERSION = 1
g_HC_IP = read_json_conf('/sc/9089.json').get('hc_ip')
g_HC_PASS = read_json_conf('/sc/9089.json').get('hc_pass')


def send_hc_msg(hc_msg_v1, hc_msg_v2):
    success = False
    client = mqtt.Client("P1", protocol=mqtt.MQTTv311)
    try:
        client.username_pw_set(username=g_HC_USER, password=g_HC_PASS)
        client.connect(g_HC_IP)
        try:
            client.loop_start()

            if int(g_HC_VERSION) >= 2:
                logger.info(hc_msg_v2)
                rs = client.publish(TOPIC_V2, payload=hc_msg_v2, qos=0)
            else:
                logger.info(hc_msg_v1)
                rs = client.publish(TOPIC_V1, payload=hc_msg_v1, qos=0)

            if rs.rc == mqtt.MQTT_ERR_SUCCESS:
                success = True
            else:
                logger.error(f"mqtt send err_code: {rs.rc}")
        except Exception as ex:
            logger.exception(ex)
        finally:
            client.loop_stop()
    except Exception as ex:
        logger.error(f"Cannot send event to hc at IP: {g_HC_IP}: {ex}")
    finally:
        client.disconnect()

    return success


current_zone_value = None
current_face_gesture_value = None


def send_event_zone_hc(event, zone_id):
    global current_zone_value
    try:
        new_zone_value = event  # Giả sử event chính là giá trị mới của ZONE_VALUE
        if current_zone_value != new_zone_value:
            current_zone_value = new_zone_value  # Cập nhật giá trị hiện tại
            hc_msg_v1 = \
                """{{
                  "CMD": "ZONE",
                  "DATA": {{
                    "DEVICE_UNICAST_ID": {0},
                    "ZONE_VALUE": {1},
                    "ID": "{2}"
                  }}
                }}""".format(int(get_unique_number(zone_id)) + 2000, event, zone_id)

            hc_msg_v2 = \
                """{{
                    "cmd": "DeviceStatus",
                    "rqi": "abc123456",
                    "data": {{
                        "device": [{{
                            "id": "{0}",
                            "data": {{
                                "zoneId": "{1}",
                                "zoneValue": {2}
                            }}
                        }}]
                    }}
                }}""".format(get_id_aihub(), zone_id, event)

            return send_hc_msg(hc_msg_v1, hc_msg_v2)
        else:
            # Giá trị không thay đổi, không cần gửi sự kiện
            return True
    except Exception as ex:
        logger.exception(ex)
        return False


def send_event_fire_hc(event, img_fire, cam_id):
    try:
        hc_msg_v1 = \
            """{{
              "CMD": "SMOKE_FIRE",
              "DATA": {{
                "DEVICE_UNICAST_ID": {0},
                "SMOKE_FIRE_VALUE": {1},
                "UUID": "{2}",
                "IMG": "{3}"
              }}
            }}""".format(int(get_unique_number(cam_id)) + 2000, event, cam_id, img_fire)

        hc_msg_v2 = \
            """{{
                "cmd": "DeviceStatus",
                "rqi": "abc123456",
                "data": {{
                    "device": [{{
                        "id": "{0}",
                        "data": {{
                            "zoneId": "{1}",
                            "zoneValue": {2}
                        }}
                    }}]
                }}
            }}""".format(get_id_aihub(), cam_id, event)
        return send_hc_msg(hc_msg_v1, hc_msg_v2)
    except Exception as ex:
        logger.exception(ex)
        return False


def send_event_fall_hc(event, img_fall, cam_id):
    # TODO: sua ban tin nga
    try:
        hc_msg_v1 = \
            """{{
              "CMD": "FALL",
              "DATA": {{
                "DEVICE_UNICAST_ID": {0},
                "FALL_VALUE": {1},
                "UUID": "{2}",
                "IMG": "{3}"
              }}
            }}""".format(int(get_unique_number(cam_id)) + 2000, event, cam_id, img_fall)

        hc_msg_v2 = \
            """{{
                "cmd": "DeviceStatus",
                "rqi": "abc123456",
                "data": {{
                    "device": [{{
                        "id": "{0}",
                        "data": {{
                            "zoneId": "{1}",
                            "zoneValue": {2}
                        }}
                    }}]
                }}
            }}""".format(get_id_aihub(), cam_id, event)
        return send_hc_msg(hc_msg_v1, hc_msg_v2)
    except Exception as ex:
        logger.exception(ex)
        return False


def get_aihub_mac():
    try:
        return ':'.join(['{:02x}'.format((uuid.getnode() >> elements) & 0xff) for elements in range(0, 8 * 6, 8)][::-1])
    except Exception as ex:
        logger.exception(ex)
        return None


def get_aihub_ip():
    ip = None
    try:
        try:
            ip = netifaces.ifaddresses('enp4s0')[netifaces.AF_INET][0]['addr']
        except Exception as ex:
            logger.exception(ex)
            try:
                ip = netifaces.ifaddresses('lo')[netifaces.AF_INET][0]['addr']
            except Exception as ex:
                logger.exception(ex)
    except Exception as ex:
        logger.exception(ex)

    return ip

import subprocess
import re


# def get_aihub_ip():
#     try:
#         # Thực thi lệnh "ip addr show" và lấy kết quả đầu ra
#         result = subprocess.run(["ip", "addr", "show"], capture_output=True, text=True)
#         output = result.stdout
#
#         # Tìm kiếm dòng có chứa thông tin địa chỉ IP của giao diện enp4s0
#         match = re.search(r'inet (\d+\.\d+\.\d+\.\d+)/\d+ brd', output)
#
#         if match:
#             ip = match.group(1)
#             return ip
#         else:
#             return None
#
#     except Exception as ex:
#         print(f"An error occurred: {ex}")
#         return None


def add_aihub_to_hc():
    device_type_id = 50331649
    mac_aihub = get_aihub_mac()
    ip_aihub = get_aihub_ip()

    if ip_aihub is None or mac_aihub is None:
        return False

    try:
        hc_msg_v1 = \
            """{{
                "CMD": "NEW_DEVICE",
                "DATA": {{
                "DEVICE_ID": "{0}",
                "DEVICE_UNICAST_ID":{1},
                "DEVICE_TYPE_ID": {2},
                "MAC_ADDRESS": "{3}",
                "FIRMWARE_VERSION": "1.0.2",
                "DEVICE_KEY": "b717f8d8-6f18-43c0-ae46-69c32998f653",
                "NET_KEY": "b717f8d8-6f18-43c0-ae46-69c32998f653",
                "APP_KEY": "b717f8d8-6f18-43c0-ae46-69c32998f653",
                "IP":"{4}"
                }}
            }}""".format(get_id_aihub(), get_unique_number(mac_aihub) + 3000, device_type_id, mac_aihub, ip_aihub)

        hc_msg_v2 = \
            """{{
               "cmd": "AddDevice",
               "rqi": "abc123456",
               "data": {{
                    "id": "{0}",
                    "type": {1},
                    "mac": "{2}",
                    "data": {{
                      "ipLan": "{3}"
                    }}
              }}
            }}""".format(get_id_aihub(), device_type_id, mac_aihub, ip_aihub)

        return send_hc_msg(hc_msg_v1, hc_msg_v2)
    except Exception as ex:
        logger.exception(ex)
        return False


def add_face_zone_device(device_name_id, device_type_string):
    try:
        if device_type_string == 'Face':
            device_unicast_id = get_unique_number(device_name_id) + 1000
            device_type_id = 81102
        else:  # zone
            device_unicast_id = get_unique_number(device_name_id) + 2000
            device_type_id = 81101

        hc_msg_v1 = \
            """{{
                "CMD": "NEW_DEVICE",
                "DATA": {{
                    "DEVICE_ID": "{0}",
                    "DEVICE_UNICAST_ID":{1},
                    "DEVICE_TYPE_ID": {2},
                    "MAC_ADDRESS": "b3:ab:23:38:c1:a4",
                    "FIRMWARE_VERSION": "1.0",
                    "DEVICE_KEY": "b717f8d8-6f18-43c0-ae46-69c32998f653",
                    "NET_KEY": "b717f8d8-6f18-43c0-ae46-69c32998f653",
                    "APP_KEY": "b717f8d8-6f18-43c0-ae46-69c32998f653"
                }}
            }}""".format(device_name_id, device_unicast_id, device_type_id)

        hc_msg_v2 = \
            """{{
                "cmd": "AddFunction",
                "rqi": "abc123456",
                "data": {{
                    "id": "{0}",
                    "data": {{
                        "type": "{1}",
                        "id": "{2}"
                    }}
                }}
            }}""".format(get_id_aihub(), device_type_string, gen_id_uuid_device(device_name_id))

        return send_hc_msg(hc_msg_v1, hc_msg_v2)
    except Exception as ex:
        logger.exception(ex)
        return False


def del_device(device_name_id):
    try:
        hc_msg_v1 = """{{
          "CMD": "RESET_NODE",
          "DATA": [
            "{0}"
          ]
        }}""".format(gen_id_uuid_device(device_name_id))
        hc_msg_v2 = \
            """{{
                "cmd": "delDev",
                "rqi": "abc123456",
                "data": {{
                    "device": ["{0}"]
                }}
            }}""".format(gen_id_uuid_device(device_name_id))

        return send_hc_msg(hc_msg_v1, hc_msg_v2)
    except Exception as ex:
        logger.exception(ex)
        return False


def send_event_face_hc(face_id):
    try:
        if face_id != 'Unknown':
            device_id = gen_id_uuid_device(face_id)
            hc_msg_v2 = \
                """{{
                    "cmd": "DeviceStatus",
                    "rqi": "abc123456",
                    "data": {{
                        "device": [{{
                            "id": "{0}",
                            "data": {{
                                "faceId": "{1}",
                                "faceValue": 1
                            }}
                        }}]
                    }}
                }}""".format(get_id_aihub(), gen_id_uuid_device(face_id))

            hc_msg_v1 = \
                """{{
                    "CMD":"FACE",
                    "DATA":
                    {{
                       "DEVICE_UNICAST_ID":{0},
                       "FACE_VALUE":1,
                       "ID": "{1}"
                    }}
                }}""".format(int(get_unique_number(face_id)) + 1000, face_id)

            return send_hc_msg(hc_msg_v1, hc_msg_v2)
        else:
            return True
    except Exception as ex:
        logger.exception(ex)
        return False


TIME_WAIT_FOR_SENDING_HC = 4

prev_gesture = None
prev_time_g = None


def send_event_gesture_hc(event, gesture_id, cam_id):
    global prev_gesture
    global prev_time_g

    current_time = time.time()
    try:
        if gesture_id != prev_gesture or prev_time_g is None or (
                prev_time_g is not None and current_time - prev_time_g > TIME_WAIT_FOR_SENDING_HC):
            prev_gesture = gesture_id
            prev_time_g = current_time

            hc_msg_v1 = \
                """{{
                  "CMD": "GESTURE",
                  "DATA": {{
                    "DEVICE_UNICAST_ID": {0},
                    "GESTURE_VALUE": {1},
                    "UUID": "{2}"
                  }}
                }}""".format(int(get_unique_number(gesture_id)) + 3000, event, cam_id)

            hc_msg_v2 = \
                """{{
                    "cmd": "DeviceStatus",
                    "rqi": "abc123456",
                    "data": {{
                        "device": [{{
                            "id": "{0}",
                            "data": {{
                                "gestureId": "{1}",
                                "gestureValue": {2}
                            }}
                        }}]
                    }}
                }}""".format(get_id_aihub(), cam_id, event)
            return send_hc_msg(hc_msg_v1, hc_msg_v2)
        else:
            return False

    except Exception as ex:
        logger.exception(ex)
        return False


prev_face_gesture = (None, None)
prev_time_fg = None


def send_event_face_gesture_hc(face_id, gesture):
    global prev_face_gesture
    global prev_time_fg

    current_time = time.time()

    try:
        if face_id != prev_face_gesture[0] or gesture != prev_face_gesture[1] or prev_time_fg is None or (
                prev_time_fg is not None and current_time - prev_time_fg > TIME_WAIT_FOR_SENDING_HC):
            device_id = gen_id_uuid_device(face_id)
            prev_face_gesture = (face_id, gesture)  # update new state
            prev_time_fg = current_time
            hc_msg_v2 = \
                """{{
                    "cmd": "DeviceStatus",
                    "rqi": "abc123456",
                    "data": {{
                        "device": [{{
                            "id": "{0}",
                            "data": {{
                                "faceId": "{1}",
                                "faceValue": 1,
                                "gesture":{2}
                            }}
                        }}]
                    }}
                }}""".format(get_id_aihub(), gen_id_uuid_device(face_id), gesture)

            hc_msg_v1 = \
                """{{
                    "CMD":"FACE_GESTURE",
                    "DATA":
                    {{
                       "DEVICE_UNICAST_ID":{0},
                       "FACE_VALUE":1,
                       "GESTURE":{1},
                       "ID": "{2}"
                    }}
                }}""".format(int(get_unique_number(face_id)) + 1000, gesture, face_id)

            return send_hc_msg(hc_msg_v1, hc_msg_v2)
        else:
            return False
    except Exception as ex:
        logger.exception(ex)
        return False


def message_udp_hc():
    mac_aihub = get_aihub_mac()
    ip_aihub = get_aihub_ip()
    aihub_id = get_id_aihub()

    message = {
        "cmd": "aiHubBroadCast",
        "rqi": "abc123456",
        "data": {
            "ip": ip_aihub,
            "mac": mac_aihub,
            "ver": "1.3",
            "id": aihub_id
        }
    }
    return message
