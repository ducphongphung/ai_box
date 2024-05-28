from src.app_core import rdhc_api

import time
# from periphery import GPIO
import json
import socket
import netifaces
from netifaces import interfaces, ifaddresses, AF_INET
import logging
import uuid
import requests
# from Crypto.Cipher import AES
# from Crypto.Util import Padding
import subprocess
import os
from src.app_core.controller_utils import *
import threading

logger = dbg.get_logger("tt_zone")

KEY_RENDER_PASS = 'RANGDONGRALSMART'
CAM_USERNAME_PASS = 'admin:admin1234'

button_press_start_time = 0
# Global flag to track if reset is in progress
reset_in_progress = False

class IOHandler:
    def __init__(self, button_pin, led_pin):
        # self.button = GPIO(button_pin, "in")
        # self.led = GPIO(led_pin, "out")
        self.button_state = 0
        self.btn_hold_on = False
        self.btn_press_once = False

    def check_button_state(self):
        button_state = self.button.read()  

        if button_state == 0 and self.button_state == 1:
            logger.info("Button change state")
            start_time = time.time()
            while button_state == 0:
                button_state = self.button.read()
                time.sleep(0.05)
            check_time = time.time() - start_time
            if check_time <= 0.5:
                logger.info("Press once")
                self.btn_press_once = True
                self.btn_hold_on = False
                threading.Thread(target=self.handle_event_press_once).start()
            else:
                logger.info(f"Button is held for {check_time} seconds")
                self.btn_press_once = False
                self.btn_hold_on = True

                if check_time < 10:
                    threading.Thread(target=self.handle_event_hold_on, args=(3,)).start()
                    scan_hc()
                elif 10 <= check_time < 15:
                    threading.Thread(target=self.handle_event_hold_on, args=(4,)).start()
                    reset_factory()
                else:
                    threading.Thread(target=self.handle_event_hold_on, args=(5,)).start()
                    logger.info("Feature has been canceled")

        self.button_state = button_state



    # def handle_event_press_once(self):
    #     while not self.btn_hold_on:
    #         num_blinks = int(check_time)  # Chuyển đổi check_time thành một số nguyên để xác định số lần nhấp
    #         for _ in range(num_blinks):
    #             self.led.write(True)
    #             time.sleep(0.15)
    #             self.led.write(False)
    #             time.sleep(0.15)

    def handle_event_hold_on(self, blinks):
        for _ in range(blinks):
            self.led.write(True)
            time.sleep(0.5)
            self.led.write(False)
            time.sleep(0.5)

    def cleanup(self):
        self.button.close()
        self.led.close()


def scan_aihub():
    mac_address = ':'.join(
        ['{:02x}'.format((uuid.getnode() >> elements) & 0xff) for elements in range(0, 8 * 6, 8)][::-1])
    client = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)  # UDP
    client.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
    client.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    client.bind(("", 37021))
    try:
        ip_address = netifaces.ifaddresses('eth0')[netifaces.AF_INET][0]['addr']
    except:
        try:
            ip_address = netifaces.ifaddresses('wlan0')[netifaces.AF_INET][0]['addr']
        except:
            ip_address = ''
    while True:
        data, addr = client.recvfrom(1024)
        #    receiveMsg = json.loads(data.decode("utf-8"))
        print(data.decode("utf-8"))
        if (data.decode("utf-8") == "SCAN_AIHUB"):
            msg = {
                "CMD": "AIHUB_RESPONSE",
                "IP": ip_address,
                "HOSTNAME": socket.gethostname(),
                "MAC": mac_address
            }
            print(msg)
            client.sendto(json.dumps(msg).encode("utf-8"), addr)


scan_HC_msg = rdhc_api.message_udp_hc()
def scanHC():
    allips = ip4_addresses()
    receive_msg = []
    for ip in allips:
        sock = sendMsg(ip=ip, msg=scan_HC_msg)
        print(sock)
        try:
            data, address = sock.recvfrom(1024)
            receive_data = json.loads(data.decode("utf-8"))
            receive_msg.append(receive_data)
        except Exception:
            pass
        sock.close()
    print(receive_msg)
    return receive_msg


def sendMsg(ip, msg):
    logging.info("sending on {}".format(ip))
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)  # UDP
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    sock.settimeout(1)
    sock.bind((ip, 8081))
    sock.sendto(json.dumps(msg).encode("utf-8"), ('<broadcast>', 8181))
    return sock


def ip4_addresses():
    ip_list = []
    for interface in interfaces():
        addr = ifaddresses(interface)
        if AF_INET in addr.keys():
            for link in addr[AF_INET]:
                ip_list.append(link['addr'])

    return ip_list


def scan_hc():
    url = 'http://127.0.0.1:9089/hc-scan'
    response = requests.get(url)
    if response.status_code == 200:
        data = response.text
        if data == "success":
            return True
        else:
            return False
    else:
        # Request was not successful
        print('Error:', response.status_code)
        return False


def run_scan_hc_thread():
    button_pin = 138
    led_pin = 135
    io_handler = IOHandler(button_pin, led_pin)
    try:
        while True:
            io_handler.check_button_state()
            time.sleep(0.05)
    except KeyboardInterrupt:
        io_handler.cleanup()
        logger.info("Exiting...")


# def hash_pass(mac):
#     text_hash = "2804" + str(mac.lower().replace(':', ''))
#     pass_mqtt = render_pass(text_hash).upper()
#     return pass_mqtt


# def render_pass(plaintext):
#     padded_plaintext = Padding.pad(bytes(plaintext, encoding='utf-8'), AES.block_size)
#     cipher = AES.new(bytes(KEY_RENDER_PASS, encoding='utf-8'), AES.MODE_ECB)
#     ciphertext = cipher.encrypt(padded_plaintext)
#     return ciphertext.hex()
 
import re
from datetime import datetime

def reset_factory():
    global reset_in_progress
    if not reset_in_progress:
        reset_in_progress = True
        script_directory = '/home/mendel/ai_hub_factory/'
        reset_script = 'reset_factory.sh'

        try:
            time.sleep(5)
            cmd = f'cd {script_directory} && ./{reset_script}'
            subprocess.run(cmd, shell=True, check=True)
            logger.info("Scheduled reset_factory with a 5-second delay")
            checkpoint_folders = os.listdir('/home/mendel')
            regex = r'ai_hub_cp_(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})'
            matching_folders = [folder for folder in checkpoint_folders if re.match(regex, folder)]
            
            if matching_folders:
                # Sắp xếp các thư mục dựa trên thời gian
                sorted_folders = sorted(matching_folders, key=lambda folder: datetime.strptime(re.search(regex, folder).group(1), "%Y-%m-%dT%H:%M:%S"))
                newest_checkpoint = sorted_folders[-1]
            else:
                logger.error("No folders with the required format found.")
                return

            # Cập nhật liên kết ai_hub
            os.system(f'ln -snf {newest_checkpoint} /home/mendel/ai_hub')  
            
            # Khởi động lại tất cả các tiến trình trong supervisor
            cmd_restart_supervisor = 'sudo supervisorctl restart all'
            subprocess.run(cmd_restart_supervisor, shell=True, check=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"Error running reset_factory: {str(e)}")
        except Exception as e:
            logger.error(f"An unexpected error occurred: {str(e)}")
        finally:
            reset_in_progress = False


