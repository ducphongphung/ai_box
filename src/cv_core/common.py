from src.global_def import *
from src import global_def

import cv2
import shutil
import zipfile
import os
import time

import struct
# from Crypto.Cipher import AES

# from cv_core import PATH_OBFUS, PATH_OBFUS_BASE

CRYPTO_KEY = '!%F=-?Pst970JHUS^&^&%**G&#^(*Y(B*YB(79d798q3475bq98iaIULaCiO*&#@%*@SWB*YDluy8'[0:32].encode('utf8')

logger = dbg.get_logger(__name__)

screen_w = 1920
screen_h = 1080

woh = float(screen_w) / screen_h


def cam_transform(frame, sv_cam_type=None, cam_ip=None):
    if sv_cam_type == SV_CAM_KB:
        return cv2.flip(frame, 1)
    elif sv_cam_type == SV_WEBCAM:
        return ut.padFullHD(frame)
    elif sv_cam_type == SV_WEBCAM_HD:
        cv_img = ut.padAspectRatio(frame, woh)
        return cv2.flip(cv_img, 1)
    else:
        return frame.copy()


# region Protect Models
# ==============================================================================
def create_obfusicated_folder_structure():
    t = time.time()
    p = []
    for i in range(1000):
        val = random.randint(0, 9)
        p.append(str(val))
        if val == 0:
            try:
                pass
                # os.makedirs('{}{}'.format(PATH_OBFUS_BASE, '/'.join(p)))
            except:
                pass
            p = []
    # print ('create_obfusicated_folder_structure: {}'.format(time.time() - t))


def unpack_all_models():
    unpack_model('m')
    unpack_model('lpr')


def try_remove(file):
    try:
        os.remove(file)
    except:
        pass


def encrypt_file(key, in_filename, out_filename=None, chunksize=64 * 1024):
    """ Encrypts a file using AES (CBC mode) with the
        given key.

        key:
            The encryption key - a string that must be
            either 16, 24 or 32 bytes long. Longer keys
            are more secure.

        in_filename:
            Name of the input file

        out_filename:
            If None, '<in_filename>.enc' will be used.

        chunksize:
            Sets the size of the chunk which the function
            uses to read and encrypt the file. Larger chunk
            sizes can be faster for some files and machines.
            chunksize must be divisible by 16.
    """
    if not out_filename:
        out_filename = in_filename + '.enc'

    iv = bytes([random.randint(0, 0xFF) for i in range(16)])
    # encryptor = AES.new(key, AES.MODE_CBC, iv)
    filesize = os.path.getsize(in_filename)

    with open(in_filename, 'rb') as infile:
        with open(out_filename, 'wb') as outfile:
            outfile.write(struct.pack('<Q', filesize))
            outfile.write(iv)

            while True:
                chunk = infile.read(chunksize)
                if len(chunk) == 0:
                    break
                elif len(chunk) % 16 != 0:
                    chunk += b' ' * (16 - len(chunk) % 16)

                # outfile.write(encryptor.encrypt(chunk))


def decrypt_file(key, in_filename, out_filename=None, chunksize=24 * 1024):
    """ Decrypts a file using AES (CBC mode) with the
        given key. Parameters are similar to encrypt_file,
        with one difference: out_filename, if not supplied
        will be in_filename without its last extension
        (i.e. if in_filename is 'aaa.zip.enc' then
        out_filename will be 'aaa.zip')
    """
    if not out_filename:
        out_filename = os.path.splitext(in_filename)[0]

    with open(in_filename, 'rb') as infile:
        origsize = struct.unpack('<Q', infile.read(struct.calcsize('Q')))[0]
        iv = infile.read(16)
        # decryptor = AES.new(key, AES.MODE_CBC, iv)

        with open(out_filename, 'wb') as outfile:
            while True:
                chunk = infile.read(chunksize)
                if len(chunk) == 0:
                    break
                # outfile.write(decryptor.decrypt(chunk))

            outfile.truncate(origsize)


def encrypt(model_oname, tmp_dir):
    # e.g. transform folder .m to file .m.cache
    model_o_dir = '{}/.{}'.format(PATH_DATA_DIR, model_oname)
    packed_model_file = '{}/.{}.cache'.format(PATH_DATA_DIR, model_oname[0])
    tmp_file = '{}.cache'.format(tmp_dir)
    try_remove(tmp_file)
    shutil.make_archive(tmp_dir, 'zip', model_o_dir)
    shutil.rmtree(model_o_dir)
    os.rename('{}.zip'.format(tmp_dir), tmp_file)
    encrypt_file(CRYPTO_KEY, tmp_file, packed_model_file)
    os.remove(tmp_file)


def decrypt(src_filename, dst_dir):
    tmp_file = '{}/{}.cache'.format(dst_dir, src_filename)
    try_remove(tmp_file)
    decrypt_file(CRYPTO_KEY, '{}/{}.cache'.format(PATH_DATA_DIR, src_filename), tmp_file)
    with zipfile.ZipFile(tmp_file, 'r') as zipf:
        zipf.extractall(dst_dir)
    try_remove(tmp_file)


def unpack_model(model_oname):
    model_o_dir = '{}/.{}'.format(PATH_DATA_DIR, model_oname)
    packed_model = '{}/.{}.cache'.format(PATH_DATA_DIR, model_oname[0])
    # unpacked_model_dir = '{}{}.{}'.format(PATH_OBFUS_BASE, PATH_OBFUS, model_oname[0])
    try:
        if not os.path.isfile(packed_model) and os.path.isdir(model_o_dir):
            pass
            # encrypt(model_oname, unpacked_model_dir)
    except Exception as ex:
        logger.exception(ex)

    try:
        pass
        # # if not os.path.exists(unpacked_model_dir) and os.path.isfile(packed_model):
        #     create_obfusicated_folder_structure()
        #     os.makedirs(unpacked_model_dir)
        #     decrypt('.' + model_oname[0], unpacked_model_dir)
    except Exception as ex:
        # logger.exception(ex)
        pass


# ==============================================================================
# endregion

# region Licensing, obfuscicated by putting it here
# ===============================================================================
import base64
import json
import hashlib
import random
from threading import Thread

UPDATE_LICENSE_FILE = True

PATH_RTL_DIR = PATH_DATA_DIR
PATH_RTL = PATH_RTL_DIR + '/.rtl2'
LICENSE_TMP_FILE = PATH_RTL + '.tmp'
LICENSE_UPDATE_INTERVAL_SEC = 60

RTL_ENCRYPT_KEY = "Oh!"

LICENSE_VER = '1'

_hdd_name = None


def get_hdd_name():
    global _hdd_name
    if _hdd_name is None:
        try:
            _hdd_name = os.popen("lsblk | grep disk | awk '{ printf $1 }'").read()
        except Exception as ex:  # hide the exception
            pass
    return _hdd_name


_hdd_serial = None  # cache to minimize conflict between subprocess and mxnet


def get_hdd_serial():
    global _hdd_serial
    if _hdd_serial is None:
        import subprocess
        import re
        try:
            hdd_name = get_hdd_name()
            p1 = subprocess.Popen(["udevadm", "info", "--query=all", "--name=/dev/{}".format(hdd_name)],
                                  stdout=subprocess.PIPE)
            p2 = subprocess.Popen(["grep", "ID_SERIAL_SHORT"], stdin=p1.stdout, stdout=subprocess.PIPE)
            p1.stdout.close()  # Allow p1 to receive a SIGPIPE if p2 exits.
            output = p2.communicate()
            _hdd_serial = re.search('ID_SERIAL_SHORT=(.+)\n', output[0].decode('utf8')).group(1)
        except Exception as ex:  # hide the exception
            # logger.exception(ex)
            pass
    return _hdd_serial


_sd_serial = None  # cache to minimize conflict between subprocess and mxnet


def get_sd_serial():
    global _sd_serial
    if _sd_serial is None:
        try:
            cid_filepath = "/sys/block/mmcblk0/device/cid"
            with open(cid_filepath, 'r') as f:
                cid = f.readline()
            _sd_serial = cid
        except Exception as ex:
            # logger.exception(ex)
            pass
    return _sd_serial


_dmidecode_uuid = None


def get_dmidecode_uuid():
    global _dmidecode_uuid
    if _dmidecode_uuid is None:
        import subprocess
        import re
        try:
            p1 = subprocess.Popen(["sudo", "dmidecode", "|", "grep", "UUID"], stdout=subprocess.PIPE)
            output = p1.communicate()
            _dmidecode_uuid = re.search('UUID: (.+)\n', output[0].decode('utf8')).group(1)
        except Exception as ex:  # hide the exception
            # logger.exception(ex)
            pass
    return _dmidecode_uuid


def get_computer_id(is_cloud=False):
    """ Conflict with mxnet (subprocess) in case using get_public_ip """
    try:
        if not is_cloud:
            serial = get_hdd_serial()
            if serial is None:
                serial = get_sd_serial()
            if serial is None:
                serial = ut.get_public_ip()
        else:
            serial = ut.get_public_ip()
        if serial is None:
            return ut.get_uuid()
        return serial
    except:
        return ut.get_uuid()


def encode(key, clear):
    enc = []
    for i in range(len(clear)):
        key_c = key[i % len(key)]
        enc_c = chr((ord(clear[i]) + ord(key_c)) % 256)
        enc.append(enc_c)

    bytes_or_str = "".join(enc)
    if isinstance(bytes_or_str, str):
        input_bytes = bytes_or_str.encode('utf8')
    else:
        input_bytes = bytes_or_str

    output_bytes = base64.urlsafe_b64encode(input_bytes)

    return output_bytes.decode('ascii')


def decode(key, enc):
    dec = []
    enc = base64.urlsafe_b64decode(enc).decode('utf8')

    for i in range(len(enc)):
        key_c = key[i % len(key)]
        dec_c = chr((256 + ord(enc[i]) - ord(key_c)) % 256)
        dec.append(dec_c)
    return "".join(dec)


def get_rtl_keys(is_cloud=False, serial=None):
    if serial is None:  # generate key
        random_hash_key = random.randint(0, 1048576)
        soft_id = int(hashlib.md5(get_computer_id(is_cloud).encode('utf8')).hexdigest(), 16) % 4096
        timestamp = int(time.time() / 300) % 256  # pass changes every 5 minutes
    else:  # parse key from serial
        random_hash_key = serial % 1048576
        soft_id = int((int(serial / 1048576) ^ random_hash_key) / 256)
        timestamp = (int(serial / 1048576) ^ random_hash_key) % 256

    key = (int(soft_id * 256 + timestamp) ^ random_hash_key) * 1048576 + random_hash_key
    return {
        'key': str(key),
        'hash': random_hash_key,
        'sid': soft_id,
        'ts': timestamp,
    }


def get_rtl_pass(rtl_keys=None, serial=None, secret='~!@#$%^&*()'):
    if rtl_keys is None and serial is not None:
        rtl_keys = get_rtl_keys(serial=serial)
    return hashlib.sha512('{}{}{}{}{}'.format(
        rtl_keys['ts'], rtl_keys['sid'], rtl_keys['hash'], secret, LICENSE_VER).encode('utf8')).hexdigest()[0:4]


def gen_rtl(input_pass, rtl_keys, limit_sec=15552000, maxi=1):
    """ limit_sec = 60 * 60 * 24 * 180 = 15552000s = 6 months """
    try:
        passwd = get_rtl_pass(rtl_keys)
        if input_pass != passwd:
            time.sleep(3)  # guard against brutal force attach
            return False
        else:
            rtl = Rtl(limit_second=limit_sec, maxi=maxi)
            rtl.flush_update()
            return True
    except:
        return False


g_license_hp = 0


def check_cloud_license(license_server='localhost', only_virtual=True):
    import requests
    import warnings
    from requests import ConnectionError
    global g_license_hp
    warnings.simplefilter('ignore', urllib.exceptions.SecurityWarning)
    try:
        if os.path.exists('/opt/intel/install.log'):
            return True  # evaluation version, not allow rtsp
        # cloud license check only apply to virtual and cloud computers
        if ut.is_virtual() or only_virtual is False:
            license_server_url = 'https://{}:9999/key'.format(license_server)
            cnt = 0
            while True:
                try:
                    logger.info('request cl')
                    serial = int(get_rtl_keys(is_cloud=True)['key'])
                    expected_pass = get_rtl_pass(serial=serial, secret='1G(m~Ym,0*X')
                    res = requests.post(
                        license_server_url, verify=False,
                        params={
                            'serial': serial, 'ip': ut.get_public_ip(),
                            'ip_lan': ut.get_lan_ip(), 'timestamp': LICENSE.timestamp,
                            'maxi': LICENSE.maxi})
                    if res.text == expected_pass:
                        logger.info('ok')
                        g_license_hp = CLOUD_LICENSE_MAX_HP  # when rtl is acquired successfully, reset license hp
                        return True  # exit loop if license obtained, else reduce HP and finally exit(1)
                except ConnectionError:
                    logger.error('ConnectionError')
                except Exception as ex:
                    logger.exception(ex)

                # retry 3 times (~3s)
                time.sleep(1)
                cnt += 1
                if cnt > 2:
                    g_license_hp -= 1  # when rtl failed to obtain, reduce license hp
                    logger.error('Fail to obtain rtl: remaining rtl HP: {}'.format(g_license_hp))
                    break
        else:
            g_license_hp = 1
    except Exception as ex:
        logger.exception(ex)

    if g_license_hp <= 0:
        logger.error('Fatal: License HP reached 0. Terminate application!')
        os.system('kill %d' % os.getpid())


class Rtl:  # Run time license
    def __init__(self, limit_second=0, software_id=None, path=None, maxi=1):
        if software_id is None:
            s = get_computer_id()
        else:
            s = software_id

        self.data = {
            "s": s,
            "l": limit_second,
            "e": 0,  # elapsed time
            "lu": time.time(),  # last update
            "fu": time.time(),  # first update
            "v": LICENSE_VER,
            "mi": maxi,
        }

        if path is not None:
            with open(path, "r") as f:
                self.data = json.loads(decode(RTL_ENCRYPT_KEY, f.read()))

    def flush_update(self):
        try:
            ut.mkdir_if_not_existed(PATH_RTL_DIR)
            data = encode(RTL_ENCRYPT_KEY, json.dumps(self.data))
            # write elapsed time to file to prevent clock manipulation
            # ensure write is atomic (license file wont be corrupted when program crash)
            if not os.path.isfile(LICENSE_TMP_FILE):
                with open(LICENSE_TMP_FILE, "w") as f:
                    f.write(data)
                    # make sure that all data is on disk
                    # see http://stackoverflow.com/questions/7433057/is-rename-without-fsync-safe
                    f.flush()
                    os.fsync(f.fileno())
            if os.path.isfile(PATH_RTL) and os.path.isfile(LICENSE_TMP_FILE):
                os.remove(PATH_RTL)
            if not os.path.isfile(PATH_RTL) and os.path.isfile(LICENSE_TMP_FILE):
                os.rename(LICENSE_TMP_FILE, PATH_RTL)
        except Exception as ex:
            # logger.exception(ex)
            pass

    def remaining_time(self):
        return self.data['l'] - self.data['e']

    def update(self):
        t = time.time()
        # protect against clock manipulation
        self.data['e'] += max(0, t - self.data['lu'])
        # protect against replacing current rtl with old rtl
        self.data['e'] = max((t - self.data['fu'], self.data['e']))
        self.data['lu'] = time.time()
        if UPDATE_LICENSE_FILE:
            self.flush_update()


class _LicenseService():
    def __init__(self, runtime_license=None, update_interval=LICENSE_UPDATE_INTERVAL_SEC):
        if runtime_license is None:
            self.load_license()
        else:
            self.runtime_license = runtime_license
        self.computer_id = get_computer_id()
        self.update_interval = update_interval
        self.stop_flag = False
        self.timestamp = int(time.time())

        self.thread = Thread(target=self.update_license, args=())
        self.thread.setDaemon(True)
        self.thread.start()

    def load_license(self):
        if os.path.isfile(PATH_RTL):
            try:
                self.runtime_license = Rtl(path=PATH_RTL)
            except:  # hide the exception
                self.runtime_license = None
        elif os.path.isfile(LICENSE_TMP_FILE):
            try:
                self.runtime_license = Rtl(path=LICENSE_TMP_FILE)
            except:  # hide the exception
                self.runtime_license = None
        else:
            self.runtime_license = None

    def update_license(self):
        cnt = 0
        while self.stop_flag is False and not global_def.g_shutdown:
            if cnt > self.update_interval:
                cnt = 0
                try:
                    if self.runtime_license is None:
                        self.load_license()

                    if self.runtime_license is not None:
                        try:
                            self.runtime_license.update()  # TODO: sua lai cho nay
                        except:
                            self.runtime_license = None

                    self.computer_id = get_computer_id()
                except:
                    pass
            else:
                cnt += 1

            time.sleep(1)

    def stop(self):
        self.stop_flag = True
        self.thread.join()

    def check(self):
        if self.runtime_license is None:
            return False
        return (self.runtime_license.data['s'] == self.computer_id) and \
            (self.runtime_license.remaining_time() > 0) and \
            (self.runtime_license.data.get('v', '') == LICENSE_VER)

    @property
    def maxi(self):
        if self.runtime_license:
            return self.runtime_license.data.get('mi', 1)
        else:
            return 1

    @property
    def maxf(self):
        if self.runtime_license:
            return self.runtime_license.data.get('mi', 1) * 100
        else:
            return 1


LICENSE = _LicenseService()
CLOUD_LICENSE_MAX_HP = 60 * 60 * 24 * 15  # 15 days
# ===============================================================================
# endregion Licensing
