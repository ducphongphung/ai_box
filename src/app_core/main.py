from flask import Flask, render_template, Response, request, redirect, url_for, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, logout_user, current_user
import sys
from collections import deque
from shapely import geometry

# Change the path to folder ai_box
sys.path.append(r'C:\Users\ducph\PycharmProjects\ai_box')

from src.app_core.apps import VideoMonitorApp
from src.app_core.controller_utils import *
from src.app_core.conf import *
from src.utils.common import *
from src.cv_core.family.FamilyDetector import FamilyDetector
from src.cv_core.family.get_output import ObjPred
import traceback
import os
import cv2
import json
import time
import requests


logger = dbg.get_logger("tt_zone")

template_dir = os.path.abspath('templates')
static_dir = os.path.abspath('static')



global capture, rec_frame, grey, switch, neg, function, rec, out, previous_time
capture = 0
function = ''
switch = 1
rec = 0


# instantiate flask app
app = Flask(__name__, template_folder=template_dir, static_folder=static_dir)
app.secret_key = os.urandom(64)

############################################################################################
# Config image upload
UPLOAD_FOLDER = 'src/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Config image capture
CAPTURE_FOLDER = 'src/shots'
os.makedirs(CAPTURE_FOLDER, exist_ok=True)
app.config['CAPTURE_FOLDER'] = CAPTURE_FOLDER

############################################################################################
# Connect DB
# Tells flask-sqlalchemy what database to connect to
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///db.sqlite"
# Enter a secret key
app.config["SECRET_KEY"] = "ENTER YOUR SECRET KEY"
# Initialize flask-sqlalchemy extension
db = SQLAlchemy()

# LoginManager is needed for our application
# to be able to log in and out users
login_manager = LoginManager()
login_manager.init_app(app)


class Users(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(250), unique=True, nullable=False)
    password = db.Column(db.String(250), nullable=False)
    email = db.Column(db.String(250), unique=True, nullable=False)


db.init_app(app)

with app.app_context():
    db.create_all()

    ###########################################################################################
#


@login_manager.user_loader
def loader_user(user_id):
    return Users.query.get(user_id)

@app.route('/set_fire', methods=['POST'])
def set_fire():
    global function
    function = 'fire'
    return redirect(url_for('live'))

@app.route('/set_fall', methods=['POST'])
def set_fall():
    global function
    function = 'fall'
    return redirect(url_for('live'))

@app.route('/set_stranger', methods=['POST'])
def set_stranger():
    global function
    function = 'stranger'
    return redirect(url_for('live'))

@app.route('/register', methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form['username']
        password = request.form['password']
        confirm_password = request.form['confirm_password']
        email = request.form['email']

        if password != confirm_password:
            return "Mật khẩu không khớp!"
        user = Users(username=username,
                     password=password,
                     email = email)
        db.session.add(user)
        db.session.commit()

        # Xử lý tệp tải lên
        if 'file' in request.files:
            files = request.files.getlist('file')
            for file in files:
                if file:
                    filename = file.filename
                    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                    print(filepath)
                    file.save(filepath)
        return redirect(url_for("login"))
    return render_template("register.html")


@app.route("/", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        user = Users.query.filter_by(
            username=request.form.get("username")).first()
        if user.password == request.form.get("password"):
            login_user(user)
            return redirect(url_for("live"))
    return render_template("login.html")


@app.route("/logout")
def logout():
    logout_user()
    return redirect(url_for("login"))

@app.route('/add-member', methods=["GET", "POST"])
def add_member():
    if request.method == "POST":
        username = request.form['username']
        password = request.form['password']
        confirm_password = request.form['confirm_password']
        email = request.form['email']

        if password != confirm_password:
            return "Mật khẩu không khớp!"
        user = Users(username=username,
                     password=password,
                     email = email)
        db.session.add(user)
        db.session.commit()

        # Xử lý tệp tải lên
        if 'file' in request.files:
            files = request.files.getlist('file')
            for file in files:
                if file:
                    filename = file.filename
                    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                    print(filepath)
                    file.save(filepath)
            model = FamilyDetector()
            model.train_data()
        return redirect(url_for("login"))

    return render_template('add-member.html')


points = []

@app.route('/zone')
def zone():
    return render_template('zone.html')


@app.route('/draw_zone', methods=['GET', 'POST'])
def draw_zone():
    if request.method == 'GET':
        window_width = 1280
        window_height = 720
        global points
        global list_point
        list_point = {}
        file_path = 'setup/sc/source/camera_zone.json'

        try:
            with open(file_path, 'r') as file:
                data = json.load(file)

            for id_cam in data:
                url = data[id_cam]['cam_url']

                cam = cv2.VideoCapture(url)

                if not cam.isOpened():
                    return "Error: Could not open camera"

                ret, frame = cam.read()

                if not ret:
                    return "Error: Could not read frame from camera"

                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                frame_resized = cv2.resize(frame, (window_width, window_height))
                json_data = []
                for zone_type in ['zone_fall', 'zone_fire', 'zone_stranger']:
                    points = []

                    def draw(event, x, y, flags, param):
                        if event == cv2.EVENT_LBUTTONDOWN:
                            if len(points) < 4:
                                points.append((x, y))
                                cv2.circle(frame_resized, (x, y), 5, (0, 255, 0), -1)
                                if len(points) == 4:
                                    cv2.line(frame_resized, points[0], points[1], (255, 0, 0), 2)
                                    cv2.line(frame_resized, points[1], points[2], (255, 0, 0), 2)
                                    cv2.line(frame_resized, points[2], points[3], (255, 0, 0), 2)
                                    cv2.line(frame_resized, points[3], points[0], (255, 0, 0), 2)
                            cv2.imshow(zone_type, frame_resized)


                    cv2.namedWindow(zone_type, cv2.WINDOW_NORMAL)
                    cv2.resizeWindow(zone_type, window_width, window_height)
                    cv2.setMouseCallback(zone_type, draw)

                    while True:
                        cv2.imshow(zone_type, frame_resized)
                        k = cv2.waitKey(1) & 0xFF
                        if k == ord('q'):
                            points = []
                            break
                        if k == 27 or len(points) == 4:
                            break
                    
                    data_json = {}
                    data_json['zone_name'] = zone_type
                    data_json['coords'] = points
                    if zone_type == 'zone_fall':
                        data_json['zone_id'] = 0
                        data_json['zone_attributes'] = {'163':0, '164':1, '165':0}
                    elif zone_type == 'zone_fire':
                        data_json['zone_id'] = 1
                        data_json['zone_attributes'] = {'163':1, '164':0, '165':0}
                    elif zone_type == 'zone_stranger':
                        data_json['zone_id'] = 2
                        data_json['zone_attributes'] = {'163':0, '164':0, '165':1}
                    if len(points) == 4:
                        json_data.append(data_json)
                    # Sau khi hoàn tất việc vẽ các vùng cho một loại, đóng cửa sổ
                    cv2.destroyAllWindows()

            # Lưu lại toàn bộ dữ liệu JSON sau khi hoàn thành
            data[id_cam]['zone'] = (json_data)
            with open(file_path, 'w') as file:
                json.dump(data, file, indent=4)

            return redirect(url_for('live'))
        except Exception as e:
            print("An error occurred:", str(e))
            print(traceback.format_exc())
            return "An internal error occurred.", 500
    else:
        return "Use POST method to draw the zone."

def render_live(err=''):
    return render_template('live.html', current_user=current_user, err=err,
                           input_url=backend.input_url if backend.input_url else '',
                           conf_url=backend.conf_url, conf_json=json.dumps(backend.conf, indent=4),
                           sys_info=backend.sys_info())


@app.route('/live', methods=['GET', 'POST'])
def live():
    try:
        err = ''
        if request.method == 'POST':
            err = backend.update_input()
        return render_live(err=err)
    except Exception as ex:
        return render_live(err=str(ex))


def gen():
    while True:
        frame = backend.frame_stream
        if frame is not None:
            ret, encoded = cv2.imencode('.jpg', frame)
            frame_mjpeg = encoded.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_mjpeg + b'\r\n\r\n')

        time.sleep(0.1)  # broadcast at 10fps max


@app.route('/video_feed_mjpeg', methods=['GET'])
def video_feed_mjpeg():
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/add_cam_v2', methods=['POST'])
def add_cam_v2():
    try:
        camera_infos = read_json_conf()
        all_cams = request.json
        for infos in all_cams:
            username, password, ip_address, url = cam_url_to_info(infos['cam_url'], infos['cam_type'])
            cam_info = {
                "id": infos['id'],
                "cam_ip": infos.get('cam_ip', ''),
                "cam_name": infos.get('cam_name', ''),
                "cam_type": infos.get('cam_type', ''),
                "cam_url": url,
                "zone": infos.get('zone', [])
            }
            camera_infos[str(infos['id'])] = cam_info
            write_cam_info_json(info=camera_infos)
        return return_json("Thành công", data=camera_infos, ret_code=0)
    except Exception as ex:
        return return_json("Lỗi.", ex, ret_code=1)



@app.route('/camera-zone/<camera_ip>/<zone_id>', methods=['DELETE'])
def delete_zone_by_id(camera_ip, zone_id):
    try:
        camera_jsons = read_json_conf()
        if camera_ip not in camera_jsons:
            return return_json(f"Lỗi. Không tồn tại camera: ID : {camera_ip}", ret_code=2)

        zones = camera_jsons[camera_ip].get('zone', [])
        deleted = False

        # Tìm và xóa zone dựa trên zone_id
        for zone in zones:
            if zone.get('zone_id') == zone_id:
                zones.remove(zone)
                deleted = True
                break

        if deleted:
            write_cam_info_json(info=camera_jsons)
            return return_json(f"Thành công. Đã xóa zone có ID {zone_id}", data=camera_jsons[camera_ip], ret_code=0)
        else:
            return return_json(f"Lỗi. Không tìm thấy zone có ID {zone_id} trong camera {camera_ip}", ret_code=3)
    except Exception as ex:
        return return_json("Lỗi.", ex, ret_code=1)


@app.route('/camera-zone/<camera_id>', methods=['GET'])
def get_camera_zone(camera_id):
    try:
        camera_jsons = read_json_conf()
        if camera_id not in camera_jsons:
            return return_json(f"Lỗi. Không tồn tại camera: ID {camera_id}", ret_code=2)
        return return_json("Thành công", data=camera_jsons[camera_id], ret_code=0)
    except Exception as ex:
        return return_json("Lỗi.", ex, ret_code=1)


@app.route('/camera-info', methods=['GET'])
def get_camera_info():
    try:
        camera_info = list(read_json_conf().values())
        return return_json("Thành công", data={"camera_info": camera_info}, ret_code=0)
    except Exception as ex:
        return return_json("Lỗi.", ex, ret_code=1)


@app.route('/synch_cam_zone', methods=['POST'])
def synch_cam_zone():
    try:
        camera_infos = read_json_conf()
        all_cams = request.json
        for infos in all_cams:
            camera_infos[str(infos['id'])] = infos
            write_cam_info_json(info=camera_infos)
        return return_json("Thành công", data=camera_infos, ret_code=0)
    except Exception as ex:
        return return_json("Lỗi.", ex, ret_code=1)


@app.route('/camera-conf', methods=['POST'])
def conf_camera():
    try:
        camera_zone = read_json_conf()
        infos = request.json
        cam_id = infos['cam_id']
        camera_info = camera_zone.get(cam_id, {})
        cam_url = camera_info.get('cam_url')
        cam_ip = camera_info.get('cam_ip')
        if not cam_ip:
            return return_json("Lỗi. Không tìm thấy IP của camera", ret_code=4)
        # Kiểm tra ping đến IP của camera
        if not ping(cam_ip):
            return return_json("Lỗi. Không ping được camera IP", ret_code=2)

        err = backend.update_cam(cam_url, infos['conf_url'], cam_id)

        if err:
            return return_json("Lỗi. Không kích hoạt được camera.", ret_code=3)
        else:
            return return_json("Thành công", ret_code=0)
    except Exception as ex:
        return return_json("Lỗi. ", ex, ret_code=1)


class Backend(VideoMonitorApp):
    def __init__(self):
        super(Backend, self).__init__()
        self.zones_cache = {}
        self.msg_throttler.max_age = 5
        self.previous_time = None
        self.tracks = deque(maxlen=global_cfg.FALL_DET_WINDOW_SIZE)


    def get_detections(self, frame, function):
        detections = []
        reg = None
        try:
            rs = requests.post(
                'http://localhost:9769/api/detect',
                json={
                    'img_src': ut.html_img_src(frame),
                    'det_type': function
                }, verify=False)

            if rs.status_code == 200:
                reg = json.loads(rs.content)
        except requests.exceptions.ConnectionError:
            logger.error("Cannot connect to inference service")
        if reg:
            detections = reg['detections']

        return detections

    def send_events(self, key):
        try:
            send_email(key)
            self.tracks.clear()
        except Exception as ex:
            logger.exception(ex)

    def set_zone_cache(self):
        try:
            new_zones_cache = {}
            for zone in self.zones:
                zone_id = zone['zone_id']
                if zone_id in self.zones_cache:
                    new_zones_cache[zone_id] = self.zones_cache[zone_id]
                else:
                    new_zones_cache[zone_id] = zone.copy()
                    new_zones_cache[zone_id]['det_seq'] = deque(maxlen=global_cfg.HUMAN_DET_TIME_WINDOW_SIZE)

                self.zones_cache = new_zones_cache
        except Exception as ex:
            logger.exception(ex)

    def send_passing_events(self, key):
        try:
            self.send_events(key)
        except Exception as ex:
            logger.exception(ex)

    def send_event_not_in_zone(self,detections, key):
        for zone_id, zone in self.zones_cache.items():
            if zone_id == 0 and key == "fall":
                zone_poly, zone_poly_expanded = expand_zone(zone["coords"])
                zone_has_motion = False
                # center of body inside zone_poly_expanded -> zone has detection
                for bb in detections:
                    pts = [[bb['bb'][0], bb['bb'][1]], [bb['bb'][2], bb['bb'][1]], [bb['bb'][2], bb['bb'][3]],
                           [bb['bb'][0], bb['bb'][3]]]  # bb thành 4 điểm
                    if zone_poly.intersects(geometry.Polygon(pts)):
                        zone_has_motion = True

                try:
                    # for each zone, apply the rules to fire events to HC to turn on/off zone-switch
                    if not zone_has_motion:
                        self.send_passing_events(key)

                    else:  # keep current switch status if detection unstable: <= 33% detection
                        logger.warn(f"zone: {zone}: skip sending email as detection is not stable")
                except:
                    self.hc_connected = False
            elif zone_id == 1 and key == 'fire':
                zone_poly, zone_poly_expanded = expand_zone(zone["coords"])
                zone_has_motion = False
                # center of body inside zone_poly_expanded -> zone has detection
                for bb in detections:
                    # pts = [[bb[0], bb[1]], [bb[2], bb[1]], [bb[2], bb[3]], [bb[0], bb[3]]]  # bb to 4 points
                    pts = [[bb['bb'][0], bb['bb'][1]], [bb['bb'][2], bb['bb'][1]], [bb['bb'][2], bb['bb'][3]],
                           [bb['bb'][0], bb['bb'][3]]]  # bb thành 4 điểm
                    if zone_poly.intersects(geometry.Polygon(pts)):
                        zone_has_motion = True

                try:
                    # for each zone, apply the rules to fire events to HC to turn on/off zone-switch
                    if not zone_has_motion:
                        self.send_passing_events(key)

                    else:  # keep current switch status if detection unstable: <= 33% detection
                        logger.warn(f"zone: {zone}: skip sending email as detection is not stable")
                except:
                    self.hc_connected = False
            else:
                self.send_passing_events(key)

    def process_frame(self, frame, t0, regs, freeze_state):
        show = frame.copy()

        detections = self.get_detections(frame, function)


        if len(detections):
            for d in detections:
                if function == 'fall':
                    bb = d['bb']
                    dr.draw_box(show, bb, line1="FALLEN"  if d['is_fallen'] == 1 else None,
                                color=(0, 0, 255) if d['is_fallen'] == 1 else None, simple=True)
                    if d['is_fallen'] == 1:
                        self.tracks.append(1)
                elif function == 'fire':
                    bb = d['bb']
                    dr.draw_box(show, bb, line1="FIRE"  if d['is_fire'] == 1 else None,
                                color=(0, 0, 255) if d['is_fire'] == 1 else None, simple=True)
                    if d['is_fire'] == 1:
                        self.tracks.append(1)
                else:
                    if len(d['bbox_human']):
                        x1,x2,y1,y2 = d['bbox_human'][0], d['bbox_human'][1], d['bbox_human'][2], d['bbox_human'][3]
                        cv2.rectangle(show, (x1, y1), (x2, y2), (255,0,0), 2)
                        cv2.putText(show, 'person', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
                        if len(d['bbox_face']):
                            x1, x2, y1, y2 = d['bbox_face'][0], d['bbox_face'][1], d['bbox_face'][2], d['bbox_face'][3]
                            cv2.rectangle(show, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(show, 'Known', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        else:
                            self.tracks.append(1)
                        if d['stranger'] == 0:
                            self.tracks.append(1)

        current_time = time.time()
        # Default passing to 60s
        if self.previous_time is None:
            self.previous_time = current_time
        elif self.previous_time is not None and current_time - self.previous_time > 7 and sum(self.tracks) >= 30:
            self.previous_time = current_time
            self.send_event_not_in_zone(detections= detections ,key = function)
            self.tracks.clear()
        return show

    def send_email(key):
        print("Attempting to send email...")  # Add a print statement to indicate the function is called
        # Email configuration
        smtp_server = "smtp.gmail.com"
        port = 465
        email = "aiboxmail0@gmail.com"

        sender_email = email
        receiver_email = "ducphongBKEU@gmail.com"

        message_fall = """
        Subject: Fall Email
        """
        message_fire = """
        Subject: Fire Email
        """
        message_stranger = """
        Subject: Stranger Email
        """
        # Connect to the SMTP server and send the email
        try:
            with smtplib.SMTP_SSL(smtp_server, port) as server:
                # key application 3th
                server.login(email, "agfb qgra wvrb xpwo")
                if key == "fall":
                    server.sendmail(sender_email, receiver_email, message_fall)
                elif key == "fire":
                    server.sendmail(sender_email, receiver_email, message_fire)
                else:
                    server.sendmail(sender_email, receiver_email, message_stranger)

            print("Email sent successfully!")  # Add a print statement to indicate successful email sending
        except Exception as e:
            print("Error sending email:", e)  # Add a print statement to print out the error message

    @property
    def zones(self):
        # print(self.cam_id)
        rs = self.conf.get(self.cam_id, {}).get('zone', [])
        if not isinstance(rs, list):
            return []
        else:
            return rs

    def draw_static_info(self, disp):
        super(Backend, self).draw_static_info(disp)
        for zone in self.zones_cache.values():
            color = dr.RED if zone.get('light') else dr.WHITE
            dr.draw_poly(disp, zone['coords'], zone['zone_name'], color=color)

    def on_conf_update(self, frame):
        super(Backend, self).on_conf_update(frame)
        self.conf['count_margin'] = 0
        self.conf['stopline_y'] = 0
        self.conf = read_json_conf()
        self.set_zone_cache()

    def add_cli_opts(self):
        super(Backend, self).add_cli_opts()
        self.parser.add_option('--n_frame', type=int, default=20)

def expand_zone(coords, factor=0.3):
    xs = [i[0] for i in coords]
    ys = [i[1] for i in coords]
    x_center = 0.5 * min(xs) + 0.5 * max(xs)
    y_center = 0.5 * min(ys) + 0.5 * max(ys)
    min_corner = geometry.Point(min(xs), min(ys))
    center = geometry.Point(x_center, y_center)
    distance = center.distance(min_corner) * factor
    zone = geometry.Polygon(coords)
    zone_with_border = zone.buffer(distance)
    return zone, zone_with_border

if __name__ == '__main__':
    # if USE_TPU:
    #     scan_hc_thread = threading.Thread(target=rdhc.run_scan_hc_thread)
    #     scan_hc_thread.start()

    AppsConfig.configure(app)

    backend = Backend()

    backend.resume()

    app.config['APP_TITLE_SHORT'] = 'FTR'

    # Run your Flask application
    app.run(host='0.0.0.0', port=8081, debug=False, threaded=True)

