import datetime
import os
import smtplib
import time
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from threading import Thread
from decouple import config
import cv2
from flask import Flask, render_template, Response, request, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, logout_user


from model import DemoModel

global capture, rec_frame, grey, switch, neg, face, rec, out
capture = 0
face = 0
switch = 1
rec = 0


# Load pretrained face detection model
model = DemoModel()

# instantiate flask app
app = Flask(__name__)

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


@login_manager.user_loader
def loader_user(user_id):
    return Users.query.get(user_id)


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


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        user = Users.query.filter_by(
            username=request.form.get("username")).first()
        if user.password == request.form.get("password"):
            login_user(user)
            return redirect(url_for("main_page"))
    return render_template("login.html")


@app.route("/logout")
def logout():
    logout_user()
    return redirect(url_for("login"))

# camera = cv2.VideoCapture("rtsp://admin:Facenet2022@192.168.1.3:554/Streaming/Channels/1")
camera = cv2.VideoCapture(0)


def draw_bounding_boxes(frame, results):
    for result in results:
        # Extract the bounding box coordinates
        boxes = result.boxes
        for box in boxes:
            # The box tensor contains [x_min, y_min, x_max, y_max, confidence, class]
            x_min, y_min, x_max, y_max = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            confidence = box[4]
            label = f"{int(box[5])} {confidence:.2f}"

            # Draw the rectangle
            frame = cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

            # Draw the label
            frame = cv2.putText(frame, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return frame

def send_email():
    print("Attempting to send email...")  # Add a print statement to indicate the function is called
    # Email configuration
    sender_email = "ducphongBKEU@gmail.com"  # Sender's email address
    receiver_email = "ducphongtester02@gmail.com"  # Receiver's email address
    password = config('EMAIL_PASSWORD')  # Sender's email password

    # Email content
    message = MIMEMultipart()
    message["From"] = sender_email
    message["To"] = receiver_email
    message["Subject"] = "Fall Detected!"

    body = "A fall has been detected. Please check the fall detection system for details."
    message.attach(MIMEText(body, "plain"))

    try:
        # Send email
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:  # SMTP server configuration
            server.login(sender_email, password)
            server.sendmail(sender_email, receiver_email, message.as_string())
        print("Email sent successfully!")  # Add a print statement to indicate successful email sending
    except Exception as e:
        print("Error sending email:", e)  # Add a print statement to print out the error message


# @app.route('/register', methods=['GET', 'POST'])
# def dang_ky():
#     if request.method == 'POST':
#         username = request.form['username']
#         password = request.form['password']
#         confirm_password = request.form['confirm_password']
#         email = request.form['email']
#
#         if password != confirm_password:
#             return "Mật khẩu không khớp!"
#         # Xử lý tệp tải lên
#         if 'file' in request.files:
#             files = request.files.getlist('file')
#             for file in files:
#                 if file:
#                     filename = file.filename
#                     filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#                     print(filepath)
#                     file.save(filepath)
#
#         # Ở đây bạn có thể thêm mã để lưu thông tin người dùng vào cơ sở dữ liệu
#
#         return redirect(url_for('main_page'))
#     return render_template('register.html')
def record(out):
    global rec_frame
    while rec:
        time.sleep(0.05)
        out.write(rec_frame)


def demo(frame):
    global model
    results = model.predict(frame)
    return frame, results


def gen_frames():  # generate frame by frame from camera
    global out, capture, rec_frame
    while True:
        success, frame = camera.read()
        if success:
            if face:
                frame, results = demo(frame)
                for result in results:
                    print(result.boxes)
                frame = draw_bounding_boxes(frame, results)
            if capture:
                capture = 0
                now = datetime.datetime.now()
                p = os.path.sep.join(['src/shots', "shot_{}.png".format(str(now).replace(":", ''))])
                cv2.imwrite(p, frame)
            if rec:
                rec_frame = frame
                frame = cv2.putText(cv2.flip(frame, 1), "Recording...", (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 4)
                frame = cv2.flip(frame, 1)

            try:
                ret, buffer = cv2.imencode('.jpg', cv2.flip(frame, 1))
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            except Exception as e:
                print("Exception {}".format(e))
        else:
            pass

@app.route('/')
def index():
    return render_template('admin.html')

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
        return redirect(url_for("login"))
    return render_template('add-member.html')


@app.route('/main')
def main_page():
    return render_template('admin.html')

#
# @app.route('/register')
# def dang_ky():
#     return render_template('register.html')


# @app.route('/login')
# def dang_nhap():
#     return render_template('login.html')


@app.route('/model')
def live():
    return render_template('live.html')


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/requests', methods=['POST', 'GET'])
def tasks():
    global switch, camera
    if request.method == 'POST':
        if request.form.get('click') == 'Capture':
            global capture
            capture = 1
        elif request.form.get('face') == 'Face':
            global face
            face = not face
            if face:
                time.sleep(4)
        elif request.form.get('stop') == 'Stop/Start':
            if switch == 1:
                switch = 0
                camera.release()
                cv2.destroyAllWindows()
            else:
                camera = cv2.VideoCapture(0)
                switch = 1
        elif request.form.get('rec') == 'Start/Stop Recording':
            global rec, out
            rec = not rec
            if rec:
                now = datetime.datetime.now()
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                out = cv2.VideoWriter('vid_{}.avi'.format(str(now).replace(":", '')), fourcc, 20.0, (640, 480))
                # Start new thread for recording the video
                thread = Thread(target=record, args=[out, ])
                thread.start()
            elif not rec:
                out.release()

    elif request.method == 'GET':
        return render_template('live.html')
    return render_template('live.html')


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)
