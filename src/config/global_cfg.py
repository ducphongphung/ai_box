USE_TPU = False
USE_TRACK = False
FACE_REG_TH = 0.5
FACE_REG_LOOSE_TH = 0.6
HUMAN_TRACK_KEEP_TH = 0.5
HUMAN_DET_TH = 0.5
# to cope with unstable detections, use temporal window which is a set of detections in n most recent past frames
HUMAN_DET_TIME_WINDOW_SIZE = 9
GESTURE_DET_WINDOW_SIZE = 9
FACE_GESTURE_DET_WINDOW_SIZE = 9
FALL_DET_WINDOW_SIZE = 64
FIRE_DET_WINDOW_SIZE = 64
SYS_PATH= "/home/quangthangggg/Documents/ai_box/setup/sc/source"

WEBAPP_CFG = {
    'BOOKMARKS': {
        'rtsp://admin:Facenet2022@192.168.1.7:554/Streaming/Channels/101;/home/quangthangggg/Documents/ai_box/setup/sc/source/camera_zone.json',
    },
    'COPYRIGHT': 'qualcomm.com',
    'LOGO': '/static/dist/img/logo-rang-dong.png',
    'DOMAIN': 'localhost',
}
