from src.utils.types_ex import Roi
import src.utils.common as ut

import cv2
import numpy as np
import hashlib
import random as rd

from PIL import Image, ImageFont, ImageDraw

STR_UNKNOWN = "Unknown"

# ===============================================================================
# Resources
# ===============================================================================
BLUE = (255, 0, 0)
VIOLET = (226, 43, 138)
RED = (0, 0, 255)
GREEN = (0, 255, 0)
YELLOW = (0, 255, 255)
CYAN = (255, 255, 100)
ORANGE = (0, 165, 255)
MAGENTA = (255, 0, 255)
GRAY = (200, 200, 200)
WHITE = (255, 255, 255)

deeppink = (147, 20, 255)
deepskyblue = (255, 191, 0)
brown = (42, 42, 165)
salmon = (160, 160, 255)
yellowgreen = (50, 205, 154)
olivedrab = (35, 142, 107)

g_colors = [GREEN, BLUE, ORANGE, YELLOW, CYAN, VIOLET, MAGENTA,
            deeppink, deepskyblue, brown, salmon, yellowgreen, olivedrab]

# g_font18 = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)
# g_font24 = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 24)
#
# g_font16 = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
# g_font22 = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 22)
#
# g_font30 = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 30)
# g_font36 = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 36)
# g_font54 = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 54)


# ===============================================================================
# Text content building
# ===============================================================================
def get_topk_match_string(match, k_id):
    """
    match is a list of top-k (name, confidence) which is result of matching a face emb to the database,
    this function returns the text for k-th highest confidence tuple in the top-k list.
    Example: match = [('peter',0.5), ('daisy', 0.9)], k_id = 1 -> 'daisy 0.9'
    """
    if len(match) > k_id:
        print('draw match: {}, len: {}'.format(match, len(match)))
        name = match[k_id][0]
        confidence = match[k_id][1]
        return '{} {}'.format(name, confidence)
    return ''


def get_title_string(best_match_dist, track_score):
    s = ""
    if best_match_dist is not None:
        if best_match_dist < 2:  # raw value? convert to (0, 100)
            best_match_dist = int(100 * best_match_dist)
        s = "s{}".format(best_match_dist)
    c = ""
    if track_score is not None:
        c = "c{}".format(track_score)
    return "{}{}".format(s, c)


def get_id_color(identity, face_db=None):
    color = GRAY
    try:
        if not identity or identity == STR_UNKNOWN:
            color = GRAY
        else:
            if face_db is not None:
                if identity in face_db.color_dict:
                    color_idx = face_db.color_dict[identity]
                else:
                    color_idx = 0
            else:
                color_idx = hash_color(identity)

            color = g_colors[color_idx]
    except:
        pass

    return color


def hash_color(name):
    try:
        return (int(hashlib.sha1(name.encode('utf8')).hexdigest(), 16) % (10 ** 8)) % 12
    except:
        return 0


def get_id_visuals(identity, best_match_dist, track_score, face_db):
    name = identity.replace('_', ' ')

    if face_db is not None:
        if identity in face_db.utf_name_dict:
            if face_db.utf_name_dict[identity] is not None:
                name = face_db.utf_name_dict[identity]

        title = get_title_string(best_match_dist, track_score)
        if identity in face_db.utf_title_dict:
            if face_db.utf_title_dict[identity] is not None:
                title = face_db.utf_title_dict[identity] + " " + title

        color = get_id_color(identity, face_db)

        return name, title, color
    else:
        return name, "", GREEN


# ===============================================================================
# Plain, simple bounding box and text rendering
# ===============================================================================
def draw_text(img, text, origin, color=WHITE, font_scale=1, thickness=1, shadow=0):
    origin = (int(origin[0]), int(origin[1]))
    thickness = max(1, int(thickness))
    if shadow:
        cv2.putText(img, text, origin, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), int(thickness + shadow))
    cv2.putText(img, text, origin, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)


def draw_face(img, face_id, bounding_boxes, points=None, matches=None, top_k=1, color=GREEN):
    face_position = bounding_boxes[face_id].astype(int)
    cv2.rectangle(img, (face_position[0], face_position[1]), (face_position[2], face_position[3]), color, 2)
    if not points is None:
        ps = points[:, face_id]
        if len(ps) > 4:
            for i in range(5):
                cv2.circle(img, (ps[i], ps[i + 5]), 3, g_colors[i])

    if matches is not None:
        for i in range(top_k):
            line = get_topk_match_string(matches[face_id], i)
            draw_text(img, line, (face_position[0], face_position[1] - 20 * (i + 1)))


def draw_faces(img, bounding_boxes, points=None, matches=None, top_k=1):
    for face_id in range(len(bounding_boxes)):
        draw_face(img, face_id, bounding_boxes, points, matches, top_k)


def gen_captcha(len=4):
    key = "{}#aZ0~bM1".format(rd.randint(10 ** (len - 1), 10 ** len - 1))
    hash = hashlib.md5(key.encode('utf8')).hexdigest()
    img = 255 * np.ones((60, len * 20 + 40, 3), dtype=np.uint8)
    col = (0, 0, 0)
    draw_text(img, key[:len], (20, 40), color=col)
    cv2.line(img, (0, 30), (img.shape[1], 30), color=col)
    cv2.line(img, (0, 25), (img.shape[1], 25), color=col)
    cv2.line(img, (0, 35), (img.shape[1], 35), color=col)
    return hash, img


def check_captcha(input_str, hash):
    return hashlib.md5("{}#aZ0~bM1".format(input_str).encode('utf8')).hexdigest() == hash


# ===============================================================================
# Unicode rendering
# ===============================================================================
def draw_unicode(img, unicode_text, origin, rgb_color=WHITE, text_size=18, max_w=500, shadow=0):
    left = max(0, int(origin[0]))
    top = max(0, int(origin[1]))
    bottom = min(img.shape[0], top + 80)
    right = min(img.shape[1], left + max_w)
    rs = ut.clip_bb([left, top, right, bottom], img.shape[1], img.shape[0])
    if rs is None:
        return
    left, top, right, bottom = rs
    crop = img[top:bottom, left:right, :]

    pil_im = Image.fromarray(crop)  # opencv Mat -> PIL image
    draw = ImageDraw.Draw(pil_im)

    g_font = ImageFont.truetype("arial.ttf", 18)

    # if text_size == 16:
    #     g_font = g_font16
    #
    # if text_size == 22:
    #     g_font = g_font22
    #
    # if text_size == 24:
    #     g_font = g_font24
    #
    # if text_size == 30:
    #     g_font = g_font30
    #
    # if text_size == 36:
    #     g_font = g_font36
    #
    # if text_size == 54:
    #     g_font = g_font54

    if shadow:
        draw.text((shadow, shadow), unicode_text, (0, 0, 0), g_font)
        draw.text((0, 0), unicode_text, (0, 0, 0), g_font)
        draw.text((0, shadow), unicode_text, (0, 0, 0), g_font)
        draw.text((shadow, 0), unicode_text, (0, 0, 0), g_font)
    draw.text((shadow / 2, shadow / 2), unicode_text, rgb_color, g_font)
    crop = np.array(pil_im)  # PIL image -> opencv Mat
    img[top:bottom, left:right, :] = crop


def draw_unicode_kerning(unicode_text, pil_font, font_size, kerning=0):
    ts = font_size
    w = int(len(unicode_text) * ts)
    h = int(ts * 1.35)
    img = np.ones((h, w, 3), dtype=np.uint8) * 255
    pil_im = Image.fromarray(img)  # opencv Mat -> PIL image
    draw = ImageDraw.Draw(pil_im)

    p = 0
    for c in unicode_text:
        w, h = draw.textsize(c, font=pil_font)
        draw.text((p, 0), c, (0, 0, 0), pil_font)
        p += (w + kerning)
    img = np.array(pil_im)  # PIL image -> opencv Mat
    img = img[:, :p + 1, :]  # crop
    return img


# ===============================================================================
# Stylish bounding box + name, id...
# ===============================================================================
def draw_aim_box(img, pt1, pt2, color=GREEN):
    thickness = 3

    p0 = (pt1[0] + int((pt2[0] - pt1[0]) / 2), pt1[1] + int((pt2[1] - pt1[1]) / 2))
    lengthofside = np.amax([pt2[0] - pt1[0], pt2[1] - pt1[1]])
    tipLength = int(lengthofside * 0.2)
    p1 = (p0[0] - int(lengthofside / 2), p0[1] - int(lengthofside / 2))
    p2 = (p0[0] + int(lengthofside / 2), p0[1] - int(lengthofside / 2))
    p3 = (p0[0] + int(lengthofside / 2), p0[1] + int(lengthofside / 2))
    p4 = (p0[0] - int(lengthofside / 2), p0[1] + int(lengthofside / 2))

    cv2.rectangle(img, p1, p3, color, 1)

    cv2.line(img, p1, (p1[0] + tipLength, p1[1]), color, thickness)
    cv2.line(img, p1, (p1[0], p1[1] + tipLength), color, thickness)
    cv2.line(img, (p2[0] - tipLength, p2[1]), p2, color, thickness)
    cv2.line(img, p2, (p2[0], p2[1] + tipLength), color, thickness)
    cv2.line(img, (p3[0], p3[1] - tipLength), p3, color, thickness)
    cv2.line(img, p3, (p3[0] - tipLength, p3[1]), color, thickness)
    cv2.line(img, (p4[0] + tipLength, p4[1]), p4, color, thickness)
    cv2.line(img, p4, (p4[0], p4[1] - tipLength), color, thickness)

    cv2.line(img, (p0[0], p1[1] - int(lengthofside * 0.07)), (p0[0], p1[1] + int(lengthofside * 0.07)), color, 1)
    cv2.line(img, (p2[0] - int(lengthofside * 0.07), p0[1]), (p2[0] + int(lengthofside * 0.07), p0[1]), color, 1)
    cv2.line(img, (p0[0], p3[1] - int(lengthofside * 0.07)), (p0[0], p3[1] + int(lengthofside * 0.07)), color, 1)
    cv2.line(img, (p4[0] - int(lengthofside * 0.07), p0[1]), (p4[0] + int(lengthofside * 0.07), p0[1]), color, 1)


def draw_box(cv_img, bb, line1=None, line2=None, color=None, text_size=30, simple=False, thickness=2):
    if len(bb) == 4:
        pt1 = (int(bb[0]), int(bb[1]))
        pt2 = (int(bb[2]), int(bb[3]))
    elif len(bb) == 2:
        pt1 = bb[0]
        pt2 = bb[1]
    if color is None:
        color = get_id_color(line1)
    if simple is False:
        draw_aim_box(cv_img, pt1, pt2, color)
    else:
        cv2.rectangle(cv_img, pt1, pt2, color, thickness=thickness)
    if line1:
        draw_unicode(cv_img, str(line1), (pt1[0] - 40, pt1[1] - (5 + text_size) * 2), color, text_size)
    if line2:
        draw_unicode(cv_img, str(line2), (pt1[0] - 40, pt1[1] - (5 + text_size)), color, text_size)

def draw_aim_face(cv_img, bb, name='', title='', color=GREEN):
    draw_box(cv_img, bb, name, title, color)


# ===============================================================================
# No bounding box, no text, profile image attached to one side of face
# ===============================================================================
def get_profile_cell(w, h, profile_image, border_thickness=3):
    if profile_image is None:
        return None

    rs = np.ones((h, w, 3), np.uint8) * 255
    # set aside 3 pixel outermost border
    rs[border_thickness:h - border_thickness, border_thickness:w - border_thickness, :] = (137, 135, 38)
    profile_size = min(w, h) * 8 // 10
    offset_x = (w - profile_size) // 2
    offset_y = (h - profile_size) // 2
    # make 3 pixel profile picture border
    rs[offset_y - border_thickness:(offset_y + profile_size) + border_thickness,
    offset_x - border_thickness:(offset_x + profile_size) + 3, :] = (255, 255, 255)
    rs[offset_y:(offset_y + profile_size), offset_x:(offset_x + profile_size), :] = cv2.resize(profile_image, (
        profile_size, profile_size))
    return rs


def draw_floating_profile_image(cv_img, bb, profile_image):
    if profile_image is None:
        return

    os = min(100, cv_img.shape[1] // 4)
    ih, iw, ic = cv_img.shape
    overlay = get_profile_cell(os * 2, os * 2, profile_image)
    fc = ((bb[0] + bb[2]) // 2, (bb[1] + bb[3]) // 2)
    fs = (bb[2] - bb[0]) // 2
    if (cv_img.shape[1] - fc[0]) > fc[0]:
        oc = (fc[0] + fs + os, fc[1])
        tc = (fc[0] + fs * 0.8, fc[1])
    else:
        oc = (fc[0] - fs - os, fc[1])
        tc = (fc[0] - fs * 0.8, fc[1])

    obb = [oc[0] - os, oc[1] - os, oc[0] + os, oc[1] + os]  # obb = overlay bb

    # intersection box clipped by input image
    i_l = min(iw, max(0, obb[0]))
    i_t = min(ih, max(0, obb[1]))
    i_r = min(iw, max(0, obb[2]))
    i_b = min(ih, max(0, obb[3]))

    # intersection box in overlay image coordinate system
    o_l = int(i_l - obb[0])
    o_t = int(i_t - obb[1])
    o_r = int(i_r - obb[0])
    o_b = int(i_b - obb[1])

    # link faces with a tip
    cv2.fillConvexPoly(cv_img, np.array([(oc[0], oc[1] - os // 2), (oc[0], oc[1] + os // 2), tc], np.int32), WHITE)

    cv_img[i_t:i_b, i_l:i_r, :] = overlay[o_t:o_b, o_l:o_r, :]


def draw_face_for_eval(cv_img, bb, identity, face_db):
    profile_img = face_db.get_profile_image(identity)
    draw_floating_profile_image(cv_img, bb, profile_img)


def get_profile_image(uncropped, face_bb, profile_img_size=320):
    tl = (face_bb[0], face_bb[1])
    br = (face_bb[2], face_bb[3])

    w = uncropped.shape[1]
    h = uncropped.shape[0]
    roi_w = br[0] - tl[0]
    roi_h = br[1] - tl[1]
    marginx = int(0.65 * roi_w)
    marginy = int(0.65 * roi_h)

    c_x = int((br[0] + tl[0]) // 2)
    c_y = int((br[1] + tl[1]) // 2)

    tl_x = tl[0] - marginx
    tl_y = tl[1] - marginy
    br_x = br[0] + marginx
    br_y = br[1] + marginy

    tl_x = max(0, tl_x)
    tl_y = max(0, tl_y)
    br_x = min(w, br_x)
    br_y = min(h, br_y)

    roi_w = br_x - tl_x
    roi_h = br_y - tl_y
    r = int(min(roi_w, roi_h) / 2)

    tl_x = max(0, c_x - r)
    tl_y = max(0, c_y - r)
    br_x = min(w, c_x + r)
    br_y = min(h, c_y + r)

    tl = (tl_x, tl_y)
    br = (br_x, br_y)

    profile_img = uncropped[tl[1]:br[1], tl[0]:br[0], :]

    return cv2.resize(profile_img, (profile_img_size, profile_img_size),
                      interpolation=cv2.INTER_LINEAR)


def draw_trace(cv_img, pts=None, bboxes=None, col=None):
    if pts is None and bboxes is not None:
        pts = [(int(ut.center(bb)[0]), int(ut.center(bb)[1])) for bb in bboxes]
    if pts is None or len(pts) < 2:
        return
    if col is None:
        col = GREEN
    for i in range(len(pts) - 1):
        if i % 2 == 0:
            cv2.line(cv_img, pts[i], pts[i + 1], col)


# ===============================================================================
# High level general face draw function
# ===============================================================================
def draw_face2(cv_img, bb, identity, face_db, best_match_dist=None, track_score=None, color=None, lines=None):
    if not identity:
        identity = STR_UNKNOWN
    name, title, id_color = get_id_visuals(identity, best_match_dist, track_score, face_db)
    if color is None:
        color = id_color
    if lines is not None:
        if len(lines) > 0:
            title = lines[0]
        if len(lines) > 1:
            name = lines[1]
    draw_aim_face(cv_img, bb, name, title, color)
    if name != STR_UNKNOWN:
        profile_img = face_db.get_profile_image(identity)
        draw_floating_profile_image(cv_img, bb, profile_img)


def draw_face3(cv_img, face_track, face_db, show_trace=False, show_prop=False, loc_dict=None, debug=True):
    identity = face_track.identity
    best_match_dist = int(face_track.spot_dist * 100)
    track_score = face_track.identity_age
    bb = face_track.bb
    trace = face_track.trace
    wait = int(face_track.identity_wait_sec)

    properties = ''
    tag = face_track.det.tag

    rf_score = tag.get('liveliness', None)
    if rf_score is not None:
        properties += "r{}".format(int(100 * rf_score))
    gender = tag.get('gender', None)
    if gender is not None:
        properties += gender
    age = face_track.avg_tag_value('age')
    if age is not None:
        properties += str(int(age))

    name, title, id_color = get_id_visuals(identity, best_match_dist, track_score, face_db)
    if debug:
        title = get_title_string(best_match_dist, track_score) + "w{}".format(wait)
    else:
        title = ''

    if identity == STR_UNKNOWN:
        bb = Roi(bbox=bb).resize(1.3).bbox
        if loc_dict:
            if STR_UNKNOWN in loc_dict:
                name = loc_dict[STR_UNKNOWN]

    if show_trace:
        draw_trace(cv_img, trace)
    if show_prop:
        title += properties

    draw_box(cv_img, bb, title, name, id_color, 54)
    if identity != STR_UNKNOWN:
        profile_img = face_db.get_profile_image(identity)
        draw_floating_profile_image(cv_img, bb, profile_img)


def draw_face_props(cv_img, track, face_db):
    name, title, id_color = get_id_visuals(track.identity, None, None, face_db)
    tag = track.tag
    properties = ''
    if tag.get('liveliness', 1) < 0.5:
        id_color = RED

    gender = tag.get('gender', None)
    if gender is not None:
        properties += gender + ' '
    age = tag.get('age', None)
    if age is not None:
        properties += 'age: {} '.format(int(age))
    emotion = tag.get('emotion_top', None)
    if emotion:
        properties += emotion + ' '

    draw_aim_face(cv_img, track.bb, name, properties, id_color)
    if name != STR_UNKNOWN:
        profile_img = face_db.get_profile_image(track.identity)
        draw_floating_profile_image(cv_img, track.bb, profile_img)


def draw_poly(cv_img, pts, label, color=RED, text_size=36):
    """
    @param pts: [1,2,3,4,5,6] -> (1,2), (3,4), (5,6)
    """
    npts = np.array(pts).astype(int)
    npts = npts.reshape(-1, 2)
    if len(npts) == 2:  # is bbox
        draw_box(cv_img, npts, label, None, color, text_size, simple=True)
    else:
        draw_text(cv_img, str(label), (npts[0][0] + 5, npts[0][1] - (5 + text_size)), color, thickness=2, shadow=5)
        cv2.drawContours(cv_img, [npts], 0, color, thickness=2)
