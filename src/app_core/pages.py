from flask import Blueprint, render_template

bp = Blueprint("pages", __name__)

@bp.route("/")
def home():
    return render_template("admin.html")

@bp.route("/dangnhap")
def dangnhap():
    return render_template("dangnhap.html")

@bp.route("/dangky")
def dangky():
    return render_template("dangky.html")
