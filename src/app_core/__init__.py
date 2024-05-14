from flask import Flask
from app_core import pages

def create_app():
    app = Flask(__name__)

    app.register_blueprint(pages.bp)
    return app