from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin

app = Flask(__name__)

# Configure the database
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///db.sqlite"
app.config["SECRET_KEY"] = "ENTER YOUR SECRET KEY"

# Initialize flask-sqlalchemy extension
db = SQLAlchemy(app)

# Initialize LoginManager
login_manager = LoginManager()
login_manager.init_app(app)

# Define the Users model
class Users(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(250), unique=True, nullable=False)
    password = db.Column(db.String(250), nullable=False)
    email = db.Column(db.String(250), unique=True, nullable=False)

# Create the database tables
with app.app_context():
    db.create_all()

    # Check database connection by performing a simple query
    try:
        # Attempt to query the database
        test_user = Users.query.first()
        if test_user:
            print("Database connected successfully. First user:", test_user.username)
        else:
            print("Database connected successfully. No users found.")
    except Exception as e:
        print("Database connection failed:", str(e))

if __name__ == "__main__":
    app.run(debug=True)