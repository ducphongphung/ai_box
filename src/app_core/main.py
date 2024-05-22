from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('admin.html')

@app.route('/main')
def main_page():
    return render_template('admin.html')

@app.route('/register')
def dang_ky():
    return render_template('register.html')

@app.route('/login')
def dang_nhap():
    return render_template('login.html')

if __name__ == '__main__':
    app.run(debug=True)
