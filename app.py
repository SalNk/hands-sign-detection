from flask import Flask, render_template
from blueprints.save_image import save_image_bp
from blueprints.analyze import analyze_bp

app = Flask(__name__)

# Enregistrement des Blueprints
app.register_blueprint(save_image_bp, url_prefix='/save_image')
app.register_blueprint(analyze_bp, url_prefix='/analyze')

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == '__main__':
    app.run(debug=True)
