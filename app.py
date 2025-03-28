from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    # Placeholder for handling uploaded image
    return "Image uploaded successfully!"

if __name__ == '__main__':
    app.run(debug=True)
