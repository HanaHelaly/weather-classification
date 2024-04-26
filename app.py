from flask import Flask, render_template, request
from base64 import b64encode
from PIL import Image
from tensorflow.keras.models import load_model
import numpy as np
from io import BytesIO
from PIL import UnidentifiedImageError
import os



destination = 'model/model.h5'

if os.path.exists('model/'):
    print("model found")
    pass
else:
    print("model not found, Downloding model.....")
    import gdown
    url = 'https://drive.google.com/uc?id=15sGlZvK-TcuSMqIvgT4OdguXC6aocMYr'
    os.makedirs('model', exist_ok=True)
    gdown.download(url, destination, quiet=False)



labels = {
    0: 'dew',
    1: 'fog smog',
    2: 'frost',
    3: 'glaze',
    4: 'hail',
    5: 'lightning',
    6: 'rain',
    7: 'rainbow',
    8: 'rime',
    9: 'sandstorm',
    10: 'snow'
}



app = Flask(__name__)

# Load the model once during application startup
model = load_model(destination)

def preprocess_image(image):
    image = image.resize((224, 224))
    image = np.array(image) / 255.
    return np.expand_dims(image, axis=0)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        file = request.files['image']
        if not any(file.filename.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png']):
            allowed_extensions_str = ', '.join(['.jpg', '.jpeg', '.png'])
            return render_template('index.html', prediction=f'Error: Only {allowed_extensions_str} files are allowed')

        image = Image.open(file)
        image_data = None  # Initialize image data

        try:
            image = preprocess_image(image)
            predicted_class = np.argmax(model.predict(image))
            buffered = BytesIO()
            Image.fromarray((image[0] * 255).astype(np.uint8)).save(buffered, format="JPEG")
            image_data = b64encode(buffered.getvalue()).decode("utf-8")
            return render_template('index.html', prediction=labels[predicted_class].capitalize(), uploaded_image=image_data)
        except UnidentifiedImageError as e:
            return render_template('index.html', prediction=f'Error: Unable to identify the image file. {e}')
        finally:
            file.close()  # Close file stream
            buffered.close()  # Close buffer

    except Exception as e:
        return render_template('index.html', prediction=f'Error: {e}')

if __name__ == '__main__':
    app.run(debug=True)
