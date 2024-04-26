from flask import Flask, render_template, request
import base64
from tensorflow.keras.preprocessing.image import load_img, img_to_array, array_to_img
from tensorflow.keras.models import load_model
import numpy as np
from io import BytesIO
from PIL import UnidentifiedImageError

labels = {0: 'dew',
 1: 'fog smog',
 2: 'frost',
 3: 'glaze',
 4: 'hail',
 5: 'lightning',
 6: 'rain',
 7: 'rainbow',
 8: 'rime',
 9: 'sandstorm',
 10: 'snow'}

# Initialize Flask application
app = Flask(__name__)
allowed_extensions = ['.jpg', '.jpeg', '.png']
def preprocess_image(image):
    # image = image.resize((224, 224))
    # Rescale the image pixels
    image = img_to_array(image)/255.
    # Expand the dimensions to create a batch of size 1
    return np.expand_dims(image, axis=0)

# Define route for home page
@app.route('/')
def index():
    return render_template('index.html')

# Define route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Get the image file from the request
    file = request.files['image']
    # Check if the file has an allowed extension
    if not any(file.filename.lower().endswith(ext) for ext in allowed_extensions):
        allowed_extensions_str = ', '.join(allowed_extensions)
        return render_template('index.html', prediction=f'Error: Only {allowed_extensions_str} files are allowed')

    try:
        model = load_model('./model/FINAL_EfficientNetB7-201.h5')
        # Read the image file into memory
        image_stream = file.stream
        # Load the image from the stream
        image = load_img(BytesIO(image_stream.read()), target_size=(224, 224))
        # Preprocess the image
        image = preprocess_image(image)
        # Make predictions
        predicted_class = np.argmax(model.predict(image))
        # Convert image to base64 string
        buffered = BytesIO()
        array_to_img(image[0]).save(buffered, format="JPEG")
        image_data = base64.b64encode(buffered.getvalue()).decode("utf-8")

        # Return the result
        return render_template('index.html', prediction=(labels[predicted_class]).capitalize(), uploaded_image=image_data)
    except UnidentifiedImageError as e:
        return render_template('index.html', prediction=f'Error: Unable to identify the image file. {e}')


if __name__ == '__main__':
    app.run(debug=True)
