from flask import Flask, request, render_template_string
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from io import BytesIO
import base64

app = Flask(__name__)
model = tf.keras.models.load_model('brain_tumor_classifier.h5')
class_labels = ['glioma', 'meningioma', 'pituitary', 'notumor']
CONFIDENCE_THRESHOLD = 0.50

HTML = '''<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>BioScanAI</title>
  <style>
    * { margin: 0; padding: 0; box-sizing: border-box; }
    body { font-family: "Segoe UI", sans-serif; background: linear-gradient(135deg, #0f0c29, #302b63, #24243e); min-height: 100vh; display: flex; align-items: center; justify-content: center; color: white; }
    .card { background: rgba(255,255,255,0.07); backdrop-filter: blur(15px); border: 1px solid rgba(255,255,255,0.15); border-radius: 24px; padding: 50px 40px; width: 480px; text-align: center; box-shadow: 0 20px 60px rgba(0,0,0,0.4); }
    .logo { font-size: 42px; margin-bottom: 10px; }
    h1 { font-size: 28px; font-weight: 700; margin-bottom: 6px; }
    .subtitle { color: rgba(255,255,255,0.5); font-size: 14px; margin-bottom: 36px; }
    .upload-area { border: 2px dashed rgba(255,255,255,0.25); border-radius: 16px; padding: 36px 20px; margin-bottom: 24px; cursor: pointer; transition: 0.3s; }
    .upload-area:hover { border-color: #a78bfa; background: rgba(167,139,250,0.08); }
    .upload-area .icon { font-size: 40px; margin-bottom: 10px; }
    .upload-area p { color: rgba(255,255,255,0.5); font-size: 14px; }
    .upload-area strong { color: white; }
    input[type="file"] { display: none; }
    .filename { margin-top: 10px; font-size: 13px; color: #a78bfa; }
    button { width: 100%; padding: 14px; background: linear-gradient(135deg, #7c3aed, #a78bfa); color: white; border: none; border-radius: 12px; font-size: 16px; font-weight: 600; cursor: pointer; transition: 0.3s; }
    button:hover { transform: translateY(-2px); box-shadow: 0 8px 25px rgba(124,58,237,0.5); }
    .footer { margin-top: 24px; font-size: 12px; color: rgba(255,255,255,0.3); }
    .result-box { margin-top: 28px; padding: 20px; border-radius: 16px; font-size: 18px; font-weight: 600; }
    .tumor { background: rgba(239,68,68,0.2); border: 1px solid rgba(239,68,68,0.4); color: #fca5a5; }
    .no-tumor { background: rgba(34,197,94,0.2); border: 1px solid rgba(34,197,94,0.4); color: #86efac; }
    .preview { width: 100%; border-radius: 12px; margin-bottom: 16px; max-height: 200px; object-fit: cover; }
  </style>
</head>
<body>
  <div class="card">
    <div class="logo">🧠</div>
    <h1>BioScanAI</h1>
    <p class="subtitle">Brain Tumour Detection using Deep Learning</p>
    {% if result %}
      <img src="data:image/jpeg;base64,{{ img_data }}" class="preview">
      <div class="result-box {{ result_class }}">{{ result }}</div>
      <br>
      <a href="/"><button type="button">🔄 Scan Another</button></a>
    {% else %}
    <form action="/predict" method="POST" enctype="multipart/form-data">
      <div class="upload-area" onclick="document.getElementById('fileInput').click()">
        <div class="icon">📤</div>
        <p><strong>Click to upload</strong> MRI scan</p>
        <p>Supports JPG, PNG, JPEG</p>
        <div class="filename" id="fname">No file chosen</div>
      </div>
      <input type="file" id="fileInput" name="file" accept="image/*" onchange="document.getElementById('fname').innerText = this.files[0].name">
      <button type="submit">🔍 Analyse Scan</button>
    </form>
    {% endif %}
    <div class="footer">VGG16 Transfer Learning · 93% Accuracy · 4 Tumour Classes</div>
  </div>
</body>
</html>'''

@app.route('/')
def index():
    return render_template_string(HTML, result=None)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return 'No file uploaded!', 400
    file = request.files['file']
    file_bytes = file.read()
    img = image.load_img(BytesIO(file_bytes), target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    predictions = model.predict(img_array)
    predicted_probabilities = predictions[0]
    max_probability = np.max(predicted_probabilities)
    predicted_class_index = np.argmax(predicted_probabilities)
    predicted_class_label = class_labels[predicted_class_index]
    img_data = base64.b64encode(file_bytes).decode('utf-8')
    if predicted_class_label == 'notumor':
        result = f'✅ No Tumour Detected ({max_probability*100:.2f}% confidence)'
        result_class = 'no-tumor'
    elif max_probability < CONFIDENCE_THRESHOLD:
        result = f'⚠️ Unknown Tumour Type ({max_probability*100:.2f}% confidence)'
        result_class = 'tumor'
    else:
        result = f'🔴 {predicted_class_label.capitalize()} Tumour Detected ({max_probability*100:.2f}% confidence)'
        result_class = 'tumor'
    return render_template_string(HTML, result=result, result_class=result_class, img_data=img_data)

if __name__ == '__main__':
    app.run(debug=True)
