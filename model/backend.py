from flask import Flask, request, jsonify
import os
from model_loader import summarize_video  # Your backend logic

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
OUTPUT_FOLDER = 'static/outputs'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

@app.route('/upload', methods=['POST'])
def upload_video():
    if 'file' not in request.files:
        return jsonify({'error': 'No file found'}), 400

    file = request.files['file']
    filename = file.filename
    upload_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(upload_path)

    # Trigger summarization
    output_name = f"summary_{filename}"
    output_path = os.path.join(OUTPUT_FOLDER, output_name)
    summarize_video(upload_path, output_path)

    return jsonify({
        'message': 'Upload and summarization successful',
        'video_path': upload_path,
        'summary_path': output_path
    })

if __name__ == '__main__':
    app.run(debug=True)
