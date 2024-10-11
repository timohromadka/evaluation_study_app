from flask import Flask, render_template, request, jsonify, send_file
import os
import subprocess
import time

app = Flask(__name__)

# Path to your model and inference script
MODEL_PATH = '/home/th716/evaluation_study_app/models/spotify_sleep_dataset/2048_128/model_step_90000'
INFERENCE_SCRIPT = 'scripts/inference_unet.py'

# Function to run the inference_unet.py script
def run_inference(batch_size):
    output_dir = os.path.join(MODEL_PATH, 'samples/audio/sch_ddim_nisteps_25')
    
    # Remove old samples if any
    if os.path.exists(output_dir):
        for f in os.listdir(output_dir):
            os.remove(os.path.join(output_dir, f))

    # Run scripts/inference_unet.py
    command = [
        'python', INFERENCE_SCRIPT, 
        '--pretrained_model_path', MODEL_PATH, 
        '--num_images', str(batch_size),
        '--num_inference_steps', '1000',
        '--eval_batch_size', str(batch_size)
    ]
    subprocess.run(command)

    # generate and return samples
    time.sleep(5)
    audio_files = [f for f in os.listdir(output_dir) if f.endswith('.wav')]
    return audio_files

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    batch_size = int(request.form['batch_size'])
    audio_files = run_inference(batch_size)
    
    return jsonify({'audio_files': audio_files})

@app.route('/audio/<filename>')
def download_audio(filename):
    file_path = os.path.join(MODEL_PATH, 'samples/audio/sch_ddim_nisteps_25', filename)
    return send_file(file_path, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
