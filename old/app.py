from flask import Flask, render_template, request, jsonify, send_file
import os
import subprocess
import time
import random

app = Flask(__name__)

# Base directory for datasets and models
MODEL_BASE_PATH = '/home/th716/evaluation_study_app/models'

# Function to run the inference_unet.py script
def run_inference(dataset, model, batch_size):
    model_path = os.path.join(MODEL_BASE_PATH, dataset, model)
    output_dir = os.path.join(model_path, 'samples/audio/sch_ddim_nisteps_25')

    # Remove old samples if any
    if os.path.exists(output_dir):
        for f in os.listdir(output_dir):
            os.remove(os.path.join(output_dir, f))

    # Run scripts/inference_unet.py
    command = [
        'python', 'scripts/inference_unet.py',
        '--pretrained_model_path', model_path,
        '--num_images', str(batch_size),
        '--num_inference_steps', '1000',
        '--eval_batch_size', str(batch_size)
    ]
    subprocess.run(command)

    # Generate and return samples
    time.sleep(5)
    audio_files = [f for f in os.listdir(output_dir) if f.endswith('.wav')]
    return audio_files

# Function to randomly select pre-generated samples
def get_pregen_samples(dataset, model, batch_size):
    audio_dir = os.path.join(MODEL_BASE_PATH, dataset, model, 'samples/audio')
    audio_files = [f for f in os.listdir(audio_dir) if f.endswith(('.wav', '.mp3'))]
    if len(audio_files) > batch_size:
        audio_files = random.sample(audio_files, batch_size)
    return audio_files

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/datasets', methods=['GET'])
def get_datasets():
    datasets = [d for d in os.listdir(MODEL_BASE_PATH) if os.path.isdir(os.path.join(MODEL_BASE_PATH, d))]
    return jsonify({'datasets': datasets})

@app.route('/models/<dataset>', methods=['GET'])
def get_models(dataset):
    dataset_path = os.path.join(MODEL_BASE_PATH, dataset)
    models = [m for m in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, m))]
    return jsonify({'models': models})

@app.route('/generate', methods=['POST'])
def generate():
    batch_size = int(request.form['batch_size'])
    model_id = request.form['model_id']
    dataset = request.form['dataset']
    mode = request.form['mode']  # Added mode to differentiate between inference and pregen samples

    if mode == 'pregen':
        audio_files = get_pregen_samples(dataset, model_id, batch_size)
        return jsonify({'audio_files': audio_files})
    else:
        audio_files = run_inference(dataset, model_id, batch_size)
        return jsonify({'audio_files': audio_files})

@app.route('/audio/<filename>')
def download_audio(filename):
    dataset = request.args.get('dataset')
    model_id = request.args.get('model_id')
    file_path = os.path.join(MODEL_BASE_PATH, dataset, model_id, 'samples/audio/sch_ddim_nisteps_25', filename)
    return send_file(file_path, as_attachment=True)

@app.route('/images/<filename>')
def download_image(filename):
    dataset = request.args.get('dataset')
    model_id = request.args.get('model_id')
    file_path = os.path.join(MODEL_BASE_PATH, dataset, model_id, 'samples/images', filename)
    return send_file(file_path, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
