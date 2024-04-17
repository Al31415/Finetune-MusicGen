
from flask import Flask, request, send_file, render_template, Response
import numpy as np
import os
import io
import torch
import scipy.io.wavfile as wavfile
import torchaudio
# Create the Flask application

from audiocraft.data.audio import audio_write
#import IPython.display as ipd
from audiocraft.models import MusicGen
import numpy as np

musicgen = MusicGen.get_pretrained('finetune')
musicgen.set_generation_params(duration=16)
app = Flask(__name__)

@app.route('/')
def index():
    # Serve the HTML file for the front-end
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate_music():
    data = request.json
    description = data['description']
    wavs = musicgen.generate([description])
    waveform = wavs[0].cpu()
    torchaudio.save("C:\\Users\\34e65\\Downloads\\MusicGen\\output.wav", waveform, sample_rate=32000, format='wav')
        
    return send_file(
        'output.wav',
        as_attachment=True,
        download_name='output.wav',
        mimetype='audio/wav'
    )

if __name__ == '__main__':
    # Run the Flask application
    app.run(debug=True)