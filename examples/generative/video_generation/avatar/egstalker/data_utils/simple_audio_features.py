"""
Simple audio feature extraction using Wav2Vec2.
No external dependencies beyond transformers, torch, and librosa.
"""

import argparse
import numpy as np
import torch
import librosa
from transformers import Wav2Vec2Processor, Wav2Vec2Model

def extract_audio_features(audio_path, output_path=None):
    """
    Extract audio features using Wav2Vec2.

    Args:
        audio_path: Path to input .wav file
        output_path: Path to output .npy file (default: replace .wav with .npy)

    Returns:
        features: numpy array of shape [T, D] where T is time steps, D is feature dimension
    """
    if output_path is None:
        output_path = audio_path.replace('.wav', '.npy')

    print(f'[INFO] Extracting audio features from {audio_path}')

    # Load audio
    audio, sr = librosa.load(audio_path, sr=16000)

    # Load Wav2Vec2 processor and model (base model, no fine-tuning needed)
    # Using facebook/wav2vec2-base as it's lightweight and doesn't require fine-tuning
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
    model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base").to(device)
    model.eval()

    # Process audio in chunks to avoid OOM
    chunk_size = 16000 * 30  # 30 seconds
    features_list = []

    for i in range(0, len(audio), chunk_size):
        chunk = audio[i:i + chunk_size]

        # Process audio
        inputs = processor(chunk, sampling_rate=16000, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Extract features
        with torch.no_grad():
            outputs = model(**inputs)

        # Get last hidden state: [batch, time, features]
        chunk_features = outputs.last_hidden_state.squeeze(0).cpu().numpy()
        features_list.append(chunk_features)

    # Concatenate all chunks
    features = np.concatenate(features_list, axis=0)  # [T, D]

    print(f'[INFO] Extracted features shape: {features.shape}')
    print(f'[INFO] Saving to {output_path}')

    # Save features
    np.save(output_path, features)

    return features


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract audio features using Wav2Vec2')
    parser.add_argument('--input', type=str, required=True, help='Path to input .wav file')
    parser.add_argument('--output', type=str, default=None, help='Path to output .npy file')

    args = parser.parse_args()

    extract_audio_features(args.input, args.output)
