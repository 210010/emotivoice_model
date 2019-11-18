import sys
sys.path.append('waveglow/')
import numpy as np
import torch

from hparams import create_hparams
from model import Tacotron2
from layers import TacotronSTFT, STFT
from audio_processing import griffin_lim
from train import load_model
from text import text_to_sequence
from denoiser import Denoiser

from utils import load_wav_to_torch
from scipy.io.wavfile import write
import matplotlib.pylab as plt
from plotting_utils import plot_spectrogram_to_numpy

def load_mel(path, stft, hparams):
    audio, sampling_rate = load_wav_to_torch(path)
    if sampling_rate != hparams.sampling_rate:
        raise ValueError("{} SR doesn't match target {} SR".format(
            sampling_rate, stft.sampling_rate))
    audio_norm = audio / hparams.max_wav_value
    audio_norm = audio_norm.unsqueeze(0)
    audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)
    melspec = stft.mel_spectrogram(audio_norm)
    melspec = melspec
    return melspec

def plot_data(fname, data, figsize=(16, 4)):
    plt.figure(figsize=figsize)
    plt.imsave(fname, data)

def generate_mels_by_ref_audio(model, waveglow, hparams, text, ref_wav, denoiser_strength=0.01, device=torch.device('cpu')):
    # Prepare ref audio input
    ref_audio_mel = load_mel(ref_wav, 
                TacotronSTFT(
                    hparams.filter_length, hparams.hop_length, hparams.win_length,
                    hparams.n_mel_channels, hparams.sampling_rate, hparams.mel_fmin,
                    hparams.mel_fmax), 
                hparams)

    # Decode text input and 
    mel_outputs, mel_outputs_postnet, _, alignments = model.inference_by_ref_audio(sequence, ref_audio_mel)

    # Plot results
    plot_data('mel.png', plot_spectrogram_to_numpy(mel_outputs.data.cpu().numpy()[0]))

    # Synthesize audio from spectrogram using WaveGlow
    with torch.no_grad():
        audio = waveglow.infer(mel_outputs_postnet, sigma=0.666, device=torch.device('cpu'))
    write("output.wav", hparams.sampling_rate, audio[0].data.cpu().numpy())

    # (Optional) Remove WaveGlow bias
    if denoiser_strength > 0:
        audio_denoised = denoiser(audio, strength=denoiser_strength)[:, 0]
        audio_denoised = audio_denoised * hparams.max_wav_value
        write("denoised_output.wav", hparams.sampling_rate, audio_denoised.squeeze().cpu().numpy().astype('int16'))

def generate_mels_by_sytle_tokens(model, waveglow, hparams, text, denoiser_strength=0.01, device=torch.device('cpu')):
    outputs_by_tokens = model.inference_by_style_tokens(sequence)

    for i, (mel_outputs, mel_outputs_postnet, _, alignments) in enumerate(outputs_by_tokens):
        # Plot results
        plot_data("mel_{}.png".format(i), plot_spectrogram_to_numpy(mel_outputs.data.cpu().numpy()[0]))

        # Synthesize audio from spectrogram using WaveGlow
        with torch.no_grad():
            audio = waveglow.infer(mel_outputs_postnet, sigma=0.666, device=device)
        write("output_{}.wav".format(i), hparams.sampling_rate, audio[0].data.cpu().numpy())

        # (Optional) Remove WaveGlow bias
        if denoiser_strength > 0:
            audio_denoised = denoiser(audio, strength=denoiser_strength)[:, 0]
            audio_denoised = audio_denoised * hparams.max_wav_value
            write("denoised_output_{}.wav".format(i), hparams.sampling_rate, audio_denoised.squeeze().cpu().numpy().astype('int16'))


if __name__=="__main__":
    device = torch.device('cpu')

    # Setup hparams
    hparams = create_hparams()

    # Load model from checkpoint
    checkpoint_path = "./outdir/1573832357/checkpoint_57500"
    model = load_model(hparams, device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device)['state_dict'])
    _ = model.eval()

    # Load WaveGlow for mel2audio synthesis and denoiser
    waveglow_path = './waveglow/waveglow_170000_22k'
    waveglow = torch.load(waveglow_path, map_location=device)['model']
    waveglow.eval()
    denoiser = Denoiser(waveglow, device=device)

    # Prepare text input
    text = "이것은 감정을 담아 말하는 음성합성기 입니다."
    sequence = np.array(text_to_sequence(text, ['korean_cleaners']))[None, :]
    sequence = torch.autograd.Variable(torch.from_numpy(sequence)).long()

    # Generate outfiles
    ref_wav = "/data/emotiontts_new/04.Emotion/ema/wav_22k/ema00001.wav"
    generate_mels_by_ref_audio(model, waveglow, hparams, text, ref_wav, denoiser_strength=0)
    generate_mels_by_sytle_tokens(model, waveglow, hparams, text, denoiser_strength=0)
