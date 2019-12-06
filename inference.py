import sys
import numpy as np
import torch

from hparams import create_hparams
from model_tacotron2 import Tacotron2
from layers import TacotronSTFT, STFT
from audio_processing import griffin_lim
from train import load_Tacotron2
from text import text_to_sequence

from utils import load_wav_to_torch
from scipy.io.wavfile import write
import matplotlib.pylab as plt
from plotting_utils import plot_spectrogram_to_numpy

import argparse

def load_mel(path, stft, hparams, device=torch.device("cpu")):
    audio, sampling_rate = load_wav_to_torch(path)
    if sampling_rate != hparams.sampling_rate:
        raise ValueError("{} SR doesn't match target {} SR".format(
            sampling_rate, stft.sampling_rate))
    audio_norm = audio / hparams.max_wav_value
    audio_norm = audio_norm.unsqueeze(0)
    audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)
    melspec = stft.mel_spectrogram(audio_norm)
    melspec = melspec.to(device)
    return melspec

def plot_data(fname, data, figsize=(16, 4)):
    plt.figure(figsize=figsize)
    plt.imsave(fname, data)

def generate_mels_by_ref_audio(model, waveglow, hparams, text, ref_wav, denoiser_strength=0.01, device=torch.device('cpu'), *, outpath='output.wav'):
    # Prepare ref audio input
    ref_audio_mel = load_mel(ref_wav, 
                TacotronSTFT(
                    hparams.filter_length, hparams.hop_length, hparams.win_length,
                    hparams.n_mel_channels, hparams.sampling_rate, hparams.mel_fmin,
                    hparams.mel_fmax), 
                hparams, device)

    # Decode text input and 
    mel_outputs, mel_outputs_postnet, _, alignments = model.inference_by_ref_audio(sequence, ref_audio_mel)

    # Plot results
    plot_data('mel.png', plot_spectrogram_to_numpy(mel_outputs.data.cpu().numpy()[0]))

    # Synthesize audio from spectrogram using WaveGlow
    with torch.no_grad():
        audio = waveglow.infer(mel_outputs_postnet, sigma=0.666)
    write(outpath, hparams.sampling_rate, audio[0].data.cpu().numpy())

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
    import time
    start = time.time()

    device = torch.device('cpu')

    # Setup hparams
    hparams = create_hparams()

    # Load model from checkpoint
    checkpoint_path = "./outdir/4/checkpoint_57500"
    model, _ = load_Tacotron2(hparams, device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device)['state_dict'])
    _ = model.eval()

    # Load WaveGlow for mel2audio synthesis
    if device == torch.device('cuda'):
        sys.path.insert(0, "waveglow/") # To look glow(original version) first
        from waveglow.denoiser import Denoiser
    else:
        sys.path.insert(0, "waveglow_cpu_components/") # To look glow(cpu version) first
        from waveglow_cpu_components.denoiser import Denoiser
    waveglow_path = './waveglow/waveglow_170000_22k'
    waveglow = torch.load(waveglow_path, map_location=device)['model']
    waveglow.eval()
    denoiser = Denoiser(waveglow).to(device)


    # CLI setup
    default_text = "이것은 감정을 담아 말하는 음성합성기 입니다."
    default_ref = "/home/tts_team/ai_workspace/data/emotiontts_new/04.Emotion/ema/wav_22k/ema00350.wav"
    default_out = "output.wav"

    # arguments parser settup
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--ref-wav', default=default_ref)
    group.add_argument('--predef-style')
    parser.add_argument('--text', default=default_text)
    parser.add_argument('--out', default=default_out)
    args = parser.parse_args()

    if args.predef_style:
        style, _idx = args.predef_style.split('_')
        idx = int(_idx)
        if style == 'neutral': # a
            idx += 0
            speaker = "ema"
        elif style == 'happy': # c
            idx += 100
            speaker = "emc"
        elif style == 'sad': # d
            idx += 200
            speaker = "emd"
        elif style == 'angry': # e
            idx += 300
            speaker = "eme"
        else:
            raise ValueError(f'invalid style {style}')
            
        ref_wav = f'/home/tts_team/ai_workspace/data/emotiontts_new/04.Emotion/{speaker}/wav_22k/{speaker}00{idx+1:03}.wav'
    else:
        ref_wav = args.ref_wav

    sequence = np.array(text_to_sequence(args.text, ['korean_cleaners']))[None, :]
    sequence = torch.autograd.Variable(torch.from_numpy(sequence)).to(device).long()

    # Generate outfiles
    generate_mels_by_ref_audio(model, waveglow, hparams, args.text, ref_wav, denoiser_strength=0, device=device, outpath=args.out)

    if False:
        generate_mels_by_sytle_tokens(model, waveglow, hparams, args.text, denoiser_strength=0, device=device)
    
    print("time :", time.time() - start)
