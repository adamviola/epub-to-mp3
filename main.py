from ebooklib import epub
import torch
from html_parser import html_to_text
import os
import json
import math
from scipy.io.wavfile import write
from pydub import AudioSegment
import re
from bar import Bar
import shutil
from pathlib import Path

# Needed to do some ugly changes to sys.path to access Hifi-GAN files
import sys
sys.path.append('hifi-gan/')
from meldataset import MAX_WAV_VALUE
from models import Generator
from inference_e2e import load_checkpoint
from env import AttrDict

line_end_chars = set(('.', '?', ';', '!'))

def main():

    # Check for path to epub
    if len(sys.argv) < 2:
        print("Missing relative path to book:\tpython main.py path/to/book.epub")
        exit()

    book_path = sys.argv[1]
    if not os.path.exists(book_path):
        print(f"Path '{book_path}' does not exist.")
        exit()

    book = epub.read_epub(book_path)
    print(f"Found '{book_path}'")
    title = '.'.join(Path(book_path).name.split('.')[:-1])

    print("\nLet's decide which sections of the epub to generate audio for:")
    # sections = []
    # for index, section in sorted([get_spine_key(book)(itm) for itm in book.get_items()]):
    #     if ask_y_n(section):
    #         sections.append(section)
    sections = ["Synopsis.xhtml"]

    print("\nProcessing sections...")
    sequences = []
    for section in sections:
        html = str(book.get_item_with_id(section).get_content(), "utf-8")
        text = html_to_text(html)

        for line in text.splitlines():
            line = line.strip()

            line = re.sub(u'[“”"]', '', line)
            line = re.sub(u'[‘’]', '\'', line)
            line = re.sub(u" \'((?:.(?! \'))+)\'([ \.;—?!…])", r' \1\2', line)

            if line == '':
                continue
        
            split_line = re.split(u'([\.;—?!…])+', line)
            for i in range(0, len(split_line), 2):
                sequence = split_line[i]
                punctuation = split_line[i + 1] if i < len(split_line) - 1 and split_line[i + 1] in line_end_chars else '.'

                sequence = sequence.strip()
                if sequence == '':
                    continue

                sequence += punctuation
                sequences.append(sequence)

        print(section)
    
    if len(sequences) == 0:
        print("No text found.")
        exit()

    os.makedirs('temp', exist_ok=True)

    # Check for CUDA
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('\nUsing CUDA')
    else:
        device = torch.device('cpu')
        print('\nCUDA unavailable. Using CPU')

    print("\nProcessing input...")
    utils = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_tts_utils')

    # NVIDIA reorders lines without telling us the order (!), so we compute order ourselves
    # https://github.com/NVIDIA/DeepLearningExamples/blob/81ee705868a11d6fe18c12d237abe4a08aab5fd6/PyTorch/SpeechSynthesis/Tacotron2/tacotron2/entrypoints.py#L150
    d = []
    for i,text in enumerate(sequences):
        d.append(torch.IntTensor(type(utils).text_to_sequence(text, ['english_cleaners'])[:]))
    order = torch.argsort(torch.LongTensor([len(x) for x in d]), dim=0, descending=True)

    # Preprocess lines for Tacotron2
    input_sequences, lengths = utils.prepare_input_sequence(sequences, cpu_run=device.type == 'cpu')

    print("\nInitializing Tacotron2 and HiFi-GAN...")

    # Initialize Tacotron2
    # Pretrained Tacotron2 assumes CUDA, so we manually load state_dict to support CPU-only machines
    tacotron2 = torch.hub.load('nvidia/DeepLearningExamples:torchhub', 'nvidia_tacotron2', pretrained=False).to(device)
    checkpoint = torch.hub.load_state_dict_from_url('https://api.ngc.nvidia.com/v2/models/nvidia/tacotron2pyt_fp32/versions/1/files/nvidia_tacotron2pyt_fp32_20190306.pth', map_location=device)
    state_dict = {key.replace("module.", ""): value for key, value in checkpoint["state_dict"].items()}
    tacotron2.load_state_dict(state_dict)
    tacotron2.decoder.max_decoder_steps = 2000
    tacotron2.eval()

    # Initialize HiFi-GAN
    with open('model/config.json') as f:
        data = f.read()
    json_config = json.loads(data)
    h = AttrDict(json_config)
    generator = Generator(h).to(device)
    state_dict_g = load_checkpoint("model/generator_v1", device)
    generator.load_state_dict(state_dict_g['generator'])
    generator.eval()
    generator.remove_weight_norm()

    # Run TTS in batches
    N = len(sequences)
    batch_size = 16
    num_batches = math.ceil(N / batch_size)
    print()
    bar = Bar('Running...', max=N, suffix='Utterance: %(index)d / %(max)d [%(elapsed_td)s / %(total_td)s]')
    bar.next(0)
    for batch_idx in range(num_batches):
        # Compute start and end indices of batch
        s = batch_idx * batch_size
        e = s + batch_size

        # Trim sequences to the maximum length of sequences in the batch
        batch_lengths = lengths[s:e]
        batch_seq_length = max(lengths[s:e])
        batch_sequences = input_sequences[s:e, :batch_seq_length]
        
        with torch.no_grad():
            # Tacotron2: text -> mel-spectrograms
            mels, mel_lengths, _ = tacotron2.infer(batch_sequences, batch_lengths)

            # HiFi-GAN: mel-spectrograms -> waveform
            audio = generator(mels).squeeze() * MAX_WAV_VALUE

        # Store audio to file
        for i, l in zip(range(batch_size), mel_lengths):
            line_audio = audio[i,:l * h.hop_size].cpu().numpy().astype('int16')
            write(f"temp/{order[batch_idx * batch_size + i]}.wav", h.sampling_rate, line_audio)

        bar.next(len(batch_lengths))
    bar.finish()
        
    print("\nConverting wavs to mp3...")
    if sys.platform.startswith("win"):
        AudioSegment.converter = os.getcwd() + "/utils/ffmpeg.exe"
    gap = AudioSegment.silent(600, frame_rate=h.sampling_rate)
    book = AudioSegment.empty()
    for i in range(N):
        book += AudioSegment.from_wav(f"{os.getcwd()}/temp/{i}.wav") + gap
    book += AudioSegment.silent(1000, frame_rate=h.sampling_rate)

    bitrate = str((book.frame_rate * book.frame_width * 8 * book.channels) / 1000)
    book.export(f"{os.getcwd()}/output/{title}.mp3", format="mp3", bitrate=bitrate)

    print("\nCleaning...")
    shutil.rmtree('temp')

    print(f"\nDone! {title}.mp3 is in the 'output' directory.")


def get_spine_key(book):
    spine_keys = {id:(ii,id) for (ii,(id,show)) in enumerate(book.spine)}
    past_end = len(spine_keys)
    return lambda itm: spine_keys.get(itm.get_id(), (past_end,itm.get_id()))

def ask_y_n(prompt):
    result = input(prompt + " ([y]/n)? ").strip()
    while result != "" and result != "y" and result != "n":
        result = input("[y/n]? ")
    return result != "n"

if __name__ == "__main__":
    main()