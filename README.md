# epub-to-mp3
Convert your favorite ebook (epub) to an audiobook (mp3) using state-of-the-art neural speech synthesis models.

To synthesize text, epub-to-mp3 uses Tacotron2 and HiFi-GAN V1 pre-trained on the LJ Speech dataset.

With a Google Colab's GPU instance, speech synthesis runs ~30x faster than realtime - e.g. 6 hour audiobook in 12 minutes.

# Getting Started
## Google Colab (easy)
1. Download `colab.ipynb` from this repository.
2. Create a new Google Colab notebook.
3. Upload `colab.ipynb` via `File > Upload notebook` and follow the instructions in the notebook.

## Local (hard)
1. Create a new Python 3.6.5 environment
2. Install [PyTorch 1.4](https://pytorch.org/get-started/previous-versions/#v140) the other required packages `pip install -r requirements.txt`
3. If not on Windows, install [FFmpeg](https://ffmpeg.org/download.html)
4. Run `python main.py path/to/book.epub`
