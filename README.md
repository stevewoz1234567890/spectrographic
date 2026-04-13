# Spectrographic

Python utilities to turn WAV audio into spectrogram features for visualization and downstream labeling. The script loads audio, splits it into overlapping windows, applies a Gaussian window and FFT, normalizes the result, and optionally aligns per-window labels from a tab-separated annotation file.

## Requirements

- Python 3
- [librosa](https://librosa.org/) — audio I/O and resampling
- [NumPy](https://numpy.org/)
- [SciPy](https://scipy.org/) — FFT, signal windows
- [Matplotlib](https://matplotlib.org/) — spectrogram plot

Install dependencies (example):

```bash
pip install librosa numpy scipy matplotlib
```

## Usage

By default, `Spectrogram.py` expects:

- **Audio:** `Audio/ch07-20210224-093335-094153-001000000000.wav`
- **Labels:** `Audio/l_ch07-20210224-093335-094153-001000000000.txt`

Run from the repository root:

```bash
python Spectrogram.py
```

To process other files, edit the `audio_file_path` and `label_file_path` variables at the bottom of `Spectrogram.py`, or import `spectrogram()` and call it with your paths.

### `spectrogram()` parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `window_size` | 256 | Samples per STFT-style window |
| `window_overlap_percentage` | 50 | Overlap between windows as a percent of `window_size` |
| `min_frequency` | 10 | Lower frequency bound (Hz) |
| `max_frequency` | 8000 | Upper frequency bound (Hz) |
| `threshold` | 0.1 | Floor for small spectrogram values after processing |

## Label file format

Each non-empty line is tab-separated: `start_seconds`, `end_seconds`, `label_name`.

Supported labels map to integer indices: `Object` → 1, `Human` → 2, `Chicken` → 3, `Click` → 4; anything else → 0.

## Outputs

For an input like `path/to/audio.wav`, the script writes:

- `{basename}_spectrographic_features_flipped.csv` — feature matrix (comma-separated)
- `{basename}_labels.csv` — one label index per time window (comma-separated)

It also opens a Matplotlib figure showing the spectrogram (magma colormap).
