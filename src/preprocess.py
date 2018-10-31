# coding: utf-8
"""
Preprocess dataset

usage: preprocess.py [options] <name> <in_dir> <out_dir>

options:
    --num_workers=<n>        Num workers.
    --hparams=<parmas>       Hyper parameters [default: ].
    --preset=<json>          Path of preset parameters (json).
    --sample_rate=<n>        Audio sample rate
    --fft_size=<n>           FFT size
    --hop_size=<n>           Hop size/Step Size
    --n_mels=<n>             Number of Melspectrogram
    -h, --help               Show help message.
"""
import os
import importlib

from docopt import docopt
from multiprocessing import cpu_count
from tqdm import tqdm
from hparams_tacotron import hparams


def preprocess(mod, in_dir, out_root, num_workers, sample_rate, fft_size, hop_size, n_mels):
    os.makedirs(out_dir, exist_ok=True)
    metadata = mod.build_from_path(in_dir, out_dir, num_workers, sample_rate, fft_size, hop_size, n_mels, tqdm=tqdm)
    write_metadata(metadata, out_dir, sample_rate)


def write_metadata(metadata, out_dir, sample_rate):
    with open(os.path.join(out_dir, 'train.txt'), 'w', encoding='utf-8') as f:
        for m in metadata:
            f.write('|'.join([str(x) for x in m]) + '\n')
    frames = sum([m[2] for m in metadata])
    # sr = hparams.sample_rate
    sr = sample_rate
    hours = frames / sr / 3600
    print('Sample rate %d' % sample_rate)
    print('FFT size %d' % fft_size)
    print('Number of mel coefficients %d' % n_mels)
    print('Wrote %d utterances, %d time steps (%.2f hours)' % (len(metadata), frames, hours))
    print('Max input length:  %d' % max(len(m[3]) for m in metadata))
    print('Max output length: %d' % max(m[2] for m in metadata))


if __name__ == "__main__":
    args = docopt(__doc__)
    name = args["<name>"]
    in_dir = args["<in_dir>"]
    out_dir = args["<out_dir>"]
    num_workers = args["--num_workers"]
    num_workers = cpu_count() if num_workers is None else int(num_workers)
    preset = args["--preset"]
    sample_rate = int(args["--sample_rate"])
    fft_size = int(args["--fft_size"])
    hop_size = int(args["--hop_size"])
    n_mels = int(args["--n_mels"])

    # Load preset if specified
    if preset is not None:
        with open(preset) as f:
            hparams.parse_json(f.read())
    # Override hyper parameters
    hparams.parse(args["--hparams"])
    assert hparams.name == "vocoder"

    print("Sampling frequency: {}".format(sample_rate))

    assert name in ["cmu_arctic", "ljspeech", "librivox", "jsut"]
    mod = importlib.import_module(name)
    preprocess(mod, in_dir, out_dir, num_workers, sample_rate, fft_size, hop_size, n_mels)
