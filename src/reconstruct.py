import numpy as np
import os
import sys
import audio
from audio import is_mulaw_quantize, is_mulaw, is_raw


def main():
    outdir = str(sys.argv[1])
    mel_filename = "ljspeech-mel-00001.npy"
    melspectrogram = np.load(os.path.join(out_dir, mel_filename))
    print(melspectrogram)

if __name__ == "__main__":
    main()
