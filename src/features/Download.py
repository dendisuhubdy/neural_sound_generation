# License: BSD 3-clause
# Authors: Kyle Kastner
# LTSD routine from jfsantos (Joao Felipe Santos)
# Harvest, Cheaptrick, D4C, WORLD routines based on MATLAB code from M. Morise
# http://ml.cs.yamanashi.ac.jp/world/english/
# MGC code based on r9y9 (Ryuichi Yamamoto) MelGeneralizedCepstrums.jl
# Pieces also adapted from SPTK
from __future__ import division
import numpy as np
import zipfile
import tarfile
import os

from numpy.lib.stride_tricks import as_strided
from scipy.interpolate import interp1d
from scipy.cluster.vq import vq
from scipy import linalg, fftpack
from numpy.testing import assert_almost_equal
from scipy.io import wavfile
from scipy.signal import firwin
from multiprocessing import Pool

try:
    import urllib.request as urllib  # for backwards compatibility
except ImportError:
    import urllib2 as urllib


def download(url, server_fname, local_fname=None, progress_update_percentage=5,
             bypass_certificate_check=False):
    """
    An internet download utility modified from
    http://stackoverflow.com/questions/22676/
    how-do-i-download-a-file-over-http-using-python/22776#22776
    """
    if bypass_certificate_check:
        import ssl
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        u = urllib.urlopen(url, context=ctx)
    else:
        u = urllib.urlopen(url)
    if local_fname is None:
        local_fname = server_fname
    full_path = local_fname
    meta = u.info()
    with open(full_path, 'wb') as f:
        try:
            file_size = int(meta.get("Content-Length"))
        except TypeError:
            print("WARNING: Cannot get file size, displaying bytes instead!")
            file_size = 100
        print("Downloading: %s Bytes: %s" % (server_fname, file_size))
        file_size_dl = 0
        block_sz = int(1E7)
        p = 0
        while True:
            buffer = u.read(block_sz)
            if not buffer:
                break
            file_size_dl += len(buffer)
            f.write(buffer)
            if (file_size_dl * 100. / file_size) > p:
                status = r"%10d  [%3.2f%%]" % (file_size_dl, file_size_dl *
                                               100. / file_size)
                print(status)
                p += progress_update_percentage


def fetch_sample_speech_tapestry():
    url = "https://www.dropbox.com/s/qte66a7haqspq2g/tapestry.wav?dl=1"
    wav_path = "tapestry.wav"
    if not os.path.exists(wav_path):
        download(url, wav_path)
    fs, d = wavfile.read(wav_path)
    d = d.astype('float32') / (2 ** 15)
    # file is stereo? - just choose one channel
    return fs, d


def fetch_sample_file(wav_path):
    if not os.path.exists(wav_path):
        raise ValueError("Unable to find file at path %s" % wav_path)
    fs, d = wavfile.read(wav_path)
    d = d.astype('float32') / (2 ** 15)
    # file is stereo - just choose one channel
    if len(d.shape) > 1:
        d = d[:, 0]
    return fs, d


def fetch_sample_music():
    url = "http://www.music.helsinki.fi/tmt/opetus/uusmedia/esim/"
    url += "a2002011001-e02-16kHz.wav"
    wav_path = "test.wav"
    if not os.path.exists(wav_path):
        download(url, wav_path)
    fs, d = wavfile.read(wav_path)
    d = d.astype('float32') / (2 ** 15)
    # file is stereo - just choose one channel
    d = d[:, 0]
    return fs, d


def fetch_sample_speech_fruit(n_samples=None):
    url = 'https://dl.dropboxusercontent.com/u/15378192/audio.tar.gz'
    wav_path = "audio.tar.gz"
    if not os.path.exists(wav_path):
        download(url, wav_path)
    tf = tarfile.open(wav_path)
    wav_names = [fname for fname in tf.getnames()
                 if ".wav" in fname.split(os.sep)[-1]]
    speech = []
    print("Loading speech files...")
    for wav_name in wav_names[:n_samples]:
        f = tf.extractfile(wav_name)
        fs, d = wavfile.read(f)
        d = d.astype('float32') / (2 ** 15)
        speech.append(d)
    return fs, speech


def fetch_sample_speech_eustace(n_samples=None):
    """
    http://www.cstr.ed.ac.uk/projects/eustace/download.html
    """
    # data
    url = "http://www.cstr.ed.ac.uk/projects/eustace/down/eustace_wav.zip"
    wav_path = "eustace_wav.zip"
    if not os.path.exists(wav_path):
        download(url, wav_path)

    # labels
    url = "http://www.cstr.ed.ac.uk/projects/eustace/down/eustace_labels.zip"
    labels_path = "eustace_labels.zip"
    if not os.path.exists(labels_path):
        download(url, labels_path)

    # Read wavfiles
    # 16 kHz wav
    zf = zipfile.ZipFile(wav_path, 'r')
    wav_names = [fname for fname in zf.namelist()
                 if ".wav" in fname.split(os.sep)[-1]]
    fs = 16000
    speech = []
    print("Loading speech files...")
    for wav_name in wav_names[:n_samples]:
        wav_str = zf.read(wav_name)
        d = np.frombuffer(wav_str, dtype=np.int16)
        d = d.astype('float32') / (2 ** 15)
        speech.append(d)

    zf = zipfile.ZipFile(labels_path, 'r')
    label_names = [fname for fname in zf.namelist()
                   if ".lab" in fname.split(os.sep)[-1]]
    labels = []
    print("Loading label files...")
    for label_name in label_names[:n_samples]:
        label_file_str = zf.read(label_name)
        labels.append(label_file_str)
    return fs, speech
