# -*- coding: utf-8; -*-
"""
Copyright (c) 2026 Mads Dørup

This file is part of the PlanetarySystemStacker tool (PSS).
https://github.com/Rolf-Hempel/PlanetarySystemStacker

PSS is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with PSS.  If not, see <http://www.gnu.org/licenses/>.

A fair comparison of PSS's stacking and multi-frame blind deconvolution (MFBD): the even/odd test.

Every video is split into its even and its odd frames. Both halves see the same object under the
same seeing, so whatever two results of the same method disagree on is noise (or an artefact that
is not reproducible). Each half is processed independently by every method, and the two results
are compared with Fourier ring correlation (FRC):

    FRC(f) = Re sum(A B*) / sqrt(sum |A|^2 sum |B|^2)  over a ring of spatial frequency f
    SNR(f) ~ FRC / (1 - FRC)

Any linear filter a method applies (smoothing, sharpening) multiplies both spectra and cancels, so an
unsharpened stack is not at a disadvantage against a deconvolved one: FRC measures how much
reproducible detail each method recovers at each scale. PSS is run at several stack sizes and
alignment point box widths, and its best setting per band is the reference. MFBD runs with its
defaults.

Where both results agree almost perfectly (FRC > 0.99, usually the coarsest scales) the SNR ratio
is dominated by tiny geometric differences between the halves and is not meaningful; such bands
are marked with "*".

Caveat: artefacts which a method produces identically in both halves (e.g. ringing at the same
edges) also correlate. Check the images as well.

Run from the "planetary_system_stacker" directory:

    python Test_programs/mfbd_evenodd_comparison.py --workdir /tmp/pss_mfbd_evenodd
"""

from argparse import ArgumentParser
from os import makedirs
from os.path import join, splitext, abspath, dirname, isfile, basename
from struct import pack, unpack
from subprocess import run
import sys
from sys import executable
from time import time

import numpy as np
from cv2 import imread, IMREAD_UNCHANGED, VideoCapture, CAP_PROP_FRAME_COUNT

# (after the cv2 import: OpenCV's loader replaces the sys.path list)
sys.path.insert(0, dirname(dirname(abspath(__file__))))
from mfbd import MultiFrameBlindDeconvolution as Mfbd

VIDEOS = ["8bit_mono.ser", "16bit_mono.ser", "another_short_video.avi", "short_video.avi"]

parser = ArgumentParser(description="Even/odd FRC comparison of PSS stacking and MFBD")
parser.add_argument("--workdir", required=True, help="scratch directory for the halves and results")
parser.add_argument("--videos", default=join(dirname(dirname(abspath(__file__))), "Videos"),
                    help="directory with the test videos")
parser.add_argument("--stack_percents", default="25,50,100",
                    help="PSS stack sizes to try (percent of frames, comma separated)")
parser.add_argument("--box_widths", default="32,48,64,96,128",
                    help="PSS alignment point box widths to try (pixels, comma separated; PSS allows 20-140)")
parser.add_argument("--mfbd_iterations", type=int, default=8, help="MFBD iterations")
parser.add_argument("--bands", default="0.02,0.05,0.10,0.15,0.20,0.25,0.30",
                    help="FRC band edges (cycles per pixel)")
parser.add_argument("--video", action="append", default=None,
                    help="only these test videos (repeatable; default: all four)")
arguments = parser.parse_args()
PSS = join(dirname(dirname(abspath(__file__))), "planetary_system_stacker.py")
EDGES = [float(v) for v in arguments.bands.split(",")]
PERCENTS = [int(v) for v in arguments.stack_percents.split(",")]
BOXES = [int(v) for v in arguments.box_widths.split(",")]
T0 = time()


def log(text):
    print("%7.0f s  %s" % (time() - T0, text), flush=True)


# ------------------------------------------------------------------ video -> even / odd SER files
def read_video(file_name):
    """
    :return: (frames as a list of 2D arrays, SER header template of 178 bytes)
    """
    if file_name.lower().endswith(".ser"):
        with open(file_name, "rb") as f:
            header = f.read(178)
            _, _, color, _, width, height, depth, count = unpack("<14s7i", header[:42])
            if color != 0:
                raise ValueError(file_name + ": only mono SER files are split by this test")
            dtype = np.dtype("<u2") if depth > 8 else np.dtype(np.uint8)
            data = np.frombuffer(f.read(width * height * dtype.itemsize * count), dtype=dtype)
        return list(data.reshape(count, height, width)), header
    capture = VideoCapture(file_name)
    frames = []
    for _ in range(int(capture.get(CAP_PROP_FRAME_COUNT))):
        ok, frame = capture.read()
        if not ok:
            break
        # The test AVIs are grey (channels differ by compression noise only): store them as mono.
        frames.append(np.round(frame.astype(np.float32).mean(axis=2)).astype(np.uint8))
    height, width = frames[0].shape
    header = pack("<14s7i", b"LUCAM-RECORDER", 0, 0, 0, width, height, 8, len(frames)) + \
        bytes(178 - 42)
    return frames, header


def write_ser(file_name, frames, header):
    header = header[:38] + pack("<i", len(frames)) + header[42:]
    with open(file_name, "wb") as f:
        f.write(header)
        for frame in frames:
            f.write(frame.tobytes())


# ------------------------------------------------------------------ FRC
def load_luminance(file_name):
    image = imread(file_name, IMREAD_UNCHANGED).astype(np.float64)
    return image.mean(axis=2) if image.ndim == 3 else image


def frc_bands(a, b):
    """
    Register b onto a (sub-pixel), crop the common interior, window and compute the FRC per band.
    """
    h, w = min(a.shape[0], b.shape[0]), min(a.shape[1], b.shape[1])
    a, b = a[:h, :w] - a[:h, :w].mean(), b[:h, :w] - b[:h, :w].mean()
    window = np.outer(np.hanning(h), np.hanning(w))
    dy, dx = Mfbd.phase_shift(np.fft.fft2(a * window), b * window)
    b = Mfbd.shift_image(b, dy, dx).astype(np.float64)
    m = int(np.ceil(max(abs(dy), abs(dx)))) + 16
    a, b = a[m:h - m, m:w - m], b[m:h - m, m:w - m]
    window = np.outer(np.hanning(a.shape[0]), np.hanning(a.shape[1]))
    fa, fb = np.fft.fft2((a - a.mean()) * window), np.fft.fft2((b - b.mean()) * window)
    radius = np.hypot(np.fft.fftfreq(a.shape[0])[:, None], np.fft.fftfreq(a.shape[1])[None, :])
    out = []
    for low, high in zip(EDGES[:-1], EDGES[1:]):
        ring = (radius >= low) & (radius < high)
        cross = np.real(fa[ring] * np.conj(fb[ring])).sum()
        out.append(cross / np.sqrt((np.abs(fa[ring]) ** 2).sum() * (np.abs(fb[ring]) ** 2).sum()))
    return np.array(out), (dy, dx)


def snr(frc):
    frc = np.clip(frc, -0.99, 0.9999)
    return frc / (1. - frc)


# ------------------------------------------------------------------ main
summary = []
for video in (arguments.video or VIDEOS):
    frames, header = read_video(join(arguments.videos, video))
    stem = splitext(video)[0]
    results = {}
    for parity, subset in (("even", frames[0::2]), ("odd", frames[1::2])):
        runs = [("pss_p%d_a%d" % (p, b), p, b) for p in PERCENTS for b in BOXES
                if (p, b) != (50, 48)] + [("pss_p50_and_mfbd", 50, 48)]
        for method, percent, box in runs:
            # (the MFBD run also produces PSS's default stack: 50 %, box width 48)
            job_dir = join(arguments.workdir, stem, parity, method)
            makedirs(job_dir, exist_ok=True)
            half = join(job_dir, stem + "_" + parity + ".ser")
            out = join(job_dir, stem + "_" + parity + "_pss")
            if method == "pss_p50_and_mfbd":
                results.setdefault("PSS 50 % a48", {})[parity] = out + ".tiff"
                results.setdefault("MFBD", {})[parity] = out + "_mfbd.tiff"
                done = isfile(out + "_mfbd.tiff")
            else:
                results.setdefault("PSS %d %% a%d" % (percent, box), {})[parity] = out + ".tiff"
                done = isfile(out + ".tiff")
            if done:
                continue                                       # reuse the result of an earlier run
            write_ser(half, subset, header)
            command = [executable, "-u", PSS, half, "--out_format", "tiff", "-s", str(percent),
                       "-a", str(box), "--protocol_detail", "0"]
            if method == "pss_p50_and_mfbd":
                command += ["--mfbd", "--mfbd_iterations", str(arguments.mfbd_iterations)]
            start = time()
            run(command, cwd=dirname(PSS))
            log("%s %s (%d frames) %s: %.0f s" % (video, parity, len(subset), method, time() - start))

    table = {}
    for name in results:
        even, odd = results[name].get("even"), results[name].get("odd")
        if not (even and odd and isfile(even) and isfile(odd)):
            log("%s %s: output missing" % (video, name))
            continue
        frc, shift = frc_bands(load_luminance(even), load_luminance(odd))
        table[name] = frc
    pss_names = [k for k in table if k.startswith("PSS")]
    pss_frc = np.array([table[k] for k in pss_names])
    best = pss_frc.argmax(axis=0)
    pss_best = snr(pss_frc.max(axis=0))
    saturated = pss_frc.max(axis=0) > 0.99
    band_names = ["%.2f-%.2f" % (lo, hi) for lo, hi in zip(EDGES[:-1], EDGES[1:])]
    print("\n%s: %d frames -> even/odd halves of %d/%d; FRC between the halves (1 = identical)" %
          (video, len(frames), len(frames[0::2]), len(frames[1::2])))
    print("%-16s %s" % ("cycles/px", "  ".join("%9s" % b for b in band_names)))
    for name in pss_names + (["MFBD"] if "MFBD" in table else []):
        print("%-16s %s" % (name, "  ".join("%9.3f" % v for v in table[name])))
    print("%-16s %s" % ("best PSS", "  ".join("%9s" % pss_names[i].replace("PSS ", "") for i in best)))
    if "MFBD" in table:
        ratio = snr(table["MFBD"]) / np.maximum(pss_best, 1.e-9)
        print("%-16s %s" % ("MFBD / best PSS", "  ".join("%8.2f%s" % (v, "*" if s else " ")
                                                          for v, s in zip(ratio, saturated))))
        summary.append((video, ratio, saturated))
    else:
        summary.append((video, None, None))

print("\nSummary: MFBD SNR / best PSS SNR per band (> 1 = MFBD recovers more reproducible detail)")
print("%-26s %s" % ("cycles/px", "  ".join("%9s" % ("%.2f-%.2f" % (lo, hi))
                                            for lo, hi in zip(EDGES[:-1], EDGES[1:]))))
for video, ratio, saturated in summary:
    print("%-26s %s" % (video, "  ".join("%8.2f%s" % (v, "*" if s else " ")
                                          for v, s in zip(ratio, saturated)) if ratio is not None
                        else "MFBD output missing"))
print("* PSS's best FRC > 0.99 in this band: both methods agree almost perfectly, the ratio is not "
      "meaningful")
