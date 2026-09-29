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

Smoke test of multi-frame blind deconvolution (module mfbd): run the command line version of PSS
with "--mfbd" on the test videos in "Videos", and check that every job writes a stacked image and
an MFBD image of the same shape, finite, not clipped, and with more fine detail than the stack.

Run from the "planetary_system_stacker" directory:

    python Test_programs/mfbd_smoke_test.py --workdir /tmp/pss_mfbd_smoke [--iterations 8]
"""

from argparse import ArgumentParser
from os import makedirs
from os.path import join, splitext, basename, isfile, abspath, dirname
from shutil import copy
from subprocess import run
from sys import executable
from time import time

import numpy as np
from cv2 import imread, IMREAD_UNCHANGED, GaussianBlur

VIDEOS = ["8bit_mono.ser", "16bit_mono.ser", "another_short_video.avi", "short_video.avi"]

parser = ArgumentParser(description="Smoke test of PSS with multi-frame blind deconvolution")
parser.add_argument("--workdir", required=True, help="scratch directory for copies and results")
parser.add_argument("--videos", default=join(dirname(dirname(abspath(__file__))), "Videos"),
                    help="directory with the test videos")
parser.add_argument("--iterations", type=int, default=8, help="MFBD iterations")
arguments = parser.parse_args()

pss = join(dirname(dirname(abspath(__file__))), "planetary_system_stacker.py")
failures = 0
for video in VIDEOS:
    job_dir = join(arguments.workdir, splitext(video)[0])
    makedirs(job_dir, exist_ok=True)
    copy(join(arguments.videos, video), job_dir)
    start = time()
    result = run([executable, "-u", pss, join(job_dir, video), "--out_format", "tiff", "-s", "50",
                  "--protocol_detail", "2", "--mfbd", "--mfbd_iterations", str(arguments.iterations)],
                 cwd=dirname(pss))
    stem = join(job_dir, splitext(video)[0] + "_pss")
    problems = []
    if not isfile(stem + ".tiff") or not isfile(stem + "_mfbd.tiff"):
        problems.append("output missing (exit code " + str(result.returncode) + ")")
    else:
        stack = imread(stem + ".tiff", IMREAD_UNCHANGED).astype(np.float32) / 65535.
        mfbd = imread(stem + "_mfbd.tiff", IMREAD_UNCHANGED).astype(np.float32) / 65535.
        if stack.shape != mfbd.shape:
            problems.append("shape " + str(mfbd.shape) + " differs from the stack's " + str(stack.shape))
        else:
            ls = stack.mean(axis=2) if stack.ndim == 3 else stack
            lm = mfbd.mean(axis=2) if mfbd.ndim == 3 else mfbd
            detail = lambda x: (GaussianBlur(x, (0, 0), 0.8) - GaussianBlur(x, (0, 0), 1.6)).std()
            gain = detail(lm) / max(detail(ls), 1.e-9)
            clipped = 100. * (lm > 0.999).mean()
            if not np.isfinite(mfbd).all():
                problems.append("non-finite pixels")
            if clipped > 1.:
                problems.append("%.1f %% clipped" % clipped)
            if abs(lm.mean() / max(ls.mean(), 1.e-9) - 1.) > 0.05:
                problems.append("mean brightness %.3f vs stack %.3f" % (lm.mean(), ls.mean()))
            if gain < 1.:
                problems.append("less fine detail than the stack (x%.2f)" % gain)
            print("%-26s %5.0f s  shape %-16s fine detail x%.2f vs the stack, %.2f %% clipped" %
                  (video, time() - start, str(mfbd.shape), gain, clipped))
    if problems:
        failures += 1
        print("%-26s FAILED: %s" % (video, "; ".join(problems)))

print("MFBD smoke test: %d of %d videos passed" % (len(VIDEOS) - failures, len(VIDEOS)))
exit(1 if failures else 0)
