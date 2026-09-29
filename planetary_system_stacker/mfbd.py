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

Multi-frame blind deconvolution (MFBD) as an alternative to shift-and-add stacking.

Lucky imaging keeps the sharpest part of each frame, shifts it and averages: the rest of the light is
thrown away, and the average is still blurred, so it is sharpened afterwards with one guessed kernel.
MFBD instead models every frame as the object blurred by that frame's OWN point spread function,
y_t = f_t * x + sky, and estimates x and all f_t together, so every frame contributes through its own
blur (Harmeling et al. 2009; Hirsch et al. 2011, A&A 531, A9). This is a batch multi-frame
Richardson-Lucy scheme:

  * frames are placed on PSS's global alignment (the intersection canvas used by stacking), refined
    to sub-pixel accuracy against a reference, and optionally frames recorded during a jump of the
    global alignment (e.g. a mount correction) are dropped
  * colour frames are reduced to an R+G luminance after moving R onto G (atmospheric dispersion is
    measured once on the mean frame); mono frames are used at full resolution
  * the canvas is cut into overlapping patches (seeing differs across the field). In every iteration,
    for every frame: its local tip-tilt is removed by one smooth polynomial warp fitted to per-patch
    shifts, then a few multiplicative updates of its PSF per patch (non-negative, with the frame's
    flux fixed), and its Richardson-Lucy correction is accumulated. The image is updated once per
    iteration from the sum over ALL frames (clipped, and only where frames constrain it)
  * patches are blended with a Hann window. For colour input the deconvolved luminance replaces the
    luminance of PSS's stacked image (LRGB), so colour comes from the conventional stack.

Validated on Saturn videos (C8, 2x Barlow, 2026-09-26): on an even/odd frame split of single videos
it matched or beat AutoStakkert (best 50 %) at every spatial scale.
"""

from math import ceil
from time import time

import numpy as np
from cv2 import findTransformECC, MOTION_TRANSLATION, TERM_CRITERIA_EPS, TERM_CRITERIA_COUNT, \
    GaussianBlur, remap, resize, INTER_CUBIC, BORDER_REPLICATE
from scipy import fft as sfft
from scipy import ndimage
from skimage.util import img_as_uint

from exceptions import ArgumentError, NotSupportedError
from miscellaneous import Miscellaneous

EPS = 1.e-6


class MultiFrameBlindDeconvolution(object):
    """
    Deconvolve all (or the best part of) the frames of a job jointly, on the canvas of the stacked
    image. Usage, after "StackFrames.merge_alignment_point_buffers":

        mfbd = MultiFrameBlindDeconvolution(configuration, frames, rank_frames, align_frames, ...)
        image = mfbd.deconvolve_and_transfer(stack_frames)   # uint16, shape of the stacked image
    """

    def __init__(self, configuration, frames, rank_frames, align_frames, progress_signal=None,
                 logfile=None):
        self.configuration = configuration
        self.frames = frames
        self.rank_frames = rank_frames
        self.align_frames = align_frames
        self.progress_signal = progress_signal
        self.logfile = logfile

        self.P = configuration.mfbd_patch_size
        self.ST = configuration.mfbd_patch_step
        self.K = configuration.mfbd_psf_size
        if self.K % 2 != 1:
            raise ArgumentError("MFBD PSF size must be odd, got " + str(self.K))
        if self.P < 2 * self.K:
            raise ArgumentError("MFBD patch size (" + str(self.P) + ") must be at least twice the"
                                " PSF size (" + str(self.K) + ")")

        self.luminance = None     # result: sky-free float32 image on the intersection canvas
        self.sky = 0.
        self.rg_shift = None
        self.used_indices = None

    def protocol(self, text, level=1):
        if self.configuration.global_parameters_protocol_level >= level:
            Miscellaneous.protocol(text, self.logfile,
                                   precede_with_timestamp=(level == 1))

    # ------------------------------------------------------------------ frame selection and luminance
    def select_frames(self):
        """
        Choose the frames to use: the best "mfbd_frame_percent" by PSS's frame ranking, minus frames
        next to a jump of the global alignment.

        :return: List of frame indices, best first.
        """
        n = self.frames.number
        order = list(self.rank_frames.quality_sorted_indices)
        threshold = self.configuration.mfbd_jump_threshold
        if threshold > 0 and n > 2:
            shifts = np.array(self.align_frames.frame_shifts, dtype=np.float64)
            step = np.hypot(*np.diff(shifts, axis=0, prepend=shifts[:1]).T)
            jumps = step > threshold
            moving = ndimage.binary_dilation(jumps, iterations=1)
            order = [i for i in order if not moving[i]]
            self.protocol("           Frames next to a jump of the global alignment (> " +
                          str(threshold) + " px): " + str(int(jumps.sum())) + " jumps, " +
                          str(int(moving.sum())) + " frames dropped", level=2)
        number = max(1, int(round(len(order) * self.configuration.mfbd_frame_percent / 100.)))
        return order[:number]

    def frame_luminance(self, index):
        """
        :param index: Frame index
        :return: float32 luminance of the frame (full resolution, frame ADU)
        """
        frame = self.frames.frames(index)
        if not self.frames.color:
            return frame.astype(np.float32)
        red = frame[:, :, 0].astype(np.float32)
        green = frame[:, :, 1].astype(np.float32)
        if self.rg_shift is not None:
            red = np.fft.ifft2(ndimage.fourier_shift(np.fft.fft2(red), self.rg_shift)).real.astype(
                np.float32)
        return 0.5 * (red + green)

    def measure_dispersion(self, indices):
        """
        Measure the shift that moves the red channel onto the green one on a mean of the given
        (globally aligned) colour frames, and store it in self.rg_shift.
        """
        red = green = None
        for index in indices:
            frame = self.cut(self.frames.frames(index).astype(np.float32), index)
            if red is None:
                red, green = frame[:, :, 0].copy(), frame[:, :, 1].copy()
            else:
                red += frame[:, :, 0]
                green += frame[:, :, 1]

        def prep(image):
            image = (image - np.percentile(image, 1)) / max(float(image.max()), EPS)
            return GaussianBlur(image.astype(np.float32), (0, 0), 1.0)

        warp = np.eye(2, 3, dtype=np.float32)
        try:
            _, warp = findTransformECC(prep(green), prep(red), warp, MOTION_TRANSLATION,
                                       (TERM_CRITERIA_EPS | TERM_CRITERIA_COUNT, 300, 1e-7), None, 5)
            self.rg_shift = (-float(warp[1, 2]), -float(warp[0, 2]))
        except Exception:
            self.rg_shift = None
        self.protocol("           MFBD: red channel moved onto green (dispersion) by " +
                      ("(%+.2f, %+.2f) px" % self.rg_shift if self.rg_shift else "0 (not measurable)"),
                      level=2)

    def cut(self, frame, index):
        """
        :return: The part of a frame which lies on the intersection canvas (global alignment).
        """
        inter = self.align_frames.intersection_shape
        sy, sx = (int(round(v)) for v in self.align_frames.frame_shifts[index])
        return frame[inter[0][0] - sy:inter[0][1] - sy, inter[1][0] - sx:inter[1][1] - sx]

    # ------------------------------------------------------------------ sub-pixel alignment
    @staticmethod
    def upsampled_dft(data, size, factor, offsets):
        """Sample the DFT of "data" on a small, finely spaced grid near "offsets"
        (matrix-multiply DFT, Guizar-Sicairos et al. 2008)."""
        out = data
        for axis in (1, 0):
            n = data.shape[axis]
            freqs = np.fft.ifftshift(np.arange(n) - n // 2)
            grid = np.arange(size) - offsets[axis]
            kernel = np.exp((-2j * np.pi / (n * factor)) * grid[:, None] * freqs[None, :])
            out = np.tensordot(kernel, out, axes=([1], [axis]))
            out = np.moveaxis(out, 0, axis)
        return out

    @staticmethod
    def phase_shift(ref_fft, image, factor=20):
        """
        :return: (dy, dx) that moves "image" onto the reference, to ~1/factor pixel.
        """
        cross = ref_fft * np.conj(np.fft.fft2(image))
        magnitude = np.abs(cross)
        magnitude[magnitude == 0] = 1.
        cross /= magnitude
        corr = np.fft.ifft2(cross)
        peak = np.unravel_index(np.argmax(np.abs(corr)), corr.shape)
        shifts = np.array(peak, dtype=np.float64)
        wrap = shifts > np.array(corr.shape) / 2
        shifts[wrap] -= np.array(corr.shape)[wrap]
        size = int(ceil(factor * 1.5))
        center = size // 2
        fine = MultiFrameBlindDeconvolution.upsampled_dft(cross.conj(), size, factor,
                                                          center - shifts * factor).conj()
        fine_peak = np.unravel_index(np.argmax(np.abs(fine)), fine.shape)
        shifts += (np.array(fine_peak, dtype=np.float64) - center) / factor
        return float(shifts[0]), float(shifts[1])

    @staticmethod
    def shift_image(image, dy, dx):
        """Move image content by (dy, dx): whole pixels with edge replication, the rest as a
        band-limited Fourier shift."""
        iy, ix = int(round(dy)), int(round(dx))
        if iy or ix:
            image = ndimage.shift(image, (iy, ix), order=0, mode="nearest")
        fy, fx = dy - iy, dx - ix
        if abs(fy) > 1.e-3 or abs(fx) > 1.e-3:
            image = np.fft.ifft2(ndimage.fourier_shift(np.fft.fft2(image), (fy, fx))).real
        return image.astype(np.float32)

    # ------------------------------------------------------------------ main entry points
    def deconvolve(self):
        """
        Run MFBD on the intersection canvas.

        :return: Sky-free float32 luminance on the intersection canvas (frame ADU).
        """
        config = self.configuration
        t0 = time()
        use = self.select_frames()
        if len(use) < 2:
            raise NotSupportedError("MFBD needs at least two frames, " + str(len(use)) +
                                    " available")
        self.used_indices = use
        n_init = max(1, int(round(len(use) * config.mfbd_init_percent / 100.)))

        if self.frames.color and config.mfbd_align_red_onto_green:
            self.measure_dispersion(use[:max(n_init, min(len(use), 50))])

        # Frames on the canvas.
        frames = np.stack([self.cut(self.frame_luminance(i), i) for i in use])
        H, W = frames.shape[1:]
        if min(H, W) < self.P:
            raise NotSupportedError("MFBD: stacking area " + str(W) + "x" + str(H) +
                                    " is smaller than one patch (" + str(self.P) + " px)")
        border = np.concatenate([frames[:, :4].ravel()[::97], frames[:, -4:].ravel()[::97],
                                 frames[:, :, :4].ravel()[::97], frames[:, :, -4:].ravel()[::97]])
        mean = frames.mean(axis=0)
        # Sky: the background level for planets on black sky. For surface videos (Moon, Sun) there is
        # no sky in the frame; the darkest parts of the mean frame bound it from above.
        sky = max(0., min(float(np.median(border)), float(np.percentile(mean, 1))))
        self.sky = sky

        # Sub-pixel refinement against the mean of the best frames.
        reference = frames[:n_init].mean(axis=0) - sky
        window = np.outer(np.hanning(H), np.hanning(W)).astype(np.float32)
        ref_fft = np.fft.fft2(reference * window)
        residual = []
        for j in range(len(use)):
            dy, dx = self.phase_shift(ref_fft, (frames[j] - sky) * window)
            residual.append(np.hypot(dy, dx))
            frames[j] = self.shift_image(frames[j], dy, dx)
        self.protocol("           MFBD: " + str(len(use)) + " frames on a " + str(W) + "x" +
                      str(H) + " canvas, sky " + "%.1f" % sky + ", sub-pixel residual of the global "
                      "alignment median %.2f px" % float(np.median(residual)), level=2)

        x0 = frames.mean(axis=0) - sky                                      # start: all frames
        self.luminance = self.iterate(frames, x0, sky, t0)
        return self.luminance

    def iterate(self, frames, x0, sky, t0):
        """Batch multi-frame Richardson-Lucy with per-frame, per-patch PSFs."""
        config = self.configuration
        P, ST, K = self.P, self.ST, self.K
        nf, H, W = frames.shape
        S = sfft.next_fast_len(P + K - 1)
        pad = K // 2

        ys = list(range(0, H - P + 1, ST))
        xs = list(range(0, W - P + 1, ST))
        if ys[-1] != H - P:
            ys.append(H - P)
        if xs[-1] != W - P:
            xs.append(W - P)
        x_max = max(float(x0.max()), EPS)
        cells = [(py, px) for py in ys for px in xs if x0[py:py + P, px:px + P].mean() > 0.05 * x_max]
        NP = len(cells)
        if NP == 0:
            raise NotSupportedError("MFBD: no patch of the stacking area contains the object")

        def gather(image, size):
            return np.stack([image[py:py + size, px:px + size] for py, px in cells])

        xpad = np.pad(np.clip(x0, 1.e-3, None), pad, mode="edge")
        X = np.zeros((NP, S, S), np.float32)
        X[:, :P + K - 1, :P + K - 1] = gather(xpad, P + K - 1)
        valid = np.zeros((S, S), np.float32)
        valid[K - 1:K - 1 + P, K - 1:K - 1 + P] = 1
        FV = sfft.rfft2(valid)
        r2 = ((np.arange(K) - pad)[:, None] ** 2 + (np.arange(K) - pad)[None, :] ** 2).astype(np.float32)
        g = np.exp(-0.5 * r2 / 1.5 ** 2)
        F0 = np.zeros((S, S), np.float32)
        F0[:K, :K] = g / g.sum()
        kk = np.mgrid[0:K, 0:K].astype(np.float32) - pad

        def conv(FX, FF):
            return sfft.irfft2(FX * FF, s=(S, S), workers=-1)

        def corr(FR, FA):
            return sfft.irfft2(FR * np.conj(FA), s=(S, S), workers=-1)

        self.protocol("           MFBD: " + str(NP) + " patches of " + str(P) + " px (step " + str(ST) +
                      "), PSF " + str(K) + " px, " + str(config.mfbd_iterations) + " iterations",
                      level=2)

        # Local tip-tilt: per-patch shifts onto the model, one smooth polynomial field per frame.
        LM = int(ceil(config.mfbd_local_max_shift)) + 2
        LN = P + 2 * LM
        lwin = np.outer(np.hanning(P), np.hanning(P)).astype(np.float32)
        lky = np.fft.fftfreq(LN)[:, None]
        lkx = np.fft.fftfreq(LN)[None, :]
        pcy = np.array([py + P / 2 for py, px in cells])
        pcx = np.array([px + P / 2 for py, px in cells])

        def basis(y, x):
            y = (y - H / 2) / H
            x = (x - W / 2) / W
            return np.stack([y ** i * x ** (d - i) for d in range(config.mfbd_warp_order + 1)
                             for i in range(d + 1)], -1)

        Bq = basis(pcy, pcx)
        gy, gx = np.mgrid[0:H, 0:W].astype(np.float32)
        grid_terms = basis(gy.ravel(), gx.ravel())

        def shifted_patches(j, sh):
            fp = np.pad(frames[j], LM, mode="edge")
            R = np.stack([fp[py:py + LN, px:px + LN] for py, px in cells]) - sky
            ramp = np.exp(-2j * np.pi * (lky[None] * sh[:, 0, None, None] +
                                         lkx[None] * sh[:, 1, None, None]))
            return (np.fft.ifft2(np.fft.fft2(R) * ramp).real[:, LM:LM + P, LM:LM + P] +
                    sky).astype(np.float32)

        def measure_local_shifts(FX):
            M = conv(FX, sfft.rfft2(np.broadcast_to(F0, (NP, S, S)), workers=-1))[
                :, K - 1:K - 1 + P, K - 1:K - 1 + P]
            FM = np.fft.fft2((M - M.mean((1, 2), keepdims=True)) * lwin)
            strong = M.std((1, 2)) > 0.1 * M.std((1, 2)).max()
            r = int(ceil(config.mfbd_local_max_shift))
            c = P // 2
            q = np.arange(NP)
            out = np.zeros((nf, NP, 2), np.float32)

            def parabola(m1, m0, p1):
                d = m1 - 2 * m0 + p1
                return np.where(d < 0, 0.5 * (m1 - p1) / np.where(d < 0, d, -1), 0.)

            for j in range(nf):
                sh = np.zeros((NP, 2))
                for _ in range(3):
                    pt = shifted_patches(j, sh) - sky
                    B = np.fft.fft2((pt - pt.mean((1, 2), keepdims=True)) * lwin)
                    cc = np.fft.fftshift(np.fft.ifft2(FM * np.conj(B)).real, axes=(1, 2))[
                         :, c - r:c + r + 1, c - r:c + r + 1]
                    py, px = np.unravel_index(cc.reshape(NP, -1).argmax(1), cc.shape[1:])
                    ok = (py > 0) & (py < 2 * r) & (px > 0) & (px < 2 * r) & strong
                    pyc, pxc = np.clip(py, 1, 2 * r - 1), np.clip(px, 1, 2 * r - 1)
                    dy = pyc - r + parabola(cc[q, pyc - 1, pxc], cc[q, pyc, pxc], cc[q, pyc + 1, pxc])
                    dx = pxc - r + parabola(cc[q, pyc, pxc - 1], cc[q, pyc, pxc], cc[q, pyc, pxc + 1])
                    sh += np.where(ok[:, None], np.stack([dy, dx], 1), 0)
                    sh = np.clip(sh, -config.mfbd_local_max_shift, config.mfbd_local_max_shift)
                out[j] = sh
            return out, strong

        def warp_frames(sh, strong):
            out = np.empty_like(frames)
            residuals = []
            for j in range(nf):
                w = strong.astype(float)
                for _ in range(2):                               # fit, drop > 2.5 sigma, refit
                    coef = [np.linalg.lstsq(Bq * w[:, None], sh[j, :, c] * w, rcond=None)[0]
                            for c in (0, 1)]
                    res = np.stack([sh[j, :, c] - Bq @ coef[c] for c in (0, 1)], 1)
                    sig = np.sqrt((res[strong] ** 2).mean()) + 1.e-6 if strong.any() else 1.e-6
                    w = (strong & (np.hypot(*res.T) < 2.5 * sig)).astype(float)
                residuals.append(sig)
                dy = (grid_terms @ coef[0]).reshape(H, W).astype(np.float32)
                dx = (grid_terms @ coef[1]).reshape(H, W).astype(np.float32)
                out[j] = remap(frames[j], gx - dx, gy - dy, INTER_CUBIC, borderMode=BORDER_REPLICATE)
            return out, float(np.median(residuals))

        def estimate_psf(Y, FX, f, iterations, scale):
            norm_f = corr(FV[None], FX)
            for _ in range(iterations):
                model = conv(FX, sfft.rfft2(f, workers=-1)) + sky
                ratio = np.where(valid > 0, Y / np.maximum(model, EPS), 0).astype(np.float32)
                f *= np.clip(corr(sfft.rfft2(ratio, workers=-1), FX) / np.maximum(norm_f, EPS), 0, 10)
                f[:, K:, :] = 0
                f[:, :, K:] = 0
                f *= (scale / np.maximum(f.sum((1, 2)), EPS))[:, None, None]
            return f

        psfs = np.zeros((nf, NP, K, K), np.float32)
        scales = np.zeros(nf, np.float32)
        warped = None
        for it in range(1, config.mfbd_iterations + 1):
            FX = sfft.rfft2(X, workers=-1)
            if config.mfbd_local_warp and (it == 1 or (it - 1) % config.mfbd_local_refresh == 0):
                sh, strong = measure_local_shifts(FX)
                warped, resid = warp_frames(sh, strong)
                self.protocol("           MFBD: local tip-tilt rms %.2f px over %d patches, "
                              "residual about the smooth field %.2f px" % (
                    float(np.sqrt((sh[:, strong] ** 2).sum(2).mean())) if strong.any() else 0.,
                    int(strong.sum()), resid), level=2)
            source = warped if warped is not None else frames
            num = np.zeros((NP, S, S), np.float32)
            den = np.zeros((NP, S, S), np.float32)
            centroids = np.zeros((NP, 2))
            for j in range(nf):
                Y = np.zeros((NP, S, S), np.float32)
                Y[:, K - 1:K - 1 + P, K - 1:K - 1 + P] = gather(source[j], P)
                if it == 1:
                    # The frame's flux (transparency) is fixed once: x*c, f/c is otherwise a free
                    # direction and the image level random-walks.
                    model = conv(FX, sfft.rfft2(np.broadcast_to(F0, (NP, S, S)), workers=-1)) * valid
                    scales[j] = max(float(((Y - sky) * valid).sum()), 1.) / \
                        max(float((model * valid).sum()), 1.e-3)
                    f = np.zeros((NP, S, S), np.float32)
                    f[:] = F0 * scales[j]
                    iterations = config.mfbd_psf_iterations_first
                else:
                    f = np.zeros((NP, S, S), np.float32)
                    f[:, :K, :K] = psfs[j]
                    iterations = config.mfbd_psf_iterations_later
                f = estimate_psf(Y, FX, f, iterations, np.full(NP, scales[j], np.float32))
                psfs[j] = f[:, :K, :K]
                fk = f[:, :K, :K]
                fs = np.maximum(fk.sum((1, 2)), EPS)
                centroids += np.stack([(fk * kk[0]).sum((1, 2)) / fs,
                                       (fk * kk[1]).sum((1, 2)) / fs], axis=1)
                FF = sfft.rfft2(f, workers=-1)
                model = conv(FX, FF) + sky
                ratio = np.where(valid > 0, Y / np.maximum(model, EPS), 0).astype(np.float32)
                num += corr(sfft.rfft2(ratio, workers=-1), FF)
                den += corr(FV[None], FF)
            # Update only where enough frames constrain x (patch borders are covered by few PSF tails).
            ratio = np.clip(num / np.maximum(den, EPS), 1. / config.mfbd_ratio_clip,
                            config.mfbd_ratio_clip)
            ok = den >= config.mfbd_min_support * den.max(axis=(1, 2), keepdims=True)
            ratio = np.where(ok, ratio, 1.)
            X[:, :P + K - 1, :P + K - 1] *= ratio[:, :P + K - 1, :P + K - 1]
            # Keep the PSFs centred: move each patch by its frames' mean PSF offset.
            c = centroids / nf
            for q in range(NP):
                X[q, :P + K - 1, :P + K - 1] = ndimage.shift(X[q, :P + K - 1, :P + K - 1], c[q],
                                                             order=3, mode="nearest")
            np.clip(X, 0, None, out=X)
            self.protocol("           MFBD: iteration " + str(it) + " of " +
                          str(config.mfbd_iterations) + " done (%.1f s)" % (time() - t0), level=2)
            if self.progress_signal is not None:
                self.progress_signal.emit("Multi-frame blind deconvolution",
                                          int(round(100 * it / config.mfbd_iterations)))

        # Blend the patches.
        win = np.outer(np.hanning(P + 2)[1:-1], np.hanning(P + 2)[1:-1]).astype(np.float32)
        acc = np.zeros((H, W), np.float32)
        wsum = np.zeros((H, W), np.float32)
        for q, (py, px) in enumerate(cells):
            acc[py:py + P, px:px + P] += win * X[q, pad:pad + P, pad:pad + P]
            wsum[py:py + P, px:px + P] += win
        return np.where(wsum > 1.e-3, acc / np.maximum(wsum, 1.e-3), x0).astype(np.float32)

    def deconvolve_and_transfer(self, stack_frames):
        """
        Run MFBD and put the result onto the stacked image's canvas: the same borders, drizzle
        factor and brightness scale. Mono: the deconvolved image replaces the stack. Colour: the
        deconvolved luminance replaces the stack's R+G luminance, colour ratios are kept.

        :param stack_frames: StackFrames object after "merge_alignment_point_buffers"
        :return: uint16 image with the shape of stack_frames.stacked_image
        """
        luminance = self.deconvolve() if self.luminance is None else self.luminance
        stacked = stack_frames.stacked_image.astype(np.float32) / 65535.

        # Same geometry as the stacked image: drizzle, then the borders StackFrames trimmed.
        factor = self.configuration.drizzle_factor
        if factor != 1:
            luminance = resize(luminance, (stack_frames.dim_x_drizzled, stack_frames.dim_y_drizzled),
                               interpolation=INTER_CUBIC)
        luminance = luminance[stack_frames.border_y_low:luminance.shape[0] - stack_frames.border_y_high,
                              stack_frames.border_x_low:luminance.shape[1] - stack_frames.border_x_high]
        if luminance.shape[:2] != stacked.shape[:2]:
            luminance = resize(luminance, (stacked.shape[1], stacked.shape[0]), interpolation=INTER_CUBIC)

        stacked_luminance = 0.5 * (stacked[:, :, 0] + stacked[:, :, 1]) if self.frames.color \
            else stacked
        # Brightness: match the stack's luminance above its own background, on the object.
        background = float(np.percentile(stacked_luminance, 1))
        top = float(np.percentile(stacked_luminance, 99.9))
        mask = stacked_luminance > background + 0.15 * (top - background)
        if mask.sum() < 16:
            mask = np.ones_like(mask)
        gain = float(np.median(stacked_luminance[mask] - background)) / \
            max(float(np.median(luminance[mask])), EPS)
        new_luminance = luminance * gain + background

        if self.frames.color:
            eps = 0.02 * max(top, EPS)
            ratio = np.clip((new_luminance + eps) / (stacked_luminance + eps), 0., 4.)
            result = stacked * ratio[:, :, None]
        else:
            result = new_luminance
        self.protocol("           MFBD: result scaled to the stacked image (gain %.4g), " % gain +
                      ("colour taken from the stacked image" if self.frames.color else "mono"),
                      level=2)
        return img_as_uint(np.clip(result, 0., 1.))
