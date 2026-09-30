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

Lucky imaging keeps the sharpest part of each frame, shifts it and averages: the rest of the light
is thrown away, and the average is still blurred, so it is sharpened afterwards with one guessed
kernel. MFBD instead models every frame as the object blurred by that frame's own point spread
function (PSF), y_t = f_t * x + sky, and estimates x and all f_t together, so every frame
contributes through its own blur.

The method comes from the literature: multi-frame blind deconvolution of astronomical images
(T. J. Schulz 1993); the formulation with per-frame PSFs, multiplicative updates and overlapping
patches for space-variant seeing (S. Harmeling, M. Hirsch, S. Sra and B. Schölkopf 2009-2011); its
use for lucky imaging (J. A. Hitchcock, D. M. Bramich, D. Foreman-Mackey, D. W. Hogg and
M. Hundertmark 2022); the Richardson-Lucy iteration (W. H. Richardson 1972, L. B. Lucy 1974) and
its blind form (D. A. Fish, A. M. Brinicombe, E. R. Pike and J. G. Walker 1995); and sub-pixel
registration by upsampled cross-correlation (M. Guizar-Sicairos, S. T. Thurman and J. R. Fienup
2008).

This implementation is a batch multi-frame Richardson-Lucy scheme:

  * Frames are placed on PSS's global alignment (the intersection canvas used by stacking),
    refined to sub-pixel accuracy against a reference. Optionally, frames recorded during a jump
    of the global alignment (e.g. a mount correction) are dropped.
  * Colour frames are reduced to an R+G luminance after moving R onto G (atmospheric dispersion is
    measured once on the mean frame). Optionally ("mfbd_binning" 2, recommended for one-shot colour
    cameras) the luminance is binned 2x2, like a Bayer superpixel: each processing pixel then
    carries four times the signal, which the per-frame PSF estimation needs. The result is
    upsampled back to the frames' resolution.
  * The canvas is cut into overlapping patches (seeing differs across the field). In every
    iteration, for every frame: its local tip-tilt is removed by one smooth polynomial warp fitted
    to per-patch shifts, a few multiplicative updates of its PSF per patch are done (non-negative,
    with the frame's flux fixed), and its Richardson-Lucy correction is accumulated. The image is
    updated once per iteration from the sum over all frames (clipped, and only where the frames
    constrain it).
  * The patches are blended with a Hann window. For colour input the deconvolved luminance
    replaces the luminance of PSS's stacked image (LRGB), so the colour comes from the stack.
"""

from math import ceil, pi
from time import time

from cv2 import findTransformECC, MOTION_TRANSLATION, TERM_CRITERIA_EPS, TERM_CRITERIA_COUNT, \
    GaussianBlur, remap, resize, INTER_CUBIC, BORDER_REPLICATE
from numpy import absolute, arange, argmax, array, broadcast_to, clip, concatenate, conj, diff, \
    empty_like, exp, eye, float32, float64, full, hanning, hypot, maximum, median, mgrid, \
    moveaxis, ones_like, outer, pad, percentile, sqrt, stack, tensordot, unravel_index, where, \
    zeros
from numpy.fft import fft2, ifft2, fftfreq, fftshift, ifftshift
from numpy.linalg import lstsq
from scipy import ndimage
from scipy.fft import rfft2, irfft2, next_fast_len
from skimage.util import img_as_uint

from exceptions import ArgumentError, NotSupportedError
from miscellaneous import Miscellaneous

# Lower bound for denominators.
EPS = 1.e-6


class MultiFrameBlindDeconvolution(object):
    """
    Deconvolve all (or the best part of the) frames of a job jointly, on the canvas of the stacked
    image. The object is used after "StackFrames.merge_alignment_point_buffers":

        mfbd = MultiFrameBlindDeconvolution(configuration, frames, rank_frames, align_frames,
                                            my_timer)
        image = mfbd.deconvolve_and_transfer(stack_frames)

    """

    def __init__(self, configuration, frames, rank_frames, align_frames, my_timer,
                 progress_signal=None, logfile=None):
        """
        Initialize the MFBD object and check the patch geometry.

        :param configuration: Configuration object with parameters
        :param frames: Frames object with all video frames
        :param rank_frames: RankFrames object with global quality ranks of all frames
        :param align_frames: AlignFrames object with global shifts of all frames
        :param my_timer: Timer object for accumulating times spent in specific code sections
        :param progress_signal: Either None (no progress signalling), or a signal with the
                                signature (str, int) with the current activity (str) and the
                                progress in percent (int).
        :param logfile: Either None, or the file to which the protocol is written as well.
        """

        self.configuration = configuration
        self.frames = frames
        self.rank_frames = rank_frames
        self.align_frames = align_frames
        self.my_timer = my_timer
        self.progress_signal = progress_signal
        self.logfile = logfile

        # Patch geometry. The sizes are given in pixels of the frames and converted to processing
        # pixels (binned). The PSF size must be odd, and the patch large enough to hold several PSF
        # widths.
        self.binning = configuration.mfbd_binning
        if self.binning not in (1, 2):
            raise ArgumentError("MFBD binning must be 1 or 2, got " + str(self.binning))
        self.patch_size = int(round(configuration.mfbd_patch_size / self.binning))
        self.patch_step = max(self.patch_size // 2, 1)
        self.psf_size = configuration.mfbd_psf_size // self.binning
        if self.psf_size % 2 != 1:
            self.psf_size += 1
        if self.patch_size < 2 * self.psf_size:
            raise ArgumentError("MFBD patch size (" + str(self.patch_size) + ") must be at least "
                                "twice the PSF size (" + str(self.psf_size) + ")")
        self.psf_half_size = self.psf_size // 2
        self.fft_size = next_fast_len(self.patch_size + self.psf_size - 1)

        # Data set up by "deconvolve".
        self.used_indices = None
        self.red_green_shift = None
        self.sky = 0.
        self.frame_stack = None
        self.frame_stack_warped = None
        self.full_canvas_height = self.full_canvas_width = None
        self.canvas_height = self.canvas_width = None
        self.start_image = None
        self.patch_corners = None
        self.number_patches = None
        self.image_patches = None
        self.valid_mask = None
        self.valid_mask_fft = None
        self.psf_start = None
        self.psf_offsets = None
        self.local_margin = self.local_size = None
        self.local_window = None
        self.local_frequencies_y = self.local_frequencies_x = None
        self.patch_basis = None
        self.canvas_y = self.canvas_x = None
        self.canvas_basis = None
        self.psfs = None
        self.frame_fluxes = None

        # The result: a sky-free float32 luminance image on the intersection canvas.
        self.luminance = None

        for name in ['MFBD: reading and aligning frames', 'MFBD: local shifts and warping',
                     'MFBD: PSFs and image updates', 'MFBD: blending and transfer']:
            self.my_timer.create_no_check(name)

    def protocol(self, text, level=1):
        """
        Write a line to the protocol if the protocol level is high enough.

        :param text: Text to be written
        :param level: Minimum protocol level at which the text is written. At level 1 a time stamp
                      is written as well.
        :return: -
        """

        if self.configuration.global_parameters_protocol_level >= level:
            Miscellaneous.protocol(text, self.logfile, precede_with_timestamp=(level == 1))

    def select_frames(self):
        """
        Choose the frames to be used: the best "mfbd_frame_percent" percent according to PSS's frame
        ranking, without the frames next to a jump of the global alignment (if requested).

        :return: List of frame indices, best frame first
        """

        number_frames = self.frames.number
        order = list(self.rank_frames.quality_sorted_indices)

        # A jump of the global alignment between consecutive frames (e.g. a mount correction during
        # the recording) smears the frames on both sides of it. Drop those frames.
        threshold = self.configuration.mfbd_jump_threshold
        if threshold > 0 and number_frames > 2:
            shifts = array(self.align_frames.frame_shifts, dtype=float64)
            step = hypot(*diff(shifts, axis=0, prepend=shifts[:1]).T)
            jumps = step > threshold
            moving = ndimage.binary_dilation(jumps, iterations=1)
            order = [index for index in order if not moving[index]]
            self.protocol("           Frames next to a jump of the global alignment (> " +
                          str(threshold) + " px): " + str(int(jumps.sum())) + " jumps, " +
                          str(int(moving.sum())) + " frames dropped", level=2)

        number_used = max(1, int(round(len(order) * self.configuration.mfbd_frame_percent / 100.)))
        return order[:number_used]

    def frame_luminance(self, index):
        """
        Compute the luminance of a frame. For colour frames this is the mean of the red and green
        channels, with the red channel moved onto the green one first (atmospheric dispersion).

        :param index: Frame index
        :return: float32 luminance of the frame (full resolution, in the frame's units)
        """

        frame = self.frames.frames(index)
        if not self.frames.color:
            return frame.astype(float32)

        red = frame[:, :, 0].astype(float32)
        green = frame[:, :, 1].astype(float32)
        if self.red_green_shift is not None:
            red = ifft2(ndimage.fourier_shift(fft2(red), self.red_green_shift)).real.astype(float32)
        return 0.5 * (red + green)

    def cut_to_canvas(self, frame, index):
        """
        Cut the part out of a frame which lies on the intersection canvas of the global alignment.

        :param frame: Frame (any number of channels)
        :param index: Frame index
        :return: The part of the frame on the intersection canvas
        """

        intersection = self.align_frames.intersection_shape
        shift_y, shift_x = (int(round(value)) for value in self.align_frames.frame_shifts[index])
        return frame[intersection[0][0] - shift_y:intersection[0][1] - shift_y,
                     intersection[1][0] - shift_x:intersection[1][1] - shift_x]

    def bin_image(self, image):
        """
        Bin an image by the factor "self.binning" (mean of binning x binning pixels). Rows and
        columns which do not fill a bin are dropped.

        :param image: 2D image
        :return: Binned image (the input itself if the binning is 1)
        """

        if self.binning == 1:
            return image
        binning = self.binning
        height, width = image.shape[0] // binning, image.shape[1] // binning
        return image[:height * binning, :width * binning].reshape(
            height, binning, width, binning).mean(axis=(1, 3))

    def upsample_to_canvas(self, image):
        """
        Undo the binning of a result: upsample it by zero-padding its spectrum (exact for
        band-limited images), move it by (binning - 1) / 2 pixels (a binned pixel is centred
        between the pixels it was made from), and replicate the edge where the binning dropped a
        row or column.

        :param image: Binned 2D image
        :return: Image with the size of the full intersection canvas
        """

        if self.binning == 1:
            return image
        binning = self.binning
        height, width = image.shape

        # The zero frequency of a centred spectrum of length n sits at n // 2, both before and after
        # padding.
        spectrum = fftshift(fft2(image))
        padded = zeros((height * binning, width * binning), dtype=spectrum.dtype)
        offset_y = (height * binning) // 2 - height // 2
        offset_x = (width * binning) // 2 - width // 2
        padded[offset_y:offset_y + height, offset_x:offset_x + width] = spectrum
        padded = ifftshift(padded)
        shift = (binning - 1) / 2.
        upsampled = (ifft2(ndimage.fourier_shift(padded, (shift, shift))).real *
                     binning * binning).astype(float32)
        return pad(upsampled, ((0, self.full_canvas_height - upsampled.shape[0]),
                               (0, self.full_canvas_width - upsampled.shape[1])), mode="edge")

    @staticmethod
    def normalized_blur(image):
        """
        Normalize an image to its 1 percentile and maximum, and blur it slightly, as input for the
        ECC registration.

        :param image: 2D image
        :return: Normalized and blurred float32 image
        """

        image = (image - percentile(image, 1)) / max(float(image.max()), EPS)
        return GaussianBlur(image.astype(float32), (0, 0), 1.0)

    def measure_dispersion(self, indices):
        """
        Measure the shift which moves the red channel onto the green one, on the mean of the given
        (globally aligned) colour frames. The result is stored in self.red_green_shift.

        :param indices: Indices of the frames to be averaged
        :return: -
        """

        red = green = None
        for index in indices:
            frame = self.cut_to_canvas(self.frames.frames(index).astype(float32), index)
            if red is None:
                red, green = frame[:, :, 0].copy(), frame[:, :, 1].copy()
            else:
                red += frame[:, :, 0]
                green += frame[:, :, 1]

        # ECC finds the warp which maps green onto red. The red channel is moved back by its
        # translation.
        warp = eye(2, 3, dtype=float32)
        try:
            criteria = (TERM_CRITERIA_EPS | TERM_CRITERIA_COUNT, 300, 1e-7)
            _, warp = findTransformECC(self.normalized_blur(green), self.normalized_blur(red), warp,
                                       MOTION_TRANSLATION, criteria, None, 5)
            self.red_green_shift = (-float(warp[1, 2]), -float(warp[0, 2]))
        except Exception:
            self.red_green_shift = None
        self.protocol("           MFBD: red channel moved onto green (dispersion) by " +
                      ("(%+.2f, %+.2f) px" % self.red_green_shift if self.red_green_shift
                       else "0 (not measurable)"), level=2)

    @staticmethod
    def upsampled_dft(data, size, factor, offsets):
        """
        Sample the DFT of "data" on a small, finely spaced grid near "offsets" (matrix-multiply
        DFT, M. Guizar-Sicairos et al. 2008). This is much cheaper than zero-padding the whole
        transform by the same factor.

        :param data: 2D array (a cross-power spectrum)
        :param size: Number of samples per coordinate direction
        :param factor: Upsampling factor
        :param offsets: Position (y, x) of the grid origin, in upsampled pixels
        :return: Complex 2D array (size, size)
        """

        output = data
        for axis in (1, 0):
            length = data.shape[axis]
            frequencies = ifftshift(arange(length) - length // 2)
            grid = arange(size) - offsets[axis]
            kernel = exp((-2j * pi / (length * factor)) * grid[:, None] * frequencies[None, :])
            output = tensordot(kernel, output, axes=([1], [axis]))
            output = moveaxis(output, 0, axis)
        return output

    @staticmethod
    def phase_shift(reference_fft, image, factor=20):
        """
        Measure the translation of an image relative to a reference by phase correlation, refined
        to about 1/factor pixel with an upsampled DFT around the correlation peak.

        :param reference_fft: FFT of the reference image
        :param image: Image of the same shape as the reference
        :param factor: Upsampling factor of the refinement
        :return: (dy, dx), the shift which moves "image" onto the reference
        """

        # Normalized cross-power spectrum: its inverse FFT peaks at the shift.
        cross = reference_fft * conj(fft2(image))
        magnitude = absolute(cross)
        magnitude[magnitude == 0] = 1.
        cross /= magnitude

        # Coarse peak of the correlation, with wrap-around to negative shifts.
        correlation = ifft2(cross)
        peak = unravel_index(argmax(absolute(correlation)), correlation.shape)
        shifts = array(peak, dtype=float64)
        wrap = shifts > array(correlation.shape) / 2
        shifts[wrap] -= array(correlation.shape)[wrap]

        # Refinement on a fine grid around the coarse peak.
        size = int(ceil(factor * 1.5))
        center = size // 2
        fine = MultiFrameBlindDeconvolution.upsampled_dft(cross.conj(), size, factor,
                                                          center - shifts * factor).conj()
        fine_peak = unravel_index(argmax(absolute(fine)), fine.shape)
        shifts += (array(fine_peak, dtype=float64) - center) / factor
        return float(shifts[0]), float(shifts[1])

    @staticmethod
    def shift_image(image, shift_y, shift_x):
        """
        Move image content by (shift_y, shift_x): whole pixels with edge replication, the fraction
        as a band-limited Fourier shift.

        :param image: 2D image
        :param shift_y: Shift in y (pixels)
        :param shift_x: Shift in x (pixels)
        :return: Shifted float32 image
        """

        # Whole pixels first (no interpolation), then the remaining fraction.
        integer_y, integer_x = int(round(shift_y)), int(round(shift_x))
        if integer_y or integer_x:
            image = ndimage.shift(image, (integer_y, integer_x), order=0, mode="nearest")
        fraction_y, fraction_x = shift_y - integer_y, shift_x - integer_x
        if abs(fraction_y) > 1.e-3 or abs(fraction_x) > 1.e-3:
            image = ifft2(ndimage.fourier_shift(fft2(image), (fraction_y, fraction_x))).real
        return image.astype(float32)

    def load_frames(self):
        """
        Select the frames, reduce them to luminance on the intersection canvas, estimate the sky
        level, refine the global alignment to sub-pixel accuracy, and compute the start image (the
        mean of all used frames).

        :return: -
        """

        configuration = self.configuration
        self.used_indices = self.select_frames()
        if len(self.used_indices) < 2:
            raise NotSupportedError("MFBD needs at least two frames, " +
                                    str(len(self.used_indices)) + " available")
        number_reference = max(1, int(round(len(self.used_indices) *
                                            configuration.mfbd_init_percent / 100.)))

        if self.frames.color and configuration.mfbd_align_red_onto_green:
            self.measure_dispersion(
                self.used_indices[:max(number_reference, min(len(self.used_indices), 50))])

        # Put the luminance of all used frames onto the canvas, and bin it (if requested).
        intersection = self.align_frames.intersection_shape
        self.full_canvas_height = intersection[0][1] - intersection[0][0]
        self.full_canvas_width = intersection[1][1] - intersection[1][0]
        self.frame_stack = stack([self.bin_image(self.cut_to_canvas(self.frame_luminance(index),
                                                                    index))
                                  for index in self.used_indices])
        self.canvas_height, self.canvas_width = self.frame_stack.shape[1:]
        if min(self.canvas_height, self.canvas_width) < self.patch_size:
            raise NotSupportedError("MFBD: stacking area " + str(self.canvas_width) + "x" +
                                    str(self.canvas_height) + " is smaller than one patch (" +
                                    str(self.patch_size) + " px)")

        # Sky: the background level for planets on a black sky. For surface videos (Moon, Sun)
        # there is no sky in the frame; the darkest parts of the mean frame bound it from above.
        border = concatenate([self.frame_stack[:, :4].ravel()[::97],
                              self.frame_stack[:, -4:].ravel()[::97],
                              self.frame_stack[:, :, :4].ravel()[::97],
                              self.frame_stack[:, :, -4:].ravel()[::97]])
        mean_frame = self.frame_stack.mean(axis=0)
        self.sky = max(0., min(float(median(border)), float(percentile(mean_frame, 1))))

        # Refine the alignment of every frame to sub-pixel accuracy against the mean of the best
        # frames.
        reference = self.frame_stack[:number_reference].mean(axis=0) - self.sky
        window = outer(hanning(self.canvas_height), hanning(self.canvas_width)).astype(float32)
        reference_fft = fft2(reference * window)
        residuals = []
        for frame_index in range(len(self.used_indices)):
            shift_y, shift_x = self.phase_shift(reference_fft,
                                                (self.frame_stack[frame_index] - self.sky) * window)
            residuals.append(hypot(shift_y, shift_x))
            self.frame_stack[frame_index] = self.shift_image(self.frame_stack[frame_index], shift_y,
                                                             shift_x)
        self.protocol("           MFBD: " + str(len(self.used_indices)) + " frames on a " +
                      str(self.canvas_width) + "x" + str(self.canvas_height) + " canvas (binning " +
                      str(self.binning) + "), sky " +
                      "%.1f" % self.sky + ", sub-pixel residual of the global alignment median "
                      "%.2f px" % float(median(residuals)), level=2)

        # The start image is the mean of all used frames, without the sky.
        self.start_image = self.frame_stack.mean(axis=0) - self.sky

    def set_up_patches(self):
        """
        Cut the canvas into overlapping patches (only those which contain the object), and set up
        the image patches, the masks and the start PSF used by the iteration.

        :return: -
        """

        patch_size, patch_step, psf_size = self.patch_size, self.patch_step, self.psf_size
        fft_size, psf_half_size = self.fft_size, self.psf_half_size

        # Patch corners on a regular grid, with the last row and column at the canvas border.
        corners_y = list(range(0, self.canvas_height - patch_size + 1, patch_step))
        corners_x = list(range(0, self.canvas_width - patch_size + 1, patch_step))
        if corners_y[-1] != self.canvas_height - patch_size:
            corners_y.append(self.canvas_height - patch_size)
        if corners_x[-1] != self.canvas_width - patch_size:
            corners_x.append(self.canvas_width - patch_size)
        image_maximum = max(float(self.start_image.max()), EPS)
        self.patch_corners = [(y, x) for y in corners_y for x in corners_x
                              if self.start_image[y:y + patch_size, x:x + patch_size].mean() >
                              0.05 * image_maximum]
        self.number_patches = len(self.patch_corners)
        if self.number_patches == 0:
            raise NotSupportedError("MFBD: no patch of the stacking area contains the object")

        # Each image patch carries a margin of half a PSF width on all sides, and is zero-padded to
        # the FFT size.
        image_padded = pad(clip(self.start_image, 1.e-3, None), psf_half_size, mode="edge")
        self.image_patches = zeros((self.number_patches, fft_size, fft_size), float32)
        self.image_patches[:, :patch_size + psf_size - 1, :patch_size + psf_size - 1] = \
            self.patch_cutouts(image_padded, patch_size + psf_size - 1)

        # The part of a convolution result which corresponds to the frame patch.
        self.valid_mask = zeros((fft_size, fft_size), float32)
        self.valid_mask[psf_size - 1:psf_size - 1 + patch_size,
                        psf_size - 1:psf_size - 1 + patch_size] = 1
        self.valid_mask_fft = rfft2(self.valid_mask)

        # Start PSF: a narrow Gaussian (sigma 1.5 px). The PSF offsets are used for centroids.
        radius_squared = ((arange(psf_size) - psf_half_size)[:, None] ** 2 +
                          (arange(psf_size) - psf_half_size)[None, :] ** 2).astype(float32)
        gaussian = exp(-0.5 * radius_squared / 1.5 ** 2)
        self.psf_start = zeros((fft_size, fft_size), float32)
        self.psf_start[:psf_size, :psf_size] = gaussian / gaussian.sum()
        self.psf_offsets = mgrid[0:psf_size, 0:psf_size].astype(float32) - psf_half_size

        # Set-up for the local tip-tilt: patch windows with a margin for shifting, and the
        # polynomial basis of the smooth distortion field (at the patch centres and on the canvas).
        self.local_margin = int(ceil(self.configuration.mfbd_local_max_shift)) + 2
        self.local_size = patch_size + 2 * self.local_margin
        self.local_window = outer(hanning(patch_size), hanning(patch_size)).astype(float32)
        self.local_frequencies_y = fftfreq(self.local_size)[:, None]
        self.local_frequencies_x = fftfreq(self.local_size)[None, :]
        patch_centers_y = array([y + patch_size / 2 for y, x in self.patch_corners])
        patch_centers_x = array([x + patch_size / 2 for y, x in self.patch_corners])
        self.patch_basis = self.polynomial_basis(patch_centers_y, patch_centers_x)
        self.canvas_y, self.canvas_x = mgrid[0:self.canvas_height,
                                             0:self.canvas_width].astype(float32)
        self.canvas_basis = self.polynomial_basis(self.canvas_y.ravel(), self.canvas_x.ravel())

        self.protocol("           MFBD: " + str(self.number_patches) + " patches of " +
                      str(patch_size) + " px (step " + str(patch_step) + "), PSF " + str(psf_size) +
                      " px, " + str(self.configuration.mfbd_iterations) + " iterations", level=2)

    def patch_cutouts(self, image, size):
        """
        Cut square pieces out of an image at all patch corners.

        :param image: 2D image
        :param size: Edge length of the pieces
        :return: Array (number of patches, size, size)
        """

        return stack([image[y:y + size, x:x + size] for y, x in self.patch_corners])

    def convolve_patches(self, image_fft, psf_fft):
        """
        Convolve all image patches with their PSFs. The part corresponding to the frame patch is
        at [psf_size - 1:psf_size - 1 + patch_size] in both coordinates.

        :param image_fft: Real FFTs of the image patches
        :param psf_fft: Real FFTs of the PSFs
        :return: Convolutions (number of patches, fft size, fft size)
        """

        return irfft2(image_fft * psf_fft, s=(self.fft_size, self.fft_size), workers=-1)

    def correlate_patches(self, first_fft, second_fft):
        """
        Correlate patches: the result at j is the sum over i of first[i] * second[i - j].

        :param first_fft: Real FFTs of the first patches
        :param second_fft: Real FFTs of the second patches
        :return: Correlations (number of patches, fft size, fft size)
        """

        return irfft2(first_fft * conj(second_fft), s=(self.fft_size, self.fft_size), workers=-1)

    def polynomial_basis(self, y, x):
        """
        Evaluate the 2D polynomial terms up to order "mfbd_warp_order" at canvas positions, with
        the coordinates normalized to the canvas size.

        :param y: y coordinates
        :param x: x coordinates
        :return: Array (number of positions, number of terms)
        """

        y = (y - self.canvas_height / 2) / self.canvas_height
        x = (x - self.canvas_width / 2) / self.canvas_width
        return stack([y ** i * x ** (degree - i)
                      for degree in range(self.configuration.mfbd_warp_order + 1)
                      for i in range(degree + 1)], -1)

    def shifted_patches(self, frame_index, shifts):
        """
        Cut the patches out of a frame, each moved by its own sub-pixel shift (band-limited Fourier
        shift).

        :param frame_index: Index into the frame stack
        :param shifts: Array (number of patches, 2) with the (dy, dx) shift of every patch
        :return: float32 array (number of patches, patch size, patch size)
        """

        margin, size, patch_size = self.local_margin, self.local_size, self.patch_size
        frame_padded = pad(self.frame_stack[frame_index], margin, mode="edge")
        pieces = stack([frame_padded[y:y + size, x:x + size]
                        for y, x in self.patch_corners]) - self.sky
        ramp = exp(-2j * pi * (self.local_frequencies_y[None] * shifts[:, 0, None, None] +
                               self.local_frequencies_x[None] * shifts[:, 1, None, None]))
        return (ifft2(fft2(pieces) * ramp).real[:, margin:margin + patch_size,
                margin:margin + patch_size] + self.sky).astype(float32)

    @staticmethod
    def parabola_offset(value_minus, value_center, value_plus):
        """
        Sub-pixel position of a maximum from three samples, by a parabola fit.

        :param value_minus: Samples left of the maximum
        :param value_center: Samples at the maximum
        :param value_plus: Samples right of the maximum
        :return: Offsets of the maxima relative to the center samples (0 where there is no maximum)
        """

        curvature = value_minus - 2 * value_center + value_plus
        return where(curvature < 0,
                     0.5 * (value_minus - value_plus) / where(curvature < 0, curvature, -1), 0.)

    def measure_local_shifts(self, image_fft):
        """
        Measure, for every frame and patch, the local shift (tip-tilt) which moves the frame's
        patch onto the current model (the image patch blurred by the start PSF). Three refinements
        of a windowed cross-correlation with parabolic peak interpolation are done.

        :param image_fft: Real FFTs of the image patches
        :return: (shifts, strong): array (number of frames, number of patches, 2), and a boolean
                 array marking the patches with enough structure to be measured
        """

        patch_size, psf_size = self.patch_size, self.psf_size
        max_shift = self.configuration.mfbd_local_max_shift
        model = self.convolve_patches(image_fft, rfft2(
            broadcast_to(self.psf_start, (self.number_patches, self.fft_size, self.fft_size)),
            workers=-1))[:, psf_size - 1:psf_size - 1 + patch_size,
                         psf_size - 1:psf_size - 1 + patch_size]
        model_fft = fft2((model - model.mean((1, 2), keepdims=True)) * self.local_window)

        # Skip patches with almost no structure (they cannot be measured).
        strong = model.std((1, 2)) > 0.1 * model.std((1, 2)).max()

        # Search the correlation peak within +/- max_shift pixels of the patch centre, three times,
        # each time with the patch moved by the shift found so far.
        search = int(ceil(max_shift))
        center = patch_size // 2
        patch_indices = arange(self.number_patches)
        shifts_all = zeros((len(self.used_indices), self.number_patches, 2), float32)
        for frame_index in range(len(self.used_indices)):
            shifts = zeros((self.number_patches, 2))
            for _ in range(3):
                pieces = self.shifted_patches(frame_index, shifts) - self.sky
                pieces_fft = fft2((pieces - pieces.mean((1, 2), keepdims=True)) * self.local_window)
                correlation = fftshift(ifft2(model_fft * conj(pieces_fft)).real, axes=(1, 2))[
                    :, center - search:center + search + 1, center - search:center + search + 1]
                peak_y, peak_x = unravel_index(
                    correlation.reshape(self.number_patches, -1).argmax(1), correlation.shape[1:])
                inside = (peak_y > 0) & (peak_y < 2 * search) & (peak_x > 0) & \
                         (peak_x < 2 * search) & strong
                peak_y = clip(peak_y, 1, 2 * search - 1)
                peak_x = clip(peak_x, 1, 2 * search - 1)
                shift_y = peak_y - search + self.parabola_offset(
                    correlation[patch_indices, peak_y - 1, peak_x],
                    correlation[patch_indices, peak_y, peak_x],
                    correlation[patch_indices, peak_y + 1, peak_x])
                shift_x = peak_x - search + self.parabola_offset(
                    correlation[patch_indices, peak_y, peak_x - 1],
                    correlation[patch_indices, peak_y, peak_x],
                    correlation[patch_indices, peak_y, peak_x + 1])
                shifts += where(inside[:, None], stack([shift_y, shift_x], 1), 0)
                shifts = clip(shifts, -max_shift, max_shift)
            shifts_all[frame_index] = shifts
        return shifts_all, strong

    def warp_frames(self, shifts, strong):
        """
        Fit one smooth polynomial distortion field per frame to its patch shifts (robust: patches
        deviating by more than 2.5 sigma are dropped and the fit is repeated), and warp every frame
        with it.

        :param shifts: Array (number of frames, number of patches, 2) from "measure_local_shifts"
        :param strong: Boolean array marking the patches which were measured
        :return: (warped frame stack, median residual of the patch shifts about the fields)
        """

        warped = empty_like(self.frame_stack)
        residuals = []
        for frame_index in range(len(self.used_indices)):
            # Least-squares fit of the field to the shifts of the measured patches, then a second
            # fit without the outliers.
            weights = strong.astype(float)
            for _ in range(2):
                coefficients = [lstsq(self.patch_basis * weights[:, None],
                                      shifts[frame_index, :, c] * weights, rcond=None)[0]
                                for c in (0, 1)]
                residual = stack([shifts[frame_index, :, c] - self.patch_basis @ coefficients[c]
                                  for c in (0, 1)], 1)
                sigma = sqrt((residual[strong] ** 2).mean()) + 1.e-6 if strong.any() else 1.e-6
                weights = (strong & (hypot(*residual.T) < 2.5 * sigma)).astype(float)
            residuals.append(sigma)

            # The patch shift moves frame content onto the model:
            # output(y, x) = frame(y - dy, x - dx).
            field_y = (self.canvas_basis @ coefficients[0]).reshape(
                self.canvas_height, self.canvas_width).astype(float32)
            field_x = (self.canvas_basis @ coefficients[1]).reshape(
                self.canvas_height, self.canvas_width).astype(float32)
            warped[frame_index] = remap(self.frame_stack[frame_index], self.canvas_x - field_x,
                                        self.canvas_y - field_y, INTER_CUBIC,
                                        borderMode=BORDER_REPLICATE)
        return warped, float(median(residuals))

    def estimate_psf(self, frame_patches, image_fft, psf, iterations, flux):
        """
        Improve the PSFs of one frame (one per patch) by multiplicative (Richardson-Lucy type)
        updates with the image fixed. The PSFs stay non-negative, are confined to the PSF support,
        and their sums are kept at the frame's flux.

        :param frame_patches: The frame's patches, zero-padded to the FFT size
        :param image_fft: Real FFTs of the image patches
        :param psf: Array (number of patches, fft size, fft size) with the start PSFs
        :param iterations: Number of updates
        :param flux: Array (number of patches) with the flux of the frame
        :return: The improved PSFs
        """

        psf_size = self.psf_size

        # The Richardson-Lucy normalization for the PSF: the image summed over the valid region.
        normalization = self.correlate_patches(self.valid_mask_fft[None], image_fft)
        for _ in range(iterations):

            # Compare the frame with its model (image blurred by the current PSF plus sky), and
            # correct the PSF multiplicatively by the back-projected ratio.
            model = self.convolve_patches(image_fft, rfft2(psf, workers=-1)) + self.sky
            ratio = where(self.valid_mask > 0, frame_patches / maximum(model, EPS),
                          0).astype(float32)
            psf *= clip(self.correlate_patches(rfft2(ratio, workers=-1), image_fft) /
                        maximum(normalization, EPS), 0, 10)
            # Confine the PSF to its support, and keep its sum at the frame's flux.
            psf[:, psf_size:, :] = 0
            psf[:, :, psf_size:] = 0
            psf *= (flux / maximum(psf.sum((1, 2)), EPS))[:, None, None]
        return psf

    def iterate(self):
        """
        Batch multi-frame Richardson-Lucy iteration. In every iteration, for every frame: remove
        its local tip-tilt (in the first iteration and every "mfbd_local_refresh" iterations),
        update its PSFs, and accumulate its Richardson-Lucy correction. Then update the image once
        from the sum over all frames, and re-center the patches so that the mean PSF stays
        centered.

        :return: -
        """

        configuration = self.configuration
        patch_size, psf_size, fft_size = self.patch_size, self.psf_size, self.fft_size
        number_frames = len(self.used_indices)
        start_time = time()

        self.psfs = zeros((number_frames, self.number_patches, psf_size, psf_size), float32)
        self.frame_fluxes = zeros(number_frames, float32)
        self.frame_stack_warped = None

        for iteration in range(1, configuration.mfbd_iterations + 1):
            image_fft = rfft2(self.image_patches, workers=-1)

            # Remove the local tip-tilt of every frame.
            if configuration.mfbd_local_warp and (
                    iteration == 1 or (iteration - 1) % configuration.mfbd_local_refresh == 0):
                self.my_timer.start('MFBD: local shifts and warping')
                shifts, strong = self.measure_local_shifts(image_fft)
                self.frame_stack_warped, residual = self.warp_frames(shifts, strong)
                self.my_timer.stop('MFBD: local shifts and warping')
                self.protocol("           MFBD: local tip-tilt rms %.2f px over %d patches, "
                              "residual about the smooth field %.2f px" % (
                                  float(sqrt((shifts[:, strong] ** 2).sum(2).mean()))
                                  if strong.any() else 0., int(strong.sum()), residual), level=2)
            source = self.frame_stack_warped if self.frame_stack_warped is not None \
                else self.frame_stack

            self.my_timer.start('MFBD: PSFs and image updates')
            numerator = zeros((self.number_patches, fft_size, fft_size), float32)
            denominator = zeros((self.number_patches, fft_size, fft_size), float32)
            centroids = zeros((self.number_patches, 2))

            # The model of every frame blurred by the start PSF has the same flux: compute it once.
            if iteration == 1:
                start_model = self.convolve_patches(image_fft, rfft2(
                    broadcast_to(self.psf_start, (self.number_patches, fft_size, fft_size)),
                    workers=-1)) * self.valid_mask
                start_model_flux = max(float((start_model * self.valid_mask).sum()), 1.e-3)

            for frame_index in range(number_frames):
                frame_patches = zeros((self.number_patches, fft_size, fft_size), float32)
                frame_patches[:, psf_size - 1:psf_size - 1 + patch_size,
                              psf_size - 1:psf_size - 1 + patch_size] = \
                    self.patch_cutouts(source[frame_index], patch_size)

                # The frame's flux (transparency) is fixed in the first iteration: otherwise x * c,
                # f / c is a free direction and the image level drifts.
                if iteration == 1:
                    self.frame_fluxes[frame_index] = max(
                        float(((frame_patches - self.sky) * self.valid_mask).sum()), 1.) / \
                        start_model_flux
                    psf = zeros((self.number_patches, fft_size, fft_size), float32)
                    psf[:] = self.psf_start * self.frame_fluxes[frame_index]
                    iterations = configuration.mfbd_psf_iterations_first
                else:
                    psf = zeros((self.number_patches, fft_size, fft_size), float32)
                    psf[:, :psf_size, :psf_size] = self.psfs[frame_index]
                    iterations = configuration.mfbd_psf_iterations_later
                psf = self.estimate_psf(frame_patches, image_fft, psf, iterations,
                                        full(self.number_patches, self.frame_fluxes[frame_index],
                                             float32))
                self.psfs[frame_index] = psf[:, :psf_size, :psf_size]

                # Accumulate the PSF centroids, and the frame's Richardson-Lucy correction.
                psf_core = psf[:, :psf_size, :psf_size]
                psf_sums = maximum(psf_core.sum((1, 2)), EPS)
                centroids += stack([(psf_core * self.psf_offsets[0]).sum((1, 2)) / psf_sums,
                                    (psf_core * self.psf_offsets[1]).sum((1, 2)) / psf_sums],
                                   axis=1)
                psf_fft = rfft2(psf, workers=-1)
                model = self.convolve_patches(image_fft, psf_fft) + self.sky
                ratio = where(self.valid_mask > 0, frame_patches / maximum(model, EPS),
                              0).astype(float32)
                numerator += self.correlate_patches(rfft2(ratio, workers=-1), psf_fft)
                denominator += self.correlate_patches(self.valid_mask_fft[None], psf_fft)

            # Update the image once from all frames: clipped, and only where enough frames constrain
            # it (patch borders are covered by a few PSF tails only).
            ratio = clip(numerator / maximum(denominator, EPS), 1. / configuration.mfbd_ratio_clip,
                         configuration.mfbd_ratio_clip)
            constrained = denominator >= configuration.mfbd_min_support * \
                denominator.max(axis=(1, 2), keepdims=True)
            ratio = where(constrained, ratio, 1.)
            extent = patch_size + psf_size - 1
            self.image_patches[:, :extent, :extent] *= ratio[:, :extent, :extent]

            # Keep the PSFs centered: move each image patch by the mean PSF offset of its frames.
            mean_offsets = centroids / number_frames
            for patch_index in range(self.number_patches):
                self.image_patches[patch_index, :extent, :extent] = ndimage.shift(
                    self.image_patches[patch_index, :extent, :extent], mean_offsets[patch_index],
                    order=3, mode="nearest")
            clip(self.image_patches, 0, None, out=self.image_patches)
            self.my_timer.stop('MFBD: PSFs and image updates')

            self.protocol("           MFBD: iteration " + str(iteration) + " of " +
                          str(configuration.mfbd_iterations) + " done (%.1f s)" %
                          (time() - start_time), level=2)
            if self.progress_signal is not None:
                percent = int(round(100 * iteration / configuration.mfbd_iterations))
                self.progress_signal.emit("Multi-frame blind deconvolution", percent)

    def blend_patches(self):
        """
        Blend the image patches into one image with a Hann window. Where no patch contributes, the
        start image is used.

        :return: float32 image on the intersection canvas (without sky)
        """

        patch_size, psf_half_size = self.patch_size, self.psf_half_size
        window = outer(hanning(patch_size + 2)[1:-1], hanning(patch_size + 2)[1:-1]).astype(float32)
        accumulated = zeros((self.canvas_height, self.canvas_width), float32)
        weight_sum = zeros((self.canvas_height, self.canvas_width), float32)
        for patch_index, (y, x) in enumerate(self.patch_corners):
            accumulated[y:y + patch_size, x:x + patch_size] += \
                window * self.image_patches[patch_index, psf_half_size:psf_half_size + patch_size,
                                            psf_half_size:psf_half_size + patch_size]
            weight_sum[y:y + patch_size, x:x + patch_size] += window
        return where(weight_sum > 1.e-3, accumulated / maximum(weight_sum, 1.e-3),
                     self.start_image).astype(float32)

    def deconvolve(self):
        """
        Run MFBD on the intersection canvas.

        :return: Sky-free float32 luminance on the (binned) intersection canvas, in the frames'
                 units
        """

        self.my_timer.start('MFBD: reading and aligning frames')
        self.load_frames()
        self.set_up_patches()
        self.my_timer.stop('MFBD: reading and aligning frames')

        self.iterate()

        self.my_timer.start('MFBD: blending and transfer')
        self.luminance = self.blend_patches()
        self.my_timer.stop('MFBD: blending and transfer')
        return self.luminance

    def deconvolve_and_transfer(self, stack_frames):
        """
        Run MFBD and put the result onto the stacked image's canvas: the same borders, drizzle
        factor and brightness scale. Mono: the deconvolved image replaces the stack. Colour: the
        deconvolved luminance replaces the stack's R+G luminance, the colour ratios are kept.

        :param stack_frames: StackFrames object after "merge_alignment_point_buffers"
        :return: uint16 image with the shape of stack_frames.stacked_image
        """

        luminance = self.deconvolve() if self.luminance is None else self.luminance
        self.my_timer.start('MFBD: blending and transfer')
        luminance = self.upsample_to_canvas(luminance)
        stacked = stack_frames.stacked_image.astype(float32) / 65535.

        # Same geometry as the stacked image: drizzle, then the borders StackFrames trimmed.
        if self.configuration.drizzle_factor != 1:
            luminance = resize(luminance,
                               (stack_frames.dim_x_drizzled, stack_frames.dim_y_drizzled),
                               interpolation=INTER_CUBIC)
        luminance = luminance[
                    stack_frames.border_y_low:luminance.shape[0] - stack_frames.border_y_high,
                    stack_frames.border_x_low:luminance.shape[1] - stack_frames.border_x_high]
        if luminance.shape[:2] != stacked.shape[:2]:
            luminance = resize(luminance, (stacked.shape[1], stacked.shape[0]),
                               interpolation=INTER_CUBIC)

        # Brightness: match the stack's luminance above its own background, on the object.
        stacked_luminance = 0.5 * (stacked[:, :, 0] + stacked[:, :, 1]) if self.frames.color \
            else stacked
        background = float(percentile(stacked_luminance, 1))
        top = float(percentile(stacked_luminance, 99.9))
        mask = stacked_luminance > background + 0.15 * (top - background)
        if mask.sum() < 16:
            mask = ones_like(mask)
        gain = float(median(stacked_luminance[mask] - background)) / \
            max(float(median(luminance[mask])), EPS)
        new_luminance = luminance * gain + background

        # Colour: keep the stack's colour ratios, replace its luminance.
        if self.frames.color:
            eps = 0.02 * max(top, EPS)
            ratio = clip((new_luminance + eps) / (stacked_luminance + eps), 0., 4.)
            result = stacked * ratio[:, :, None]
        else:
            result = new_luminance
        self.my_timer.stop('MFBD: blending and transfer')
        self.protocol("           MFBD: result scaled to the stacked image (gain %.4g), " % gain +
                      ("colour taken from the stacked image" if self.frames.color else "mono"),
                      level=2)
        return img_as_uint(clip(result, 0., 1.))
