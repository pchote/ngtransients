#!/usr/bin/env python3

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""
   detect-variability.py

   This script compares each image in an action against a specified reference frame
   and searches for variable sources using the following technique:

   1. A background map is calculated using the reference image's background source mask to exclude
      pixels associated with known sources, and subtracted from the science frame.

   2. An estimate of the extinction relative to the reference frame is calculated using the
      reference image's mean source mask (which includes stars that are not too bright nor
      too faint).

   3. The reference image is scaled to match the science frame's extinction, then subtracted to
      create a difference / delta image.  The images are not regridded before subtracting, so
      constant sources do not cleanly subtract out.  The large undersampling and precise guiding
      of NGTS fields ensures that stars do not move more than 1 pixel relative to the reference.

   4. A source detection is done on the absolute value of the delta frame, using a threshold
      tunable from the --detection-sigma argument.  The absolute value is used to ensure that
      both the positive and negative parts of the delta-psf are detected.

   5. For each detected source, the delta-flux, science flux, and read/photon shot noise are
      integrated over all pixels within the rectangular bounding box covering the source detection.

   5a.   Detected sources are discarded if the absolute delta-flux is less than a minimum
         absolute threshold, tunable from the --delta-minimum-threshold argument.  This avoids
         false positives from faint objects due to uncertainties in the background estimation.

   5b.   Detected sources are discarded if the delta-flux is less than a minimum percentage of
         the original reference flux, tunable from the --delta-percentage-threshold argument.
         This avoids false positives from bright objects due to uncertainties in the extinction
         estimation.

   6. The output file consists of a primary fits extension with header keys defining reduction
      properties and a BINTABLE extension containing the actual detections.
      Extension 0 header keys:
          REFMASK: The filename of the reference image used.
         ACTIONID: The integer id of the action reduced.
            NIGHT: YYYYMMDD date of the night of the action reduced.
         DETSIGMA: The value of the --detection-sigma argument used.
           MINTHR: The value of the --delta-minimum-threshold argument used.
          PCNTTHR: The value of the --delta-percentage-threshold argument used.
          CCDGAIN: The value of the --ccd-gain argument used.
          CCDREAD: The value of the --ccd-readnoise argument used.

      Extension 1 BINTABLE columns:
                filename: The name of the science image that the detection was made.
               timestamp: The unix timestamp defining the start time of the science image.
                       x: The x position of the center of the detected source.
                       y: The y position of the center of the detected source.
              delta_flux: The integrated delta-flux of the detected source.
        delta_flux_error: The uncertainty (read noise + photon shot noise) in the delta-flux.
"""

# pylint: disable=invalid-name
# pylint: disable=too-many-statements
# pylint: disable=too-many-branches
# pylint: disable=too-many-locals
# pylint: disable=too-many-arguments

import argparse as ap
import bz2
import datetime
import glob
import math
import numpy as np
import os
import sep
import sys
import time
import astropy.io.fits as fits

# pyds9 isn't available or wanted in the cluster environment
try:
    import pyds9
except ImportError:
    pass

class ProgressNotifier:
    """Helper class for managing status output in realtime to either a tty or log file"""
    def __init__(self, steps, no_tty_percent_step=5):
        self._steps = steps
        self._isatty = sys.stdout.isatty()
        self._no_tty_percent_step = no_tty_percent_step
        self._percent_step = 0

    def update(self, message, step):
        """Updates the progress indicator"""
        percent = round((step + 1) * 100. / self._steps)
        percent_step = int(percent / self._no_tty_percent_step)

        # ttys are updated immediately, redirected outputs are updated once per percentage step
        if self._isatty:
            sys.stdout.write('\r')
        elif percent_step <= self._percent_step:
            return

        sys.stdout.write(message)
        sys.stdout.write(': {} / {} ({}%)'.format(step + 1, self._steps, percent))
        if not self._isatty:
            sys.stdout.write('\n')

        sys.stdout.flush()
        self._percent_step = percent_step

    def message(self, message):
        """Inserts a message into the log output"""
        if self._isatty:
            sys.stdout.write('\n')
        sys.stdout.write(message + '\n')
        sys.stdout.flush()

    def complete(self):
        """Notify that the action is complete and should be cleaned up"""
        if self._isatty:
            sys.stdout.write('\n')
        sys.stdout.flush()

class DetectionTable:
    """Helper class for managing the table of detected objects"""
    def __init__(self):
        # PERF: astropy's Table.add_row is a serious bottleneck
        # so we manage our own column state and then convert to a BinTableHDU at the end
        self._filename = []
        self._timestamp = []
        self._x = []
        self._y = []
        self._threshold = []
        self._delta_flux = []
        self._delta_flux_error = []

    def add_row(self, filename, timestamp, x, y, threshold, delta_flux, delta_flux_error):
        """Adds a detection row to the table"""
        self._filename.append(filename)
        self._timestamp.append(timestamp)
        self._x.append(x)
        self._y.append(y)
        self._threshold.append(threshold)
        self._delta_flux.append(delta_flux)
        self._delta_flux_error.append(delta_flux_error)

    def to_hdu(self):
        """Exports table data as a fits binary table extension"""
        return fits.BinTableHDU.from_columns([
            fits.Column(name='filename', format='32A', array=self._filename),
            fits.Column(name='timestamp', format='K', array=self._timestamp),
            fits.Column(name='x', format='D', array=self._x),
            fits.Column(name='y', format='D', array=self._y),
            fits.Column(name='detection_threshold', format='D', array=self._threshold),
            fits.Column(name='delta_flux', format='D', array=self._delta_flux),
            fits.Column(name='delta_flux_error', format='D', array=self._delta_flux_error),
        ])

def make_timestamp(header_datestring):
    """Parses time extracted from fits header as a POSIX timestamp"""
    date_datetime = datetime.datetime.strptime(header_datestring, "'%Y-%m-%dT%H:%M:%S'")
    return int(time.mktime((date_datetime.year, date_datetime.month, date_datetime.day,
                            date_datetime.hour, date_datetime.minute, date_datetime.second,
                            -1, -1, -1)) + date_datetime.microsecond / 1e6)

def load_bz2_image(path):
    """Load a bzipped fits image and return the image data as a 2D array of doubles plus a
       minimal set of header keys."""
    header = {}
    # PERF: astropy.io.fits parsing is far too slow for our purposes
    # The data layout is consistent across frames, so load directly from known byte offsets
    with bz2.BZ2File(path) as im:
        # Read the header keywords that we are interested in from the start of the stream
        im.seek(0x1540+9)
        header['OBSSTART'] = im.read(71).decode('ascii').split()[0]

        im.seek(0x29e0+9)
        header['AG_ERRX'] = float(im.read(71).decode('ascii').split()[0])

        im.seek(0x2a30+9)
        header['AG_ERRY'] = float(im.read(71).decode('ascii').split()[0])

        im.seek(0x2c60+9)
        header['AGREFIMG'] = int(im.read(71).decode('ascii').split()[0])

        # Finally the image data
        im.seek(4*2880)
        data = np.frombuffer(im.read(4*2088*2048), dtype=np.dtype('>i4')).reshape((2048, 2088))

    # Convert to a float array for use by our algorithms
    # pylint: disable=no-member
    data = data.astype(np.float64, copy=False)
    # pylint: enable=no-member

    # Add the BZERO offset
    data += 2147483648
    return (data, header)

def reduce_night(actionid, reference_path, output_path, config, action_frame_path,
                 ds9_title, ds9_update_rate):
    """Core analysis function, operating as described in the module description"""
    start_time = datetime.datetime.utcnow()

    frame_glob = os.path.join(action_frame_path, 'action{0}_observeField/'.format(actionid),
                              '*.fits.bz2')
    frames = sorted(glob.glob(frame_glob))

    if len(frames) == 0:
        print('No frames found for action {0}'.format(action_id))
        return

    # pylint: disable=no-member
    with fits.open(reference_path) as r:
        reference_agrefimg = r[0].header['AGREFIMG']

        # Assume the reference was created from 25 source images unless told otherwise
        reference_frame_count = r[0].header['FRAMECNT'] if 'FRAMECNT' in r[0].header else 25

        reference_image = r[0].data.astype(np.float64)
        bright_star_mask = r[1].data.astype(np.float64)
        background_mask = r[2].data.astype(np.float64)
        mean_mask = r[3].data.astype(np.float64)
    # pylint: enable=no-member

    reference_mean = np.ma.mean(np.ma.array(reference_image, mask=mean_mask))
    overscan_mask = np.ones(np.shape(background_mask))
    overscan_mask[3:2047, 20:2067] = 0

    ds9_window = None
    try:
        if ds9_title is not None:
            ds9_window = pyds9.DS9(ds9_title)
    except NameError:
        print('WARNING: pyds9 is not available. DS9 previews disabled')

    with fits.open(frames[0]) as nightframe:
        nightdate = nightframe[0].header['NIGHT']

    print('Reducing action {} ({})'.format(actionid, nightdate))

    detections = DetectionTable()
    progress = ProgressNotifier(len(frames))
    for i, f in enumerate(frames):
        fname = os.path.basename(f)
        progress.update('Reducing ' + fname, i)

        # pylint: disable=broad-except
        try:
            science_image, header = load_bz2_image(f)
        except Exception as e:
            progress.message('{} failed to parse: {}'.format(f, e))
            continue
        # pylint: enable=broad-except

        if reference_agrefimg != header['AGREFIMG']:
            progress.message('{} incorrect reference: {} != {}'.format(fname, header['AGREFIMG'],
                                                                       reference_agrefimg))
            continue

        # Ignore any images with more than 1px offset (AG_ERR* is defined in arcseconds)
        if abs(header['AG_ERRX']) > 5 or abs(header['AG_ERRY']) > 5:
            progress.message('{} offset too large: {}, {}'.format(fname,
                                                                  round(header['AG_ERRX'], 2),
                                                                  round(header['AG_ERRY'], 2)))
            continue

        # Subtract bias/dark/sky background from the science image
        science_background = sep.Background(science_image, mask=background_mask)
        science_image -= science_background

        # Estimate the extinction in the science image and rescale to match the reference
        science_mean = np.ma.mean(np.ma.array(science_image, mask=mean_mask))
        science_image *= reference_mean / science_mean

        # Search for objects in the difference image
        # We are deliberately not masking out bright star pixels here because that can lead to
        # spurious detections at the edge of the mask.  It is more reliably to find the true
        # unmasked object positions and then check against the mask later
        delta_image = science_image - reference_image
        delta_threshold = config['detection_sigma'] * science_background.globalrms
        delta_objects = sep.extract(np.abs(delta_image), delta_threshold, mask=overscan_mask)

        update_ds9 = ds9_window is not None and i % ds9_update_rate == 0
        if update_ds9:
            ds9_window.set_np2arr(delta_image)

        for o in delta_objects:
            x1 = o['xmin']
            x2 = o['xmax']+1
            y1 = o['ymin']
            y2 = o['ymax']+1

            # Ignore anything that overlaps with a bright star
            if np.any(bright_star_mask[y1:y2, x1:x2]):
                continue

            # Integrate flux in the rectangular region covering the
            # positive and negative pixels in the difference image
            delta_flux = np.sum(delta_image[y1:y2, x1:x2])
            delta_flux_variance = 0

            orig_flux = np.sum(science_image[y1:y2, x1:x2])
            rdsq = config['ccd_readnoise'] ** 2
            gain = config['ccd_gain']
            for y in range(y1, y2 + 1):
                for x in range(x1, x2 + 1):
                    # Error comes from the readnoise and photon noise in the unsubtracted frame
                    delta_flux_variance += rdsq + science_image[y, x] / gain

                    # Error from the reference frame
                    delta_flux_variance += (rdsq + reference_image[y, x] / gain) / \
                        reference_frame_count

            delta_flux_error = math.sqrt(delta_flux_variance)

            # Flag objects with integrated delta-flux above the threshold
            fractional_threshold = config['delta_percentage_threshold'] / 100

            if np.abs(delta_flux) > config['delta_minimum_threshold'] and \
                    np.abs(delta_flux / orig_flux) > fractional_threshold:
                detection_x = o['x'] + header['AG_ERRX'] / 5
                detection_y = o['y'] + header['AG_ERRY'] / 5

                detections.add_row(os.path.basename(frames[i]), make_timestamp(header['OBSSTART']),
                                   detection_x, detection_y, o['thresh'], delta_flux,
                                   delta_flux_error)

                if update_ds9:
                    ds9_window.set('regions',
                                   'image; box({0},{1},{2},{3})'.format((x1 + x2) / 2 + 1,
                                                                        (y1 + y2) / 2 + 1,
                                                                        x2 - x1, y2 - y1))
    progress.complete()

    primary_hdu = fits.PrimaryHDU()
    primary_hdu.header['REFMASK'] = reference_path
    primary_hdu.header['ACTIONID'] = actionid
    primary_hdu.header['NIGHT'] = nightdate
    primary_hdu.header['DETSIGMA'] = config['detection_sigma']
    primary_hdu.header['MINTHR'] = config['delta_minimum_threshold']
    primary_hdu.header['PCNTTHR'] = config['delta_percentage_threshold']
    primary_hdu.header['CCDGAIN'] = config['ccd_gain']
    primary_hdu.header['CCDREAD'] = config['ccd_readnoise']

    # pylint: disable=unexpected-keyword-arg
    fits.HDUList([
        primary_hdu,
        detections.to_hdu()
    ]).writeto(output_path, clobber=True)
    # pylint: enable=unexpected-keyword-arg

    print('Completed in {}'.format(datetime.datetime.utcnow() - start_time))

if __name__ == "__main__":
    parser = ap.ArgumentParser(description="Detect variable objects in a NGTS observation action.")
    parser.add_argument('action',
                        type=int,
                        help='integer action ID.')
    parser.add_argument('reference',
                        type=str,
                        help='Path to the reference frame.')
    parser.add_argument('output',
                        type=str,
                        help='Path to save the output detections.')
    parser.add_argument('--detection-sigma',
                        type=float,
                        default=3,
                        help='Threshold for detecting objects in the absolute-difference image.')
    parser.add_argument('--delta-minimum-threshold',
                        type=float,
                        default=300,
                        help='Absolute minimum integrated delta to count as a detection.')
    parser.add_argument('--delta-percentage-threshold',
                        type=float,
                        default=5,
                        help='Minimum % change from the reference to count as a detection.')
    parser.add_argument('--ccd-gain',
                        type=float,
                        default=2.0,
                        help='Gain applied during CCD readout.')
    parser.add_argument('--ccd-readnoise',
                        type=float,
                        default=13.0,
                        help='Readout noise in ADU.')
    parser.add_argument('--action-frame-path',
                        type=str,
                        default='/ngts/raw/01',
                        help='Path to the directory where the actions are stored.')
    parser.add_argument('--ds9-title',
                        type=str,
                        default=None,
                        help='Display running progress in a DS9 window.')
    parser.add_argument('--ds9-update-rate',
                        type=int,
                        default=5,
                        help='Number of frames between updating DS9 window.')

    args = parser.parse_args()
    detection_args = {
        'detection_sigma': args.detection_sigma,
        'delta_minimum_threshold': args.delta_minimum_threshold,
        'delta_percentage_threshold': args.delta_percentage_threshold,
        'ccd_gain': args.ccd_gain,
        'ccd_readnoise': args.ccd_readnoise
    }

    reduce_night(args.action, args.reference, args.output, detection_args, args.action_frame_path,
                 args.ds9_title, args.ds9_update_rate)

