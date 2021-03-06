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
   generate-lightcurve.py

   This script generates light curves for the positions listed in the targets table of the input
   file, generated by generate-target-list.py.  Fixed circular apertures are placed at fixed
   locations relative to the reference frame, using the autoguider offsets in each image.
   They are not recentered or resized to match the seeing.  Output times are given in BJD_TDB
   (relative to the solar system barycenter, in barycentric dynamical time)

   Optional flags allow the fixed aperture size to be customized and lightcurves to be generated
   for only a single action.
"""

# pylint: disable=invalid-name
# pylint: disable=too-many-arguments
# pylint: disable=too-many-locals
# pylint: disable=too-many-instance-attributes

import argparse as ap
import bz2
import datetime
import glob
import numpy as np
import os
import sep
import sys
import astropy.io.fits as fits
import astropy.time as time
import astropy.units as u
import astropy.coordinates as coords

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

def generate_action_lightcurve(output_path, input_path, action_id, action_frame_path,
                               reference_frame_path, aperture_radius):
    start_time = datetime.datetime.utcnow()

    # Parse targets and actions from the input target file
    with fits.open(input_path) as input_targets:
        # pylint: disable=no-member
        header = input_targets[0].header
        targets = input_targets[1].data
        action_nights = input_targets[4].data
        # pylint: enable=no-member

    # Has the user asked to reduce only a specific night?
    if action_id >= 0 and action_id not in action_nights['action']:
        raise Exception('Action {0} not input data'.format(action_id))

    action_ids = action_nights['action'] if action_id < 0 else [action_id]
    apertures = np.array([(t['x'], t['y']) for t in targets])
    with fits.open(os.path.join(reference_frame_path, header['REFMASK'])) as reference:
        # pylint: disable=no-member
        reference_agrefimg = reference[0].header['AGREFIMG']
        field_ra = reference[0].header['CMD_HMS']
        field_dec = reference[0].header['CMD_DMS']

        site_lon = coords.Longitude(reference[0].header['SITELONG'], unit=u.deg)
        site_lat = coords.Latitude(reference[0].header['SITELAT'], unit=u.deg)
        site_alt = float(reference[0].header['SITEALT']) * u.meter

        reference_image = reference[0].data.astype(np.float64)
        background_mask = reference[2].data.astype(np.float64)
        mean_mask = reference[3].data.astype(np.float64)
        # pylint: enable=no-member

    field_coords = coords.SkyCoord(field_ra, field_dec, unit=(u.hourangle, u.deg), frame='icrs')
    paranal = coords.EarthLocation.from_geodetic(site_lon, site_lat, site_alt)

    reference_mean = np.ma.mean(np.ma.array(reference_image, mask=mean_mask))
    overscan_mask = np.ones(np.shape(background_mask))
    overscan_mask[3:2047, 20:2067] = 0

    results = []
    for action_id in action_ids:
        frame_glob = os.path.join(action_frame_path, 'action{0}_observeField/'.format(action_id),
                                  '*.fits.bz2')

        frames = sorted(glob.glob(frame_glob))
        if len(frames) == 0:
            print('No frames found for action {0}'.format(action_id))
            continue

        print('Reducing action {0}'.format(action_id))
        progress = ProgressNotifier(len(frames))
        for i, f in enumerate(frames):
            progress.update('Reducing frames', i)

            try:
                science_image, header = load_bz2_image(f)
            except Exception as e:
                progress.message('{} failed to parse: {}'.format(f, e))
                continue

            if reference_agrefimg != header['AGREFIMG']:
                progress.message('{} incorrect reference: {} != {}'.format(f, header['AGREFIMG'],
                                                                           reference_agrefimg))
                continue

            # Ignore any images with more than 1px offset (AG_ERR* is defined in arcseconds)
            if abs(header['AG_ERRX']) > 5 or abs(header['AG_ERRY']) > 5:
                progress.message('{} offset too large: {}, {}'.format(f,
                                                                      round(header['AG_ERRX'], 2),
                                                                      round(header['AG_ERRY'], 2)))
                continue

            # Subtract bias/dark/sky background from the science image
            science_background = sep.Background(science_image, mask=background_mask)
            science_image -= science_background

            # Estimate the extinction in the science image and rescale to match the reference
            science_mean = np.ma.mean(np.ma.array(science_image, mask=mean_mask))
            science_image *= reference_mean / science_mean

            apertures_x = apertures[:, 0] - header['AG_ERRX'] / 5
            apertures_y = apertures[:, 1] - header['AG_ERRY'] / 5

            # TODO: Calculate more realistic errors by accounting for read noise and gain
            flux, fluxerr, _ = sep.sum_circle(science_image, apertures_x, apertures_y,
                                              aperture_radius, err=science_background.globalrms,
                                              gain=1.0)

            # Calculate and apply barycentric offset
            frame_time = time.Time(header['OBSSTART'][1:-1], scale='utc', location=paranal)
            barycentric_offset = frame_time.light_travel_time(field_coords)
            row = [(frame_time.tdb + barycentric_offset).mjd]
            for i, f in enumerate(flux):
                row.extend((f, fluxerr[i]))

            results.append(row)
        progress.complete()

    np.savetxt(output_path, np.array(results))
    print('Completed in {}'.format(datetime.datetime.utcnow() - start_time))

if __name__ == "__main__":
    parser = ap.ArgumentParser(description="Detect variable objects in a NGTS observation action.")
    parser.add_argument('output',
                        type=str,
                        help='Path to save the output fits file.')
    parser.add_argument('input',
                        type=str,
                        help='Input fits files generated by generate-target-list.py.')
    parser.add_argument('--action-id',
                        type=int,
                        default=-1,
                        help='Generates the lightcurve for a specific action only.')
    parser.add_argument('--action-frame-path',
                        type=str,
                        default='/ngts/raw/01',
                        help='Path to the directory where the actions are stored.')
    parser.add_argument('--reference-frame-path',
                        type=str,
                        default='.',
                        help='Path to the directory where reference frames are stored')
    parser.add_argument('--aperture-radius',
                        type=float,
                        default=2.5,
                        help='Aperture radius to use.')
    args = parser.parse_args()
    generate_action_lightcurve(args.output, args.input, args.action_id, args.action_frame_path,
                               args.reference_frame_path, args.aperture_radius)

