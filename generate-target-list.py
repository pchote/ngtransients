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
   generate-target-list.py

   This script aggregates over one or more detection files from detect-variability.py, searching
   for detections that are clustered in both position and time.  Each detection is scored with a
   weight that is inversely proportional to the time since the last detection, weighted so that
   subsequent detections will incremement the score by 1 each.

"""

# pylint: disable=invalid-name
# pylint: disable=too-many-arguments
# pylint: disable=too-many-locals
# pylint: disable=too-many-instance-attributes

import argparse as ap
import datetime
import numpy as np
import os
import sys
import sep
import astropy.io.fits as fits
import astropy.time as time
from astropy.table import Table

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

class ScoreGrid:
    """Helper class for quantizing positions into a regular grid and accumulating running scores"""
    def __init__(self, name, x1, x2, y1, y2, size):
        if (x2 - x1) % size != 0 or (y2 - y1) % size != 0:
            raise Exception('Search window must be an integer multiple of the grid size')

        self._name = name
        self._x1 = x1
        self._x2 = x2
        self._y1 = y1
        self._y2 = y2
        self._size = size

        self._bins_x = (x2 - x1) // size
        self._bins_y = (y2 - y1) // size
        self._score = np.zeros((self._bins_y, self._bins_x))
        self._detections = [[[] for x in range(self._bins_x)] for y in range(self._bins_y)]

    def add_detection(self, d):
        """ Adds a detection to the score grid """
        if d['x'] < self._x1 or d['x'] >= self._x2 or d['y'] < self._y1 or d['y'] >= self._y2:
            return

        b_x = int((d['x'] - self._x1) / self._size)
        b_y = int((d['y'] - self._y1) / self._size)

        if len(self._detections[b_y][b_x]) == 0:
            last_timestamp = 0
        else:
            last_timestamp = self._detections[b_y][b_x][-1]['timestamp']

        # Only update the score in a bin once per frame
        if last_timestamp < d['timestamp']:
            score = 10. / (d['timestamp'] - last_timestamp)
            self._score[b_y, b_x] += score

        self._detections[b_y][b_x].append(d)

    def apertures_above_score(self, threshold):
        """ Returns a list of targets that have a detection score above a given threshold """
        detections = []
        progress = ProgressNotifier(self._bins_y * self._bins_x)
        for b_y in range(self._bins_y):
            for b_x in range(self._bins_x):
                i = b_y * self._bins_x + b_x
                progress.update('Enumerating ' + self._name, i)

                score = self._score[b_y, b_x]
                if score > threshold:
                    median_x = np.median([d['x'] for d in self._detections[b_y][b_x]])
                    median_y = np.median([d['y'] for d in self._detections[b_y][b_x]])

                    frame_data = []
                    for d in self._detections[b_y][b_x]:
                        frame_data.append((d['filename'], d['timestamp'], d['x'], d['y'],
                                           d['detection_threshold'],
                                           d['delta_flux'], d['delta_flux_error']))

                    detections.append({
                        'median_x': median_x,
                        'median_y': median_y,
                        'score': score,
                        'bin_x1': self._x1 + b_x * self._size,
                        'bin_x2': self._x1 + (b_x + 1) * self._size,
                        'bin_y1': self._y1 + b_y * self._size,
                        'bin_y2': self._y1 + (b_y + 1) * self._size,
                        'frame_data': frame_data,
                    })

        progress.complete()
        return detections

class DetectionTable:
    """Helper class for managing the table of detected objects"""
    def __init__(self):
        # PERF: astropy's Table.add_row is a serious bottleneck
        # so we manage our own column state and then convert to a BinTableHDU at the end
        self._target_id = []
        self._action_id = []
        self._filename = []
        self._timestamp = []
        self._x = []
        self._y = []
        self._threshold = []
        self._delta_flux = []
        self._delta_flux_error = []

    def add_row(self, target_id, action_id, d):
        """Adds a detection generated by find_targets to the table"""
        self._target_id.append(target_id)
        self._action_id.append(action_id)

        # HACK: These should be indexed by name to improve robustness
        self._filename.append(d[0])
        self._timestamp.append(d[1])
        self._x.append(d[2])
        self._y.append(d[3])
        self._threshold.append(d[4])
        self._delta_flux.append(d[5])
        self._delta_flux_error.append(d[6])

    def to_hdu(self):
        """Exports table data as a fits binary table extension"""
        return fits.BinTableHDU.from_columns([
            fits.Column(name='id', format='K', array=self._target_id),
            fits.Column(name='action', format='K', array=self._action_id),
            fits.Column(name='filename', format='32A', array=self._filename),
            fits.Column(name='timestamp', format='K', array=self._timestamp),
            fits.Column(name='x', format='D', array=self._x),
            fits.Column(name='y', format='D', array=self._y),
            fits.Column(name='detection_threshold', format='D', array=self._threshold),
            fits.Column(name='delta_flux', format='D', array=self._delta_flux),
            fits.Column(name='delta_flux_error', format='D', array=self._delta_flux_error),
        ])

def find_targets(detections, score_threshold, grid_size):
    """ Search for clustered detections in both time and position """
    # Search for clusted detections using 4 overlapping grids offset each way by half a cell
    # This avoids any issues with lost detections due to jitter across a grid cell
    # Detections are deduplicated at the end by taking the one with the highest score
    left = 20 + grid_size
    right = 2068 - grid_size
    bottom = grid_size
    top = 2048 - grid_size
    offset = grid_size // 2

    a = ScoreGrid('grid 1/4', left, right, bottom, top, grid_size)
    b = ScoreGrid('grid 2/4', left, right, bottom + offset, top - offset, grid_size)
    c = ScoreGrid('grid 3/4', left + offset, right - offset, bottom, top, grid_size)
    d = ScoreGrid('grid 4/4', left + offset, right - offset, bottom + offset, top - offset,
                  grid_size)

    progress = ProgressNotifier(len(detections))
    for i, detection in enumerate(detections):
        progress.update('Enumerating detections', i)

        a.add_detection(detection)
        b.add_detection(detection)
        c.add_detection(detection)
        d.add_detection(detection)
    progress.complete()

    apertures = a.apertures_above_score(score_threshold)
    apertures.extend(b.apertures_above_score(score_threshold))
    apertures.extend(c.apertures_above_score(score_threshold))
    apertures.extend(d.apertures_above_score(score_threshold))

    deduplicated_apertures = []
    progress = ProgressNotifier(len(apertures))
    # This could be done more efficiently, but is already fast enough
    for j, a in enumerate(apertures):
        progress.update('Processing', j)

        append_aperture = True
        # Check whether we have already seen another aperture within the search radius
        for i, b in enumerate(deduplicated_apertures):
            dx = abs(a['median_x'] - b['median_x'])
            dy = abs(a['median_y'] - b['median_y'])
            if dx <= grid_size // 2 and dy <= grid_size // 2:
                append_aperture = False
                if a['score'] > b['score']:
                    # We are better than the existing aperture, so replace it
                    deduplicated_apertures[i] = a
                    continue

                # We are no better than the existing apeture, so we can ignore this one
                break

        if append_aperture:
            deduplicated_apertures.append(a)

    progress.complete()
    return deduplicated_apertures

def generate_target_list(input_paths, output_path, score_threshold, grid_size, merge_distance,
                         reference_frame_path, reference_aperture_radius):
    """Core analysis function, operating as described in the module description"""
    start_time = datetime.datetime.utcnow()
    reference_name = None

    # Peek the reference frame from the first file
    # pylint: disable=no-member
    with fits.open(input_paths[0]) as testframe:
        reference_name = testframe[0].header['REFMASK']

    with fits.open(os.path.join(reference_frame_path, reference_name)) as reference:
        reference_image = reference[0].data.astype(np.float64)
        field = reference[0].header['FIELD']
    # pylint: enable=no-member

    targets_table = Table(names=('id', 'x', 'y', 'reference_flux'), dtype=('i4', 'f8', 'f8', 'f8'))
    action_targets_table = Table(names=('id', 'action', 'x', 'y', 'bin_x1', 'bin_x2',
                                        'bin_y1', 'bin_y2', 'score'),
                                 dtype=('i4', 'i4', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8'))
    action_night_table = Table(names=('action', 'night', 'mjd'), dtype=('i4', 'i4', 'f8'))

    total_input = len(input_paths)

    detections_table = DetectionTable()
    for k, action_path in enumerate(input_paths):
        # pylint: disable=no-member
        with fits.open(action_path) as action:
            action_detections = action[1].data
            action_id = action[0].header['ACTIONID']
            action_night = action[0].header['NIGHT']
            action_mjd = time.Time(datetime.datetime.strptime(action_night, "%Y%m%d"),
                                   scale='utc').mjd
            if reference_name != action[0].header['REFMASK']:
                raise Exception('REFMASK mismatch for ' + action_path)

            action_night_table.add_row((action_id, action_night, action_mjd))
        # pylint: enable=no-member

        percent = round(k * 100. / total_input)
        print('Parsing {0}: {1} / {2} ({3}%)'.format(action_path, k + 1, total_input, percent))
        action_targets = find_targets(action_detections, score_threshold, grid_size)
        for t in action_targets:
            target_id = len(targets_table)
            filter_nearby_x = abs(targets_table['x'] - t['median_x']) <= merge_distance
            filter_nearby_y = abs(targets_table['y'] - t['median_y']) <= merge_distance
            nearby_targets = targets_table[np.logical_and(filter_nearby_x, filter_nearby_y)]

            reference_flux, _, _ = sep.sum_circle(reference_image, t['median_x'], t['median_y'],
                                                  reference_aperture_radius, gain=1.0)

            if len(nearby_targets) == 0:
                targets_table.add_row((target_id, t['median_x'], t['median_y'], reference_flux))
            else:
                # TODO: It would be more correct to set based on nearest, not first
                target_id = nearby_targets[0]['id']

            action_targets_table.add_row((target_id, action_id, t['median_x'], t['median_y'],
                                          t['bin_x1'], t['bin_x2'], t['bin_y1'], t['bin_y2'],
                                          t['score']))

            for d in t['frame_data']:
                detections_table.add_row(target_id, action_id, d)

    primary_hdu = fits.PrimaryHDU()
    primary_hdu.header['REFMASK'] = reference_name
    primary_hdu.header['GRIDSIZE'] = grid_size
    primary_hdu.header['SCTHRESH'] = score_threshold
    primary_hdu.header['FIELD'] = field

    hdus = [
        primary_hdu,
        fits.table_to_hdu(targets_table),
        fits.table_to_hdu(action_targets_table),
        detections_table.to_hdu(),
        fits.table_to_hdu(action_night_table)
    ]

    fits.HDUList(hdus).writeto(output_path, overwrite=True)
    print('Completed in {}'.format(datetime.datetime.utcnow() - start_time))

if __name__ == "__main__":
    parser = ap.ArgumentParser(description="Aggregates detection files into a unique targets list.")
    parser.add_argument('output',
                        type=str,
                        help='Path to save the output fits file')
    parser.add_argument('input',
                        type=str,
                        nargs='+',
                        help='Input fits files generated by detect-variability.py')
    parser.add_argument('--score-threshold',
                        type=int,
                        default=30,
                        help='Score threshold for valid targets')
    parser.add_argument('--grid-size',
                        type=int,
                        default=8,
                        help='Size of score grid in pixels')
    parser.add_argument('--merge-distance',
                        type=int,
                        default=5,
                        help='Merge targets across multiple nights if closer than this')
    parser.add_argument('--reference-frame-path',
                        type=str,
                        default='.',
                        help='Path to the directory where reference frames are stored')
    parser.add_argument('--aperture-radius',
                        type=float,
                        default=2.5,
                        help='Aperture radius to use for calculating the target reference flux.')
    args = parser.parse_args()
    generate_target_list(args.input, args.output, args.score_threshold, args.grid_size,
                         args.merge_distance, args.reference_frame_path, args.aperture_radius)

