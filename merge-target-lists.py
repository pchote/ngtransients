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

   This script merges target list files generated by generate-target-list.py

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
        self._filename.append(d[2])
        self._timestamp.append(d[3])
        self._x.append(d[4])
        self._y.append(d[5])
        self._threshold.append(d[6])
        self._delta_flux.append(d[7])
        self._delta_flux_error.append(d[8])

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

def merge_target_lists(input_paths, output_path,  merge_distance):
    """Core analysis function, operating as described in the module description"""
    start_time = datetime.datetime.utcnow()
    reference_name = None

    # Peek the reference frame from the first file
    # pylint: disable=no-member
    with fits.open(input_paths[0]) as testframe:
        reference_name = testframe[0].header['REFMASK']
        grid_size = testframe[0].header['GRIDSIZE']
        score_threshold = testframe[0].header['SCTHRESH']
        field = testframe[0].header['FIELD']
    # pylint: enable=no-member

    targets_table = Table(names=('id', 'x', 'y', 'reference_flux'), dtype=('i4', 'f8', 'f8', 'f8'))
    action_targets_table = Table(names=('id', 'action', 'x', 'y', 'bin_x1', 'bin_x2',
                                        'bin_y1', 'bin_y2', 'score'),
                                 dtype=('i4', 'i4', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8'))
    action_night_table = Table(names=('action', 'night', 'mjd'), dtype=('i4', 'i4', 'f8'))

    total_input = len(input_paths)

    detections_table = DetectionTable()
    for k, targets_path in enumerate(input_paths):
        # pylint: disable=no-member
        with fits.open(targets_path) as action:
            targets = action[1].data
            action_targets = action[2].data
            action_detections = action[3].data
            action_nights = action[4].data

            if reference_name != action[0].header['REFMASK']:
                raise Exception('REFMASK mismatch for ' + targets_path)

            if grid_size != action[0].header['GRIDSIZE']:
                raise Exception('GRIDSIZE mismatch for ' + targets_path)

            if score_threshold != action[0].header['SCTHRESH']:
                raise Exception('SCTHRESH mismatch for ' + targets_path)

            if len(action_targets) > 100:
                print('ignoring {} with {}'.format(targets_path, len(action_targets)))
                continue
        # pylint: enable=no-member

        for n in action_nights:
            action_night_table.add_row((n['action'], n['night'], n['mjd']))


        percent = round(k * 100. / total_input)
        print('Parsing {0}: {1} / {2} ({3}%)'.format(targets_path, k + 1, total_input, percent))
        for t in action_targets:
            target_id = len(targets_table)
            filter_nearby_x = abs(targets_table['x'] - t['x']) <= merge_distance
            filter_nearby_y = abs(targets_table['y'] - t['y']) <= merge_distance
            nearby_targets = targets_table[np.logical_and(filter_nearby_x, filter_nearby_y)]
            reference_flux = targets[t['id']]['reference_flux']

            if len(nearby_targets) == 0:
                targets_table.add_row((target_id, t['x'], t['y'], reference_flux))
            else:
                # TODO: It would be more correct to set based on nearest, not first
                target_id = nearby_targets[0]['id']

            action_targets_table.add_row((target_id, t['action'], t['x'], t['y'],
                                          t['bin_x1'], t['bin_x2'], t['bin_y1'], t['bin_y2'],
                                          t['score']))

            # Copy detections, replacing the old target id with new
            for d in action_detections[action_detections['id'] == t['id']]:
                detections_table.add_row(target_id, t['action'], d)

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
    parser = ap.ArgumentParser(description="Aggregates target lists.")
    parser.add_argument('output',
                        type=str,
                        help='Path to save the output fits file')
    parser.add_argument('input',
                        type=str,
                        nargs='+',
                        help='Input fits files generated by generate-target-list.py')
    parser.add_argument('--merge-distance',
                        type=int,
                        default=5,
                        help='Merge targets across multiple nights if closer than this')
    args = parser.parse_args()
    merge_target_lists(args.input, args.output, args.merge_distance)

