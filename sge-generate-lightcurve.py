#!/usr/local/python/bin/python

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
   sge-generate-lightcurve.py

   Helper script to spread lightcurve generation across multiple nodes
"""

# pylint: disable=invalid-name

import argparse as ap
import astropy.io.fits as fits
import subprocess

def reduce_nights(targets_file, output_suffix, action_frame_path, reference_frame_path,
                  aperture_radius):
    with fits.open(targets_file) as data:
        # pylint: disable=no-member
        field = data[0].header['FIELD']
        action_nights = data[4].data
        # pylint: enable=no-member

    suffix = '' if len(output_suffix) == 0 else '-' + output_suffix

    for actionid in action_nights['action']:
        cmd = list([str(x) for x in [
            'qsub', '-N', 'transientlightcurve-{0}-{1}'.format(field, actionid),
            '-S', '/usr/local/python/bin/python',
            '-o', '.',
            '-pe', 'parallel', 1,
            '-j', 'y',
            '-b', 'n',

            './generate-lightcurve.py',
            '--action-id', actionid,
            '--reference-frame-path', reference_frame_path,
            '--action-frame-path', action_frame_path,
            '--aperture-radius', aperture_radius,
            'lightcurves-{0}-{1}{2}.dat'.format(field, actionid, suffix),
            targets_file
        ]])

        subprocess.check_call(cmd)

if __name__ == "__main__":
    parser = ap.ArgumentParser(description="Queues lightcurve generation jobs on the NGTS cluster.")
    parser.add_argument('targets_file',
                        type=str,
                        help='Data file generated by generate-target-list.py')
    parser.add_argument('--output-suffix',
                        type=str,
                        default='',
                        help='Append a suffix to the generated lightcurve files.')
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
    reduce_nights(args.targets_file, args.output_suffix, args.action_frame_path,
                  args.reference_frame_path, args.aperture_radius)

