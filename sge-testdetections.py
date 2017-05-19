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
   sge-testdetections.py

   Helper script to queue jobs running the detection script over a hardcoded list of actions
"""

# pylint: disable=invalid-name

import subprocess

field = 'NG0603-3056'
#actions = [134784, 134884, 134984, 135084, 135184, 135284, 135384]
#actions = [131247, 131347, 131447, 131547, 131647, 131747, 131847,
#           131947, 132047, 132147, 132252, 132352, 132452, 132552]
actions = [132652, 132752, 132879, 132977, 133076, 133175, 133273,
           133371, 133471, 133571, 133671, 133771, 133871, 133971,
           134071, 134171, 134271, 134371, 134471, 134584, 134684]

detection_sigma = 3
delta_minimum_threshold = 300
delta_percent_threshold = 5

for actionid in actions:
    cmd = list([str(x) for x in [
        'qsub', '-N', 'transientdetection-{0}-{1}'.format(field, actionid),
        '-S', '/usr/local/python/bin/python',
        '-o', 'logs',
        '-pe', 'parallel', 1,
        '-j', 'y',
        '-b', 'n',

        './detect-variability-integration2.py',
        '--detection-sigma', detection_sigma,
        '--delta-minimum-threshold', delta_minimum_threshold,
        '--delta-percentage-threshold', delta_percent_threshold,
        actionid,
        'reference-{0}-scalemask.fits'.format(field),
        'detections-{0}-{1}-integration2-{2}-{3}-{4}.fits'.format(field, actionid, detection_sigma,
                                                                  delta_minimum_threshold,
                                                                  delta_percent_threshold)
    ]])

    subprocess.check_call(cmd)
