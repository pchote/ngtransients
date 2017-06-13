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
#actions = [132652, 132752, 132879, 132977, 133076, 133175, 133273,
#           133371, 133471, 133571, 133671, 133771, 133871, 133971,
#           134071, 134171, 134271, 134371, 134471, 134584, 134684]

#actions = [112897, 127295, 127384, 127515, 127620, 127743, 127858,
#           127973, 128091, 128207, 128322, 128898, 129014, 129130,
#           129246, 129491, 132795, 129607, 129725, 129970, 130081,
#           130191, 130292, 130392, 130490, 130588, 130688, 130810,
#           130910, 131010]

#actions = [135484, 135584, 135684, 135784, 135884, 135984, 136084,
#           136184, 136284, 136317, 136395, 136495, 136595, 136695,
#           136795, 136908, 137008, 137122, 137235, 137353, 137453,
#           137555, 137655, 137755, 137855, 137955, 138055, 138155,
#           138255, 138355, 138455, 138555, 138655, 138755, 138855,
#           138968, 139068, 139168, 139268, 139368, 139468, 139568,
#           139668, 139765, 139869, 139960, 140059, 140155, 140251,
#           140471, 140697, 140899, 141007, 141106, 141238, 141333]

actions = [141478, 141656, 141775, 141894, 142013, 142145, 142264,
           142383, 142502, 142622, 142741, 142860, 142979, 143101,
           143220, 143352, 143473, 143592, 143851, 143989, 144111,
           144239, 144387, 144530, 144701, 144784, 144905, 145024,
           145126, 145288, 145372, 145502, 145623, 145744, 145865,
           145986, 146107, 146228, 146397, 146518, 146639, 146760,
           146881, 147236, 147357, 147492, 147613, 147735, 147982,
           148103, 148233, 148353, 148474, 148731, 148857, 148986,
           149116]

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

        './detect-variability.py',
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
