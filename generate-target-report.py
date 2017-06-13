#!/usr/bin/env python3

import argparse as ap
import datetime
import glob
import json
import numpy as np
import os
import urllib.parse as urlparse
import urllib.request
import astropy.io.fits as fits
import astropy.coordinates as coord
import astropy.stats as stats
import astropy.units as u
import sqlite3
import math
import matplotlib
import matplotlib.gridspec as gridspec

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.ticker as ticker


from PyAstronomy.pyTiming import pyPDM

# pylint: disable=invalid-name
# pylint: disable=too-many-arguments
# pylint: disable=too-many-locals
# pylint: disable=too-many-instance-attributes

APASS_QUERY = """
    SELECT m.ID, m.ra_apass, m.dec_apass, m.b, m.berr, m.v, m.verr, m.g, m.gerr, m.r, m.rerr, m.i, m.ierr,
    p.jmag, p.e_jmag, p.hmag, p.e_hmag, p.kmag, p.e_kmag,
    g.fuv_mag, g.fuv_magerr, g.nuv_mag, g.nuv_magerr, w.w1, w.e_w1, w.w2, w.e_w2, w.w3, w.e_w3, w.w4, w.e_w4
    FROM main as m
    LEFT OUTER JOIN ppmxl as p ON m.ID==p.ID
    LEFT OUTER JOIN galex_nuv_unique as g ON m.ID==g.ID
    LEFT OUTER JOIN wise as w ON m.ID==w.ID
    WHERE m.ra_apass BETWEEN ? AND ?
    AND m.dec_apass BETWEEN ? AND ?
"""

APASS_FIELDS = [
    'apass_id', 'ra', 'dec',
    'b', 'b_err',
    'v', 'v_err',
    'g', 'g_err',
    'r', 'r_err',
    'i', 'i_err',
    'j', 'j_err',
    'h', 'h_err',
    'k', 'k_err',
    'fuv', 'fuv_err',
    'nuv', 'nuv_err',
    'w1', 'w1_err',
    'w2', 'w2_err',
    'w3', 'w3_err',
    'w4', 'w4_err',
]

APASS_MEASUREMENTS = ['b', 'v', 'g', 'r', 'i', 'j', 'h', 'k', 'fuv', 'nuv', 'w1', 'w2'] #, 'w3', 'w4']
APASS_LAMBDA = {
    'b': 4297.17,
    'v': 5394.29,
    'g': 4640.42,
    'r': 6122.33,
    'i': 7439.49,
    'j': 12350.00,
    'h': 16620.00,
    'k': 21590.00,
    'fuv': 1542.26,
    'nuv': 2274.37,
    'w1': 33526.00,
    'w2': 46028.00,
    'w3': 115608.00,
    'w4': 220883.00,
}


PREVIEW_HTML = """
<html>
<header>
<style>
body, html {width: 100%; height: 100%; margin: 0; padding: 0}
.header {position: absolute;top: 0; left: 0; right: 0; height: 30px;}
.target {position: absolute; top: 30px; left: 0; right: 0; bottom: 0;}
.target iframe {display: block; width: 100%; height: 100%; border: none;}
</style>

<script src="http://code.jquery.com/jquery-1.9.1.min.js"></script>
<script src="targets.js"></script>
<script>
  var current_id = 0;
  var current_list = [];

  function refresh() {
      $('#iframe').attr('src', current_list[current_id] + '.html');
      $('#id').text(current_id + 1);
      $('#total').text(current_list.length);
  }

  $(document).ready(function() {
    var type_selector = $('#type');
    var prev_button = $('#prev');
    var next_button = $('#next');

    current_list = targets['periodic'];
    refresh();

    type_selector.change(function() {
      current_id = 0;
      current_list = targets[$(this).val()]
      refresh();
    });

    prev_button.click(function() {
      current_id = (current_id > 0 ? current_id : current_list.length) - 1;
      refresh();
    });

    next_button.click(function() {
      current_id = (current_id < current_list.length - 1 ? current_id : -1) + 1
      refresh();
    });

  });

  $(document).keydown(function(e) {
    if (e.which == 37)
      select(current_group, parseInt(current_index) - 1);
    if (e.which == 39)
      select(current_group, parseInt(current_index) + 1)
  });
</script>

</header>
<body>
<div class="header">
<select id="type">
  <option value="periodic" selected="true">Probably Periodic</option>
  <option value="transient">Probably Transient</option>
  <option value="junk">Probably Junk</option>
</select>
<button type="button" id="prev">Previous</button>
<span><span id="id">X</span> / <span id="total">Y</span>
<button type="button" id="next">Next</button>

</div>
<div class="target">
  <iframe id="iframe" src="0.html" style="width: 100%; height: 100%"></iframe>
</div>
</body>
</html>
"""

def rescale_image_data(data, clip_low, clip_high, force_contrast=True):
    """ Returns a normalised array where clip_low percent of the pixels are 0 and
        clip_high percent of the pixels are 255
    """
    high = np.percentile(data, clip_high)
    low = np.percentile(data, clip_low)
    if force_contrast and high - low < 100:
        high = low + 100
    scale = 255. / (high - low)
    data = np.clip(data, low, high)
    return scale * (data - low)

def dss_thumbnail(ra, dec, size):
    """ra, dec, size all in degrees"""

    # Enumerate surveys in order of priority
    surveys = {
        'poss2ukstu_blue': 'DSS2-blue',
        'poss2ukstu_red': 'DSS2-red',
        'poss2ukstu_ir': 'DSS2-ir'
    }

    for s in surveys:
        url = 'http://archive.stsci.edu/cgi-bin/dss_search?r=' + str(ra) + '&dec=' + str(dec) \
            + '&v=' + s + '&f=dss1&s=on&e=J2000&h=' + str(size) + '&w=' + str(size)
        filename, _ = urllib.request.urlretrieve(url)

        try:
            with fits.open(filename) as hdulist:
                return hdulist[0].data, surveys[s]
        finally:
            os.remove(filename)

def target_row_html(target):
    """ Returns the HTML string defining a row for a target detection """
    html = '<tr>'
    html += '<td><img src="{0}.gif"></td>'.format(target['id'])
    html += '<td>{0}</td><td>{1}</td>'.format(round(target['x'], 2), round(target['y'], 2))
    html += '<td><img src="range_{0}.gif" width="800" height="30"'.format(target['id'])
    html += 'style="image-rendering: -moz-crisp-edges;"></td>'
    html += '</tr>'
    return html

def frame_path(detection, action_frame_path):
    return os.path.join(action_frame_path, 'action{0}_observeField'.format(detection['action']),
                        detection['filename'])

def significant_frequency(time, flux, min_freq, max_freq, plot):
    # Initial estimate based on Lomb Scargle
    frequency, power = stats.LombScargle(time, flux).autopower(minimum_frequency=min_freq,
                                                               maximum_frequency=max_freq,
                                                               samples_per_peak=10)
    best_index = np.argmax(power)

    best_freq = frequency[best_index]
    best_ampl = math.sqrt(power[best_index])
    mean_ampl = np.mean(np.sqrt(power))

    # Coarse PDM search around +/0 5% of best-fit sinusoid
    pdm = pyPDM.PyPDM(time, flux)
    pdm_scanner = pyPDM.Scanner(minVal=best_freq / 1.05, maxVal=best_freq * 1.05,
                                dVal=0.0025, mode="frequency")
    pdm_freq, pdm_theta = pdm.pdmEquiBinCover(10, 3, pdm_scanner)
    pdm_best_index = np.argmin(pdm_theta)

    # Fine PDM search around +/- 0.5% of best-fit fold period
    pdm_scanner = pyPDM.Scanner(minVal=pdm_freq[pdm_best_index] / 1.005,
                                maxVal=pdm_freq[pdm_best_index] * 1.005,
                                dVal=0.0005, mode="frequency")
    pdm_freq, pdm_theta = pdm.pdmEquiBinCover(10, 3, pdm_scanner)
    pdm_best_index = np.argmin(pdm_theta)

    # Check half frequency / double period for better fit
    pdm_scanner = pyPDM.Scanner(minVal=pdm_freq[pdm_best_index] / 2.01,
                                maxVal=pdm_freq[pdm_best_index] * 0.50025,
                                dVal=0.00025, mode="frequency")
    pdm_half_freq, pdm_half_theta = pdm.pdmEquiBinCover(10, 3, pdm_scanner)
    pdm_half_best_index = np.argmin(pdm_half_theta)

    if pdm_theta[pdm_best_index] < pdm_half_theta[pdm_half_best_index]:
        best_freq = pdm_freq[pdm_best_index]
    else:
        best_freq = pdm_half_freq[pdm_half_best_index]

    if plot is not None:
        if best_ampl >= 4 * mean_ampl:
            plot.plot([best_freq, best_freq], [0, 1], 'r-')
        plot.plot(frequency, power, 'b-')

    # Require Amplitude > 4<A> to be significant
    if best_ampl < 4 * mean_ampl:
        return None, best_ampl / mean_ampl

    return best_freq, best_ampl / mean_ampl

def generate_target_report(targets_file, output_dir, reference_frame_path):
    start_time = datetime.datetime.utcnow()

    lightcurve_glob = 'lightcurves-NG0603-3056-[0-9][0-9][0-9][0-9][0-9][0-9]-integration2-3-300-5.dat'

    with fits.open(targets_file) as metadata:
        night_targets = metadata[2].data
        nights = metadata[4].data
        targets = metadata[1].data
        target_count = len(targets)
        reference_frame = metadata[0].header['REFMASK']

    # Load data
    time = []
    flux = []

    files = glob.glob(lightcurve_glob)
    for i, f in enumerate(files):
        data = np.loadtxt(f)
        if len(data[0]) != 2 * target_count + 1:
            print(f, 'wrong target count, ignoring')
            continue
        time.extend(data[:, 0])
        flux.extend(data[:, 1:])
        print(i, len(files))

    flux = np.array(flux)
    time = np.array(time)
    time_range = [np.min(time), np.max(time)]

    with fits.open(os.path.join(reference_frame_path, reference_frame)) as reference_fits:
        reference = reference_fits[0].data

    report_dir = os.path.join(output_dir, os.path.basename(targets_file).split('.')[0])
    if not os.path.exists(report_dir):
        os.mkdir(report_dir)

    probably_periodic_targets = []
    probably_transient_targets = []
    probably_junk_targets = []
    apass_db = sqlite3.connect('apass-dr9.db')
    apass_cursor = apass_db.cursor()

    for target in targets:
        html = '<html><body>'
        print(target['id'])
        detected_nights = night_targets[night_targets['id'] == target['id']]

        target_flux = flux[:, 2*target['id']]
        target_err = flux[:, 2*target['id'] + 1]
        mean = np.mean(target_flux)
        std = np.std(target_flux)

        np.savetxt(os.path.join(report_dir, '{0}.dat'.format(target['id'])),
                   np.transpose((time, target_flux, target_err)),
                   header='BMJD         FLUX         ERR',
                   fmt=['%.8f', '%.6f', '%.6f'])

        event_actions = []

        # Generate top plot
        toprow = gridspec.GridSpec(1, 3, width_ratios=[4, 1, 1])
        toprow.update(top=0.95, bottom=0.75, wspace=0.01, left=0.3, right=0.9)

        middlerow = gridspec.GridSpec(1, 3)
        middlerow.update(top=0.575, bottom=0.125, wspace=0.1, left=0.1, right=0.9)

        top_fig = plt.figure(figsize=(12, 5))

        # Full run photometry
        ax_fullphot = plt.subplot(toprow[0])

        for night in nights:
            detected = np.any(np.logical_and(night_targets['action'] == night['action'],
                                             night_targets['id'] == target['id']))
            if detected:
                ax_fullphot.add_patch(
                    patches.Rectangle(
                        (night['mjd'] + 0.75, mean - 7 * std),   # (x,y)
                        1,
                        12 * std,
                        facecolor="#f3f3f3",
                        edgecolor="none"
                    )
                )

        reference_flux = target['reference_flux']
        ax_fullphot.plot(time_range, [reference_flux, reference_flux], 'r-')
        ax_fullphot.plot(time, target_flux, 'b.', markersize=0.5)
        ax_fullphot.set_ylim(mean - 7*std, mean + 5*std)
        ax_fullphot.set_xlim(time_range)
        ax_fullphot.set_xlabel('BMJD')
        ax_fullphot.set_ylabel('Counts')
        ax_fullphot.set_title('Full Lightcurve')

        # Thumbnail
        thumb_margin = 16
        x1 = int(target['x'] - thumb_margin)
        x2 = int(target['x'] + thumb_margin + 1)
        y1 = int(target['y'] - thumb_margin)
        y2 = int(target['y'] + thumb_margin + 1)

        ax_thumb = plt.subplot(toprow[1])
        ax_thumb.set_aspect(1)
        thumb_clipped = rescale_image_data(reference[y1:y2, x1:x2], 5, 95)
        ax_thumb.imshow(thumb_clipped, cmap='gray', interpolation='nearest')
        # TODO: Get aperture size from header
        ax_thumb.add_artist(plt.Circle((target['x'] - x1, target['y'] - y1),
                                       2.5, color='r', fill=False))
        ax_thumb.xaxis.set_visible(False)
        ax_thumb.yaxis.set_visible(False)
        ax_thumb.set_title('Reference')

        # DSS thumbnail
        # Width in arcmin
        dss_width = (2*thumb_margin) * 5. / 60
        dss_thumb, dss_title = dss_thumbnail(target['ra'], target['dec'], dss_width)
        ax_dss = plt.subplot(toprow[2])
        ax_dss.set_aspect(1)
        if dss_thumb is not None:
            dss_clipped = rescale_image_data(dss_thumb, 2.5, 97.5)
            ax_dss.imshow(dss_clipped, origin='lower', cmap='gray', interpolation='nearest')
            ax_dss.set_title(dss_title)
        ax_dss.xaxis.set_visible(False)
        ax_dss.yaxis.set_visible(False)

        # Periodogram
        ax_periodogram = plt.subplot(middlerow[0])
        frequency, significance = significant_frequency(time, target_flux, 1./60, 24,
                                                        ax_periodogram)
        ax_periodogram.set_title('Periodogram')
        ax_periodogram.set_xlabel('Frequency (c/d)')
        ax_periodogram.set_ylabel('L-S Power')

        # Folded Lightcurve
        ax_folded = plt.subplot(middlerow[1])
        if frequency is not None:
            phase = frequency * time % 1
            ax_folded.plot(phase, target_flux, 'b.', markersize=0.5)
            ax_folded.plot(phase + 1, target_flux, 'b.', markersize=0.5)

        ax_folded.yaxis.set_visible(False)
        ax_folded.set_ylim(mean - 7*std, mean + 5*std)
        ax_folded.set_xlim(0, 2)
        ax_folded.set_title('Folded Lightcurve')
        ax_folded.set_ylabel('Flux (Counts)')
        ax_folded.set_xlabel('Phase')

        # SED
        apass_query_box = (target['ra'] - 5./3600, target['ra'] + 5./3600,
                           target['dec'] - 5./3600, target['dec'] + 5./3600)

        ax_sed = plt.subplot(middlerow[2])
        for i, row in enumerate(apass_cursor.execute(APASS_QUERY, apass_query_box)):
            row_data = {}
            for j, name in enumerate(APASS_FIELDS):
                row_data[name] = row[j]

            x = []
            y = []
            y_err = []
            for m in APASS_MEASUREMENTS:
                if row_data[m] and row_data[m] < 50:
                    x.append(APASS_LAMBDA[m] * 1e-4)
                    y.append(row_data[m])
                    y_err.append(row_data[m + '_err'] or 0)

            if len(x) > 0:
                ax_sed.errorbar(x, y, yerr=y_err, fmt='b.')

        ax_sed.invert_yaxis()
        ax_sed.set_xscale("log", nonposx='clip')
        ax_sed.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: str(round(x, 1))))
        ax_sed.yaxis.tick_right()
        ax_sed.yaxis.set_label_position('right')
        ax_sed.set_title('SED')
        ax_sed.set_ylabel('Magnitude')
        ax_sed.set_xlabel(r'Wavelength ($\mu$m)')
        ax_sed.set_xlim(0.1, 10)

        plt.savefig(os.path.join(report_dir, '{}_combined.png'.format(target['id'])), format='png')
        plt.close(top_fig)

        target_err = flux[:, 2*target['id'] + 1]
        mean = np.mean(target_flux)
        std = np.std(target_flux)

        if len(detected_nights) < 7:
            for a in night_targets[night_targets['id'] == target['id']]['action']:
                event_actions.append(a)
                night = nights[nights['action'] == a][0]
                data_filt = np.abs(time - night['mjd'] - 1) < 5

                event_time = time[data_filt] - time_range[0]
                event_flux = target_flux[data_filt]
                plot_ylim = [np.min(event_flux), np.max(event_flux) + 0.1 * (np.max(event_flux) - np.min(event_flux))]

                fig = plt.figure(figsize=(12, 1))
                grid = gridspec.GridSpec(1, 1)
                grid.update(top=0.95, bottom=0.25, left=0.1, right=0.9)
                event = plt.subplot(grid[0])
                event.plot(event_time, event_flux, 'b.', markersize=0.5)
                event.plot([night['mjd']+0.75 - time_range[0], night['mjd']+0.75 - time_range[0]], plot_ylim)
                event.plot([night['mjd']+1.75 - time_range[0], night['mjd']+1.75 - time_range[0]], plot_ylim)
                event.set_xlim([night['mjd'] - 4 - time_range[0], night['mjd'] + 7 - time_range[0]])
                event.set_ylim(plot_ylim)
                event.set_ylabel('Counts')
                plt.savefig(os.path.join(report_dir, '{}_{}.png'.format(target['id'], a)), format='png')
                plt.close(fig)

        target_ra_string = coord.Angle(target['ra'] * u.deg).to_string(unit=u.hourangle, sep=':')
        target_dec_string = coord.Angle(target['dec'] * u.deg).to_string(sep=':', alwayssign=True)

        html += '<div style="position: relative; width: 100%">'
        html += '<img src="{0}_combined.png">'.format(target['id'])
        html += '<table style="position: absolute; top: 25px; left: 35px;">'
        html += '<tr><td>RA:</td><td>' + target_ra_string
        html += '<tr><td>Dec:</td><td>' + target_dec_string + '</td></tr>'

        simbad_args = {
            'Coord': target_ra_string + target_dec_string,
            'Radius': 10,
            'Radius.unit': 'arcsec'
        }

        if frequency is not None:
            display_period = 1./frequency
            display_units = 'd'
            if display_period < 1.1:
                display_period *= 24
                display_units = 'h'
            html += '<tr><td>Period:</td><td>{}{} ({})</td></tr>'.format(round(display_period, 2), display_units, round(significance, 2))

        html += '<tr><td width="75"><a href="http://simbad.u-strasbg.fr/simbad/sim-coo?{}" target="_blank">Simbad</a></td>'.format(urlparse.urlencode(simbad_args))
        html += '<td><a href="{}.dat" target="_blank">Light Curve</a></td></tr>'.format(target['id'])
        html += '</table>'
        html += '</div>'

        for a in event_actions:
            html += '<img src="{0}_{1}.png">'.format(target['id'], a)
            html += '<br />'
        html += '</body></html>'

        with open(os.path.join(report_dir, '{}.html'.format(target['id'])), 'w') as report:
            report.write(html)



        if len(detected_nights) < 7:
            probably_transient_targets.append(int(target['id']))
        elif frequency is not None and min([abs(x/frequency - 0.99727) for x in [0.5, 1., 2.]]) < 0.01:
            probably_junk_targets.append(int(target['id']))
        else:
            probably_periodic_targets.append(int(target['id']))

    with open(os.path.join(report_dir, 'targets.js'), 'w') as outfile:
        json_data = {
            'junk': probably_junk_targets,
            'periodic': probably_periodic_targets,
            'transient': probably_transient_targets
        }

        outfile.write('targets = ')
        json.dump(json_data, outfile)
        outfile.write(';')

    with open(os.path.join(report_dir, 'index.html'), 'w') as outfile:
        outfile.write(PREVIEW_HTML)

    print('Completed in {}'.format(datetime.datetime.utcnow() - start_time))

if __name__ == "__main__":
    parser = ap.ArgumentParser(description="Detect variable objects in a NGTS observation action.")
    parser.add_argument('targets_file',
                        type=str,
                        help='Data file generated by generate-target-list.py')
    parser.add_argument('--output-dir',
                        type=str,
                        default='.',
                        help='Directory to write the output report package to')
    parser.add_argument('--reference-frame-path',
                        type=str,
                        default='.',
                        help='Path to the directory where reference frames are stored')

    args = parser.parse_args()
    generate_target_report(args.targets_file, args.output_dir, args.reference_frame_path)

