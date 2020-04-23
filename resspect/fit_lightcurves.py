# Copyright 2020 resspect software
# Author: Emille E. O. Ishida
#
# created on 14 April 2020
#
# Licensed GNU General Public License v3.0;
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.gnu.org/licenses/gpl-3.0.en.html
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from resspect.bazin import bazin, fit_scipy
from resspect.snana_fits_to_pd import read_fits

import io
import matplotlib.pylab as plt
import numpy as np
import os
import pandas as pd
import tarfile

__all__ = ['LightCurve', 'fit_snpcc_bazin', 'fit_resspect_bazin',
           'fit_plasticc_bazin']


class LightCurve(object):
    """ Light Curve object, holding meta and photometric data.

    Attributes
    ----------
    bazin_features_names: list
        List of names of the Bazin function parameters.
    bazin_features: list
        List with the 5 best-fit Bazin parameters in all filters.
        Concatenated from blue to red.
    dataset_name: str
        Name of the survey or data set being analyzed.
    filters: list
        List of broad band filters.
    id: int
        SN identification number
    id_name:
        Column name of object identifier.
    photometry: pd.DataFrame
        Photometry information. Keys --> [mjd, band, flux, fluxerr, SNR, MAG, MAGERR].
    redshift: float
        Redshift
    sample: str
        Original sample to which this light curve is assigned
    sim_peakmag: np.array
        Simulated peak magnitude in each filter
    sncode: int
        Number identifying the SN model used in the simulation
    sntype: str
        General classification, possibilities are: Ia, II or Ibc

    Methods
    -------
    check_queryable(mjd: float, r_lim: float)
        Check if this light can be queried in a given day.
    evaluate_bazin(param: list, time: np.array) -> np.array
        Evaluate the Bazin function given parameter values.
    load_snpcc_lc(path_to_data: str)
        Reads header and photometric information for 1 light curve
    load_plasticc_lc(photo_file: str, snid: int)
	Load photometric information for 1 PLAsTiCC light curve
    load_resspect_lc(photo_file: str, snid: int)
	Load photometric information for 1 RESSPECT light curve
    fit_bazin(band: str) -> list
        Calculates best-fit parameters from the Bazin function in 1 filter
    fit_bazin_all()
        Calculates  best-fit parameters from the Bazin func for all filters
    plot_bazin_fit(save: bool, show: bool, output_file: srt)
        Plot photometric points and Bazin fitted curve

    Examples
    --------

    ##### for RESSPECT and PLAsTiCC light curves it is necessary to
    ##### input the object identification for dealing with 1 light curve

    >>> import io
    >>> import pandas as pd
    >>> import tarfile

    >>> from resspect import LightCurve

    # path to header file
    >>> path_to_header = '~/RESSPECT_PERFECT_V2_TRAIN_HEADER.tar.gz'

    # openning '.tar.gz' files requires some juggling ...
    >>> tar = tarfile.open(path_to_header, 'r:gz')
    >>> fname = tar.getmembers()[0]
    >>> content = tar.extractfile(fname).read()
    >>> header = pd.read_csv(io.BytesIO(content))
    >>> tar.close()

    # choose one object
    >>> snid = header['objid'].values[4]

    # once you have the identification you can use this class
    >>> path_to_lightcurves = '~/RESSPECT_PERFECT_V2_TRAIN_LIGHTCURVES.tar.gz'

    >>> lc = LightCurve()                        # create light curve instance
    >>> lc.load_snpcc_lc(path_to_lightcurves)    # read data
    >>> lc.photometry
             mjd band       flux   fluxerr         SNR
    0    53214.0    u   0.165249  0.142422    1.160276
    1    53214.0    g  -0.041531  0.141841   -0.292803
    ..       ...  ...        ...       ...         ...
    472  53370.0    z  68.645930  0.297934  230.406460
    473  53370.0    Y  63.254270  0.288744  219.067050

    >>> lc.fit_bazin_all()               # perform Bazin fit in all filters
    >>> lc.bazin_features                # display Bazin parameters
    [198.63302952843623, -9.38297128588733, 43.99971014717201,
    ... ...
    -1.546372806815066]

    for fitting the entire sample ...

    >>> output_file = 'RESSPECT_PERFECT_TRAIN.DAT'
    >>> fit_resspect_bazin(path_to_lightcurves, path_to_header,
                           output_file, sample='train')
    """

    def __init__(self):
        self.bazin_features = []
        self.bazin_features_names = ['a', 'b', 't0', 'tfall', 'trsise']
        self.dataset_name = ' '
        self.filters = []
        self.id = 0
        self.id_name = None
        self.photometry = pd.DataFrame()
        self.redshift = 0
        self.sample = ' '
        self.sim_peakmag = []
        self.sncode = 0
        self.sntype = ' '

    def load_snpcc_lc(self, path_to_data: str):
        """Reads one LC from SNPCC data.

        Populates the attributes: dataset_name, id, sample, redshift, sncode,
        sntype, photometry and sim_peakmag.

        Parameters
        ---------
        path_to_data: str
            Path to text file with data from a single SN.
        """

        # set the designation of the data set
        self.dataset_name = 'SNPCC'

        # set filters
        self.filters = ['g', 'r', 'i', 'z']

        # set SN types
        snii = ['2', '3', '4', '12', '15', '17', '19', '20', '21', '24', '25',
                '26', '27', '30', '31', '32', '33', '34', '35', '36', '37',
                '38', '39', '40', '41', '42', '43', '44']

        snibc = ['1', '5', '6', '7', '8', '9', '10', '11', '13', '14', '16',
                 '18', '22', '23', '29', '45', '28']

        # read light curve data
        op = open(path_to_data, 'r')
        lin = op.readlines()
        op.close()

        # separate elements
        data_all = np.array([elem.split() for elem in lin])

        # flag useful lines
        flag_lines = np.array([True if len(line) > 1 else False for line in data_all])

        # get only informative lines
        data = data_all[flag_lines]

        photometry_raw = []               # store photometry
        header = []                      # store parameter header

        # get header information
        for line in data:
            if line[0] == 'SNID:':
                self.id = int(line[1])
                self.id_name = 'SNID'
            elif line[0] == 'SNTYPE:':
                if line[1] == '-9':
                    self.sample = 'test'
                else:
                    self.sample = 'train'
            elif line[0] == 'SIM_REDSHIFT:':
                self.redshift = float(line[1])
            elif line[0] == 'SIM_NON1a:':
                self.sncode = line[1]
                if line[1] in snibc:
                    self.sntype = 'Ibc'
                elif line[1] in snii:
                    self.sntype = 'II'
                elif line[1] == '0':
                    self.sntype = 'Ia'
                else:
                    raise ValueError('Unknown supernova type!')
            elif line[0] == 'VARLIST:':
                header: list = line[1:]
            elif line[0] == 'OBS:':
                photometry_raw.append(np.array(line[1:]))
            elif line[0] == 'SIM_PEAKMAG:':
                self.sim_peakmag = np.array([float(item) for item in line[1:5]])

        # transform photometry into array
        photometry_raw = np.array(photometry_raw)

        # put photometry into data frame
        self.photometry['mjd'] = np.array([float(item) for item in photometry_raw[:, header.index('MJD')]])
        self.photometry['band'] = np.array(photometry_raw[:, header.index('FLT')])
        self.photometry['flux'] = np.array([float(item) for item in photometry_raw[:, header.index('FLUXCAL')]])
        self.photometry['fluxerr'] = np.array([float(item) for item in photometry_raw[:, header.index('FLUXCALERR')]])
        self.photometry['SNR'] = np.array([float(item) for item in photometry_raw[:, header.index('SNR')]])
        self.photometry['MAG'] = np.array([float(item) for item in photometry_raw[:, header.index('MAG')]])
        self.photometry['MAGERR'] = np.array([float(item) for item in photometry_raw[:, header.index('MAGERR')]])

    def load_resspect_lc(self, photo_file, snid):
        """
        Return 1 light curve from RESSPECT simulations.
    
        Parameters
        ----------
        photo_file: str
            Complete path to light curves file.
        snid: int
            Identification number for the desired light curve.
        """

        if '.tar.gz' in photo_file:
            tar = tarfile.open(photo_file, 'r:gz')
            fname = tar.getmembers()[0]
            content = tar.extractfile(fname).read()
            all_photo = pd.read_csv(io.BytesIO(content))
            tar.close()
        elif '.FITS' in photo_file:
            df_header, all_photo = read_fits(photo_file, drop_separators=True)
        else:    
            all_photo = pd.read_csv(photo_file, index_col=False)

            if ' ' in all_photo.keys()[0]:
                all_photo = pd.read_csv(photo_file, sep=' ', index_col=False)

        if 'SNID' in all_photo.keys():
            flag = all_photo['SNID'] == snid
            self.id_name = 'SNID'
        elif 'snid' in all_photo.keys():
            flag = all_photo['snid'] == snid
            self.id_name = 'snid'
        elif 'objid' in all_photo.keys():
            flag = all_photo['objid'] == snid
            self.id_name = 'objid'
        elif 'id' in all_photo.keys():
            flag = all_photo['id'] == snid
            self.id_name = 'id'

        photo = all_photo[flag]

        self.dataset_name = 'RESSPECT'                      # name of data set
        self.filters = ['u', 'g', 'r', 'i', 'z', 'Y']       # list of filters  
        
        # check filter name
        if 'b' in str(photo['FLT'].values[0]):
            band = []
            for i in range(photo.shape[0]):
                for f in self.filters:
                    if "b'" + f + " '" == str(photo['FLT'].values[i]) or \
                    "b'" + f + "'" == str(photo['FLT'].values[i]) or \
                    "b'" + f + "' " == str(photo['FLT'].values[i]):
                        band.append(f)
            photo.insert(1, 'band', band, True)

        else:
            photo.insert(1, 'band', photo['FLT'].values, True)
                        
        self.id = snid 
        self.photometry = {}
        self.photometry['mjd'] = photo['MJD'].values
        self.photometry['band'] = photo['band'].values
        self.photometry['flux'] = photo['FLUXCAL'].values
        self.photometry['fluxerr'] = photo['FLUXCALERR'].values

        if 'SNR' in photo.keys():
            self.photometry['SNR'] = photo['SNR'].values
        else:
            signal = self.photometry['flux']
            noise = self.photometry['fluxerr']
            self.photometry['SNR'] = \
                np.array([signal[i]/noise[i] for i in range(signal.shape[0])])
            
        self.photometry = pd.DataFrame(self.photometry)
        
    def load_plasticc_lc(self, photo_file: str, snid: int):
        """
        Return 1 light curve from PLAsTiCC simulations.
    
        Parameters
        ----------
        photo_file: str
            Complete path to light curve file.
        snid: int
            Identification number for the desired light curve.
        """

        if '.tar.gz' in photo_file:
            tar = tarfile.open(photo_file, 'r:gz')
            fname = tar.getmembers()[0]
            content = tar.extractfile(fname).read()
            all_photo = pd.read_csv(io.BytesIO(contente))
        else:
            all_photo = pd.read_csv(photo_file, index_col=False)

            if ' ' in all_photo.keys()[0]:
                all_photo = pd.read_csv(photo_file, sep=' ', index_col=False)

        if 'object_id' in all_photo.keys():
            flag = all_photo['object_id'] == snid
            self.id_name = 'object_id'
        elif 'SNID' in all_photo.keys():
            flag = all_photo['SNID'] == snid
            self.id_name = 'SNID'
        elif 'snid' in all_photo.keys():
            flag = all_photo['snid'] == snid
            self.id_name = 'snid'

        photo = all_photo[flag]

        filter_dict = {0:'u', 1:'g', 2:'r', 3:'i', 4:'z', 5:'Y'}
           
        self.dataset_name = 'PLAsTiCC'              # name of data set
        self.filters = ['u', 'g', 'r', 'i', 'z', 'Y']       # list of filters
        self.id = snid
        self.photometry = {}
        self.photometry['mjd'] = photo['mjd'].values
        self.photometry['band'] = [filter_dict[photo['passband'].values[k]] 
                                   for k in range(photo['passband'].shape[0])]
        self.photometry['flux'] = photo['flux'].values
        self.photometry['fluxerr'] = photo['flux_err'].values
        self.photometry['detected_bool'] = photo['detected_bool'].values
        self.photometry = pd.DataFrame(self.photometry)

    def check_queryable(self, mjd: float, r_lim: float):
        """Check if this light can be queried in a given day.

        This checks only r-band mag limit in a given epoch.
        It there is no observation on that day, use the last available
        observation.

        Parameters
        ----------
        mjd: float
            MJD where the query will take place.
        r_lim: float
            r-band magnitude limit below which query is possible.

        Returns
        -------
        bool
            If true, sample is changed to `queryable`.
        """

        # create photo flag
        photo_flag = self.photometry['mjd'].values <= mjd
        rband_flag = self.photometry['band'].values == 'r'
        surv_flag = np.logical_and(photo_flag, rband_flag)

        if 'MAG' in self.photometry.keys():
            # check surviving photometry
            surv_mag = self.photometry['MAG'].values[surv_flag]

        else:
            surv_flux = self.photometry['flux'].values[surv_flag]
            if surv_flux[-1] > 0:
                surv_mag = [2.5 * (11 - np.log10(surv_flux[-1]))]
            else:
                surv_mag = []

        if len(surv_mag) > 0 and 0 < surv_mag[-1] <= r_lim:
            return True
        else:
            return False

    def fit_spline(self, band:str, method:str = 'linear'):
        """Fit light curve to a spline for one filter.

        Parameters
        ----------
        band: str
            Choice of broad band filter

        Returns
        -------
        spline_fit: scipy.interpolate.UnivariateSpline
            Interpolated light curve
        """

        # build filter flag
        filter_flag = self.photometry['band'] == band

        # get info for this filter
        time = self.photometry['mjd'].values[filter_flag]
        flux = self.photometry['flux'].values[filter_flag]

        # fit spline
        #spline_fit = interp1d(time - time[0], flux, kind=method)
        spline_fit = UnivariateSpline(time, flux, kind=method, bounds_error=False, fill_value = np.median(flux))

        return spline_fit
    
    def fit_bazin(self, band: str):
        """Extract Bazin features for one filter.

        Parameters
        ----------
        band: str
            Choice of broad band filter

        Returns
        -------
        bazin_param: list
            Best fit parameters for the Bazin function: [a, b, t0, tfall, trise]
        """

        # build filter flag
        filter_flag = self.photometry['band'] == band

        # get info for this filter
        time = self.photometry['mjd'].values[filter_flag]
        flux = self.photometry['flux'].values[filter_flag]

        # fit Bazin function
        bazin_param = fit_scipy(time - time[0], flux)

        return bazin_param

    def evaluate_bazin(self, param: list, time: np.array):
        """Evaluate the Bazin function given parameter values.

        Parameters
        ----------
        param: list
            List of Bazin parameters in order [a, b, t0, tfall, trise] 
            for all filters, concatenated from blue to red
        time: np.array or list
            Time since maximum where to evaluate the Bazin fit.

        Returns
        -------
        np.array
            Value of the Bazin function in each required time
        """
        # store flux values and starting points
        flux = []
        first_obs = []
        tmax_all = []

        for k in range(len(self.filters)):
            # find day of maximum
            x = range(400)
            y = [bazin(epoch, param[0 + k * 5], 
                      param[1 + k * 5], param[2 + k * 5], param[3 + k * 5], param[4 + k * 5])
                      for epoch in x]

            t_max = x[y.index(max(y))]
            tmax_all.append(t_max)
            
            for item in time:
                epoch = t_max + item
                flux.append(bazin(epoch, param[0 + k * 5], 
                      param[1 + k * 5], param[2 + k * 5], param[3 + k * 5], param[4 + k * 5]))

            first_obs.append(t_max + time[0])

        return np.array(flux), first_obs, tmax_all
        

    def fit_bazin_all(self):
        """Perform Bazin fit for all filters independently and concatenate results.

        Populates the attributes: bazin_features.
        """
        # remove previous fit attempts
        self.bazin_features = []

        for band in self.filters:
            # build filter flag
            filter_flag = self.photometry['band'] == band

            if sum(filter_flag) > 4:
                best_fit = self.fit_bazin(band)

                if sum([str(item) == 'nan' for item in best_fit]) == 0:
                    for fit in best_fit:
                        self.bazin_features.append(fit)
                else:
                    for i in range(5):
                        self.bazin_features.append('None')
            else:
                for i in range(5):
                    self.bazin_features.append('None')

    def plot_bazin_fit(self, save=True, show=False, output_file=' ', figscale=1):
        """
        Plot data and Bazin fitted function.

        Parameters
        ----------
        save: bool (optional)
             Save figure to file. Default is True.
        show: bool (optinal)
             Display plot in windown. Default is False.
        output_file: str (optional)
            Name of file to store the plot.
        figscale: float (optional)
            Allow to control the size of the figure.
        """

        # number of columns in the plot
        ncols = len(self.filters) / 2 + len(self.filters) % 2
        fsize = (figscale * 5 * ncols , figscale * 10)
        
        plt.figure(figsize=fsize)

        for i in range(len(self.filters)):
            plt.subplot(2, ncols, i + 1)
            plt.title('Filter: ' + self.filters[i])

            # filter flag
            filter_flag = self.photometry['band'] == self.filters[i]
            x = self.photometry['mjd'][filter_flag].values
            y = self.photometry['flux'][filter_flag].values
            yerr = self.photometry['fluxerr'][filter_flag].values

            if len(x) > 4:
                # shift to avoid large numbers in x-axis
                time = x - min(x)
                xaxis = np.linspace(0, max(time), 500)[:, np.newaxis]
                # calculate fitted function
                fitted_flux = np.array([bazin(t, self.bazin_features[i * 5],
                                              self.bazin_features[i * 5 + 1],
                                              self.bazin_features[i * 5 + 2],
                                              self.bazin_features[i * 5 + 3],
                                              self.bazin_features[i * 5 + 4])
                                        for t in xaxis])

                plt.errorbar(time, y, yerr=yerr, color='blue', fmt='o')
                plt.plot(xaxis, fitted_flux, color='red', lw=1.5)
                plt.xlabel('MJD - ' + str(min(x)))

            else:
                plt.xlabel('MJD')
            plt.ylabel('FLUXCAL')
            plt.tight_layout()

        if save:
            plt.savefig(output_file)
            plt.show('all')
        if show:
            plt.show()
            
    def extract_raw_features(self):
        """Adapted from avocado"""
        """Extract raw features from an object
        Featurizing is slow, so the idea here is to extract a lot of different
        things, and then in `select_features` these features are postprocessed
        to select the ones that are actually fed into the classifier. This
        allows for rapid iteration of training on different feature sets. Note
        that the features produced by this method are often unsuitable for
        classification, and may include data leaks. Make sure that you
        understand what features can be used for real classification before
        making any changes.
        This class implements a featurizer that is tuned to the PLAsTiCC
        dataset.
        Parameters
        ----------
        astronomical_object : :class:`AstronomicalObject`
            The astronomical object to featurize.
        return_model : bool
            If true, the light curve model is also returned. Defaults to False.
        Returns
        -------
        raw_features : dict
            The raw extracted features for this object.
        model : dict (optional)
            A dictionary with the light curve model in each band. This is only
            returned if return_model is set to True.
        """
        from scipy.signal import find_peaks

        features = dict()

        ## Fit the GP and produce an output model
        gp_start_time = int(min(self.photometry['mjd']))
        gp_end_time = int(max(self.photometry['mjd']))
        gp_times = np.arange(gp_start_time, gp_end_time)
        #gp, gp_observations, gp_fit_parameters = (
        #    astronomical_object.fit_gaussian_process()
        #)
        #gp_fluxes = astronomical_object.predict_gaussian_process(
        #    plasticc_bands, gp_times, uncertainties=False, fitted_gp=gp
        #)
        
        gp_fluxes = np.array([self.fit_spline(band)(gp_times) for band in self.filters])

        times = self.photometry["mjd"]
        fluxes = self.photometry["flux"]
        flux_errors = self.photometry["fluxerr"]
        bands = self.photometry["band"]
        s2ns = fluxes / flux_errors
        
        #metadata = self.metadata

        ## Features from the metadata
        #features["host_specz"] = metadata["host_specz"]
        #features["host_photoz"] = metadata["host_photoz"]
        #features["host_photoz_error"] = metadata["host_photoz_error"]
        #features["ra"] = metadata["ra"]
        #features["decl"] = metadata["decl"]
        #features["mwebv"] = metadata["mwebv"]
        #features["ddf"] = metadata["ddf"]

        # Count how many observations there are
        features["count"] = len(fluxes)

        ## Features from GP fit parameters
        #for i, fit_parameter in enumerate(gp_fit_parameters):
        #    features["gp_fit_%d" % i] = fit_parameter

        # Maximum fluxes and times.
        max_times = gp_start_time + np.argmax(gp_fluxes, axis=1)
        med_max_time = np.median(max_times)
        max_dts = max_times - med_max_time
        max_fluxes = np.array(
            [
                gp_fluxes[band_idx, time - gp_start_time]
                for band_idx, time in enumerate(max_times)
            ]
        )
        features["max_time"] = med_max_time
        for band, max_flux, max_dt in zip(self.filters, max_fluxes, max_dts):
            features["max_flux_%s" % band] = max_flux
            features["max_dt_%s" % band] = max_dt

        # Minimum fluxes.
        min_fluxes = np.min(gp_fluxes, axis=1)
        for band, min_flux in zip(self.filters, min_fluxes):
            features["min_flux_%s" % band] = min_flux

        # Calculate the positive and negative integrals of the lightcurve,
        # normalized to the respective peak fluxes. This gives a measure of the
        # "width" of the lightcurve, even for non-bursty objects.
        positive_widths = np.sum(np.clip(gp_fluxes, 0, None), axis=1) / max_fluxes
        negative_widths = np.sum(np.clip(gp_fluxes, None, 0), axis=1) / min_fluxes
        for band_idx, band_name in enumerate(self.filters):
            features["positive_width_%s" % band_name] = positive_widths[band_idx]
            features["negative_width_%s" % band_name] = negative_widths[band_idx]

        # Calculate the total absolute differences of the lightcurve. For
        # supernovae, they typically go up and down a single time. Periodic
        # objects will have many more ups and downs.
        abs_diffs = np.sum(np.abs(gp_fluxes[:, 1:] - gp_fluxes[:, :-1]), axis=1)
        for band_idx, band_name in enumerate(self.filters):
            features["abs_diff_%s" % band_name] = abs_diffs[band_idx]

        # Find times to fractions of the peak amplitude
        fractions = [0.8, 0.5, 0.2]
        for band_idx, band_name in enumerate(self.filters):
            forward_times = find_time_to_fractions(gp_fluxes[band_idx], fractions)
            backward_times = find_time_to_fractions(
                gp_fluxes[band_idx], fractions, forward=False
            )
            for fraction, forward_time, backward_time in zip(
                fractions, forward_times, backward_times
            ):
                features["time_fwd_max_%.1f_%s" % (fraction, band_name)] = forward_time
                features["time_bwd_max_%.1f_%s" % (fraction, band_name)] = backward_time

        # Count the number of data points with significant positive/negative
        # fluxes
        thresholds = [-20, -10, -5, -3, 3, 5, 10, 20]
        for threshold in thresholds:
            if threshold < 0:
                count = np.sum(s2ns < threshold)
            else:
                count = np.sum(s2ns > threshold)
            features["count_s2n_%d" % threshold] = count

        # Count the fraction of data points that are "background", i.e. less
        # than a 3 sigma detection of something.
        features["frac_background"] = np.sum(np.abs(s2ns) < 3) / len(s2ns)

        for band_idx, band_name in enumerate(self.filters):
            mask = bands == band_name
            band_fluxes = fluxes[mask]
            band_flux_errors = flux_errors[mask]

            # Sum up the total signal-to-noise in each band
            total_band_s2n = np.sqrt(np.sum((band_fluxes / band_flux_errors) ** 2))
            features["total_s2n_%s" % band_name] = total_band_s2n

            # Calculate percentiles of the data in each band.
            for percentile in (10, 30, 50, 70, 90):
                try:
                    val = np.percentile(band_fluxes, percentile)
                except IndexError:
                    val = np.nan
                features["percentile_%s_%d" % (band_name, percentile)] = val

        # Count the time delay between the first and last significant fluxes
        thresholds = [5, 10, 20]
        for threshold in thresholds:
            significant_times = times[np.abs(s2ns) > threshold]
            if len(significant_times) < 2:
                dt = -1
            else:
                dt = np.max(significant_times) - np.min(significant_times)
            features["time_width_s2n_%d" % threshold] = dt

        # Count how many data points are within a certain number of days of
        # maximum light. This provides some estimate of the robustness of the
        # determination of maximum light and rise/fall times.
        time_bins = [
            (-5, 5, "center"),
            (-20, -5, "rise_20"),
            (-50, -20, "rise_50"),
            (-100, -50, "rise_100"),
            (-200, -100, "rise_200"),
            (-300, -200, "rise_300"),
            (-400, -300, "rise_400"),
            (-500, -400, "rise_500"),
            (-600, -500, "rise_600"),
            (-700, -600, "rise_700"),
            (-800, -700, "rise_800"),
            (5, 20, "fall_20"),
            (20, 50, "fall_50"),
            (50, 100, "fall_100"),
            (100, 200, "fall_200"),
            (200, 300, "fall_300"),
            (300, 400, "fall_400"),
            (400, 500, "fall_500"),
            (500, 600, "fall_600"),
            (600, 700, "fall_700"),
            (700, 800, "fall_800"),
        ]
        diff_times = times - med_max_time
        for start, end, label in time_bins:
            mask = (diff_times > start) & (diff_times < end)

            # Count how many observations there are in the time bin
            count = np.sum(mask)
            features["count_max_%s" % label] = count

            if count == 0:
                bin_mean_fluxes = np.nan
                bin_std_fluxes = np.nan
            else:
                # Measure the GP flux level relative to the peak flux. We do
                # this by taking the median flux in each band and comparing it
                # to the peak flux.
                bin_start = np.clip(
                    int(med_max_time + start - gp_start_time), 0, len(gp_times)
                )
                bin_end = np.clip(
                    int(med_max_time + end - gp_start_time), 0, len(gp_times)
                )

                if bin_start == bin_end:
                    scale_gp_fluxes = np.nan
                    bin_mean_fluxes = np.nan
                    bin_std_fluxes = np.nan
                else:
                    scale_gp_fluxes = (
                        gp_fluxes[:, bin_start:bin_end] / max_fluxes[:, None]
                    )
                    bin_mean_fluxes = np.mean(scale_gp_fluxes)
                    bin_std_fluxes = np.std(scale_gp_fluxes)

            features["mean_max_%s" % label] = bin_mean_fluxes
            features["std_max_%s" % label] = bin_std_fluxes

        # Do peak detection on the GP output
        for positive in (True, False):
            for band_idx, band_name in enumerate(self.filters):
                if positive:
                    band_flux = gp_fluxes[band_idx]
                    base_name = "peaks_pos_%s" % band_name
                else:
                    band_flux = -gp_fluxes[band_idx]
                    base_name = "peaks_neg_%s" % band_name
                peaks, properties = find_peaks(
                    band_flux, height=np.max(np.abs(band_flux) / 5.0)
                )
                num_peaks = len(peaks)

                features["%s_count" % base_name] = num_peaks

                sort_heights = np.sort(properties["peak_heights"])[::-1]
                # Measure the fractional height of the other peaks.
                for i in range(1, 3):
                    if num_peaks > i:
                        rel_height = sort_heights[i] / sort_heights[0]
                    else:
                        rel_height = np.nan
                    features["%s_frac_%d" % (base_name, (i + 1))] = rel_height


        return features

    def select_features(self, raw_features):
        """Adapted from avocado"""
        """Select features to use for classification
        This method should take a DataFrame or dictionary of raw features,
        produced by `featurize`, and output a list of processed features that
        can be fed to a classifier.
        Parameters
        ----------
        raw_features : pandas.DataFrame or dict
            The raw features extracted using `featurize`.
        Returns
        -------
        features : pandas.DataFrame or dict
            The processed features that can be fed to a classifier.
        """
        rf = raw_features

        # Make a new dict or pandas DataFrame for the features. Everything is
        # agnostic about whether raw_features is a dict or a pandas DataFrame
        # and the output will be the same as the input.
        features = type(rf)()

        ## Keys that we want to use directly for classification.
        #copy_keys = ["host_photoz", "host_photoz_error"]

        #for copy_key in copy_keys:
        #    features[copy_key] = rf[copy_key]

        #features["length_scale"] = rf["gp_fit_1"]

        max_flux = rf["max_flux_i"]
        max_mag = -2.5 * np.log10(np.abs(max_flux))

        features["max_mag"] = max_mag

        features["pos_flux_ratio"] = rf["max_flux_i"] / (
            rf["max_flux_i"] - rf["min_flux_i"]
        )
        features["max_flux_ratio_red"] = np.abs(rf["max_flux_y"]) / (
            np.abs(rf["max_flux_y"]) + np.abs(rf["max_flux_i"])
        )
        features["max_flux_ratio_blue"] = np.abs(rf["max_flux_g"]) / (
            np.abs(rf["max_flux_i"]) + np.abs(rf["max_flux_g"])
        )

        features["min_flux_ratio_red"] = np.abs(rf["min_flux_lssty"]) / (
            np.abs(rf["min_flux_y"]) + np.abs(rf["min_flux_i"])
        )
        features["min_flux_ratio_blue"] = np.abs(rf["min_flux_g"]) / (
            np.abs(rf["min_flux_i"]) + np.abs(rf["min_flux_g"])
        )

        features["max_dt"] = rf["max_dt_y"] - rf["max_dt_g"]

        features["positive_width"] = rf["positive_width_i"]
        features["negative_width"] = rf["negative_width_i"]

        features["time_fwd_max_0.5"] = rf["time_fwd_max_0.5_i"]
        features["time_fwd_max_0.2"] = rf["time_fwd_max_0.2_i"]

        features["time_fwd_max_0.5_ratio_red"] = rf["time_fwd_max_0.5_y"] / (
            rf["time_fwd_max_0.5_y"] + rf["time_fwd_max_0.5_i"]
        )
        features["time_fwd_max_0.5_ratio_blue"] = rf["time_fwd_max_0.5_g"] / (
            rf["time_fwd_max_0.5_g"] + rf["time_fwd_max_0.5_i"]
        )
        features["time_fwd_max_0.2_ratio_red"] = rf["time_fwd_max_0.2_y"] / (
            rf["time_fwd_max_0.2_y"] + rf["time_fwd_max_0.2_i"]
        )
        features["time_fwd_max_0.2_ratio_blue"] = rf["time_fwd_max_0.2_g"] / (
            rf["time_fwd_max_0.2_g"] + rf["time_fwd_max_0.2_i"]
        )

        features["time_bwd_max_0.5"] = rf["time_bwd_max_0.5_i"]
        features["time_bwd_max_0.2"] = rf["time_bwd_max_0.2_i"]

        features["time_bwd_max_0.5_ratio_red"] = rf["time_bwd_max_0.5_y"] / (
            rf["time_bwd_max_0.5_y"] + rf["time_bwd_max_0.5_i"]
        )
        features["time_bwd_max_0.5_ratio_blue"] = rf["time_bwd_max_0.5_g"] / (
            rf["time_bwd_max_0.5_g"] + rf["time_bwd_max_0.5_i"]
        )
        features["time_bwd_max_0.2_ratio_red"] = rf["time_bwd_max_0.2_y"] / (
            rf["time_bwd_max_0.2_y"] + rf["time_bwd_max_0.2_i"]
        )
        features["time_bwd_max_0.2_ratio_blue"] = rf["time_bwd_max_0.2_g"] / (
            rf["time_bwd_max_0.2_g"] + rf["time_bwd_max_0.2_i"]
        )

        features["frac_s2n_5"] = rf["count_s2n_5"] / rf["count"]
        features["frac_s2n_-5"] = rf["count_s2n_-5"] / rf["count"]
        features["frac_background"] = rf["frac_background"]

        features["time_width_s2n_5"] = rf["time_width_s2n_5"]

        features["count_max_center"] = rf["count_max_center"]
        features["count_max_rise_20"] = (
            rf["count_max_rise_20"] + features["count_max_center"]
        )
        features["count_max_rise_50"] = (
            rf["count_max_rise_50"] + features["count_max_rise_20"]
        )
        features["count_max_rise_100"] = (
            rf["count_max_rise_100"] + features["count_max_rise_50"]
        )
        features["count_max_fall_20"] = (
            rf["count_max_fall_20"] + features["count_max_center"]
        )
        features["count_max_fall_50"] = (
            rf["count_max_fall_50"] + features["count_max_fall_20"]
        )
        features["count_max_fall_100"] = (
            rf["count_max_fall_100"] + features["count_max_fall_50"]
        )

        all_peak_pos_frac_2 = [
            rf["peaks_pos_u_frac_2"],
            rf["peaks_pos_g_frac_2"],
            rf["peaks_pos_r_frac_2"],
            rf["peaks_pos_i_frac_2"],
            rf["peaks_pos_z_frac_2"],
            rf["peaks_pos_y_frac_2"],
        ]

        with np.warnings.catch_warnings():
            np.warnings.filterwarnings("ignore", r"All-NaN slice encountered")
            features["peak_frac_2"] = np.nanmedian(all_peak_pos_frac_2, axis=0)

        features["total_s2n"] = np.sqrt(
            rf["total_s2n_u"] ** 2
            + rf["total_s2n_g"] ** 2
            + rf["total_s2n_r"] ** 2
            + rf["total_s2n_i"] ** 2
            + rf["total_s2n_z"] ** 2
            + rf["total_s2n_y"] ** 2
        )

        all_frac_percentiles = []
        for percentile in (10, 30, 50, 70, 90):
            frac_percentiles = []
            for band in self.filters:
                percentile_flux = rf["percentile_%s_%d" % (band, percentile)]
                max_flux = rf["max_flux_%s" % band]
                min_flux = rf["min_flux_%s" % band]
                frac_percentiles.append(
                    (percentile_flux - min_flux) / (max_flux - min_flux)
                )
            all_frac_percentiles.append(np.nanmedian(frac_percentiles, axis=0))

        features["percentile_diff_10_50"] = (
            all_frac_percentiles[0] - all_frac_percentiles[2]
        )
        features["percentile_diff_30_50"] = (
            all_frac_percentiles[1] - all_frac_percentiles[2]
        )
        features["percentile_diff_70_50"] = (
            all_frac_percentiles[3] - all_frac_percentiles[2]
        )
        features["percentile_diff_90_50"] = (
            all_frac_percentiles[4] - all_frac_percentiles[2]
        )

        return features


def find_time_to_fractions(fluxes, fractions, forward=True):
    """Adapted from avocado"""
    """Find the time for a lightcurve to decline to specific fractions of
    maximum light.
    Parameters
    ----------
    fluxes : numpy.array
        A list of GP-predicted fluxes at 1 day intervals.
    fractions : list
        A decreasing list of the fractions of maximum light that will be found
        (eg: [0.8, 0.5, 0.2]).
    forward : bool
        If True (default), look forward in time. Otherwise, look backward in
        time.
    Returns
    -------
    times : numpy.array
        A list of times for the lightcurve to decline to each of the given
        fractions of maximum light.
    """
    max_time = np.argmax(fluxes)
    max_flux = fluxes[max_time]

    result = np.zeros(len(fractions))
    result[:] = np.nan

    frac_idx = 0

    # Start at maximum light, and move along the spectrum. Whenever we cross
    # one threshold, we add it to the list and keep going. If we hit the end of
    # the array without crossing the threshold, we return a large number for
    # that time.
    offset = 0
    while True:
        offset += 1
        if forward:
            new_time = max_time + offset
            if new_time >= fluxes.shape:
                break
        else:
            new_time = max_time - offset
            if new_time < 0:
                break

        test_flux = fluxes[new_time]
        while test_flux < max_flux * fractions[frac_idx]:
            result[frac_idx] = offset
            frac_idx += 1
            if frac_idx == len(fractions):
                break

        if frac_idx == len(fractions):
            break

    return result



def fit_snpcc_bazin(path_to_data_dir: str, features_file: str):
    """Perform Bazin fit to all objects in the SNPCC data.

    Parameters
    ----------
    path_to_data_dir: str
        Path to directory containing the set of individual files, one for each light curve.
    features_file: str
        Path to output file where results should be stored.
    """

    # read file names
    file_list_all = os.listdir(path_to_data_dir)
    lc_list = [elem for elem in file_list_all if 'DES_SN' in elem]

    # count survivers
    count_surv = 0

    # add headers to files
    with open(features_file, 'w') as param_file:
        param_file.write('id redshift type code orig_sample gA gB gt0 gtfall gtrise rA rB rt0 rtfall rtrise iA iB it0 ' +
                         'itfall itrise zA zB zt0 ztfall ztrise\n')

    for file in lc_list:

        # fit individual light curves
        lc = LightCurve()
        lc.load_snpcc_lc(path_to_data_dir + file)
        lc.fit_bazin_all()

        print(lc_list.index(file), ' - id:', lc.id)

        # append results to the correct matrix
        if 'None' not in lc.bazin_features:
            count_surv = count_surv + 1
            print('Survived: ', count_surv)

            # save features to file
            with open(features_file, 'a') as param_file:
                param_file.write(str(lc.id) + ' ' + str(lc.redshift) + ' ' + str(lc.sntype) + ' ')
                param_file.write(str(lc.sncode) + ' ' + str(lc.sample) + ' ')
                for item in lc.bazin_features:
                    param_file.write(str(item) + ' ')
                param_file.write('\n')

    param_file.close()

def fit_resspect_bazin(path_photo_file: str, path_header_file:str,
                       output_file: str, sample=None):
    """Perform Bazin fit to all objects in a given RESSPECT data file.

    Parameters
    ----------
    path_photo_file: str
        Complete path to light curve file.
    path_header_file: str
        Complete path to header file.
    output_file: str
        Output file where the features will be stored.
    sample: str
	'train' or 'test'. Default is None.
    """
    # count survivers
    count_surv = 0

    # read header information
    if '.tar.gz' in path_header_file:
        tar = tarfile.open(path_header_file, 'r:gz')
        fname = tar.getmembers()[0]
        content = tar.extractfile(fname).read()
        header = pd.read_csv(io.BytesIO(content))
        tar.close()
    elif 'FITS' in path_header_file:
        header, photo = read_fits(path_photo_file, drop_separators=True)    
    else:    
        header = pd.read_csv(path_header_file, index_col=False)
        if ' ' in header.keys()[0]:
            header = pd.read_csv(path_header_file, sep=' ', index_col=False)
    
    # add headers to files
    with open(output_file, 'w') as param_file:
        param_file.write('id redshift type code orig_sample uA uB ut0 utfall ' +
                         'utrise gA gB gt0 gtfall gtrise rA rB rt0 rtfall ' +
                         'rtrise iA iB it0 itfall itrise zA zB zt0 ztfall ' + 
                         'ztrise YA YB Yt0 Ytfall Ytrise\n')

    # check id flag
    if 'SNID' in header.keys():
        id_name = 'SNID'
    elif 'snid' in header.keys():
        id_name = 'snid'
    elif 'objid' in header.keys():
        id_name = 'objid'

    # check redshift flag
    if 'redshift' in header.keys():
        z_name = 'redshift'
    elif 'REDSHIFT_FINAL' in header.keys():
        z_name = 'REDSHIFT_FINAL'

    # check type flag
    if 'type' in header.keys():
        type_name = 'type'
    elif 'SIM_TYPE_NAME' in header.keys():
        type_name = 'SIM_TYPE_NAME'
    elif 'TYPE' in header.keys():
        type_name = 'TYPE'

    # check subtype flag
    if 'code' in header.keys():
        subtype_name = 'code'
    elif 'SIM_TYPE_INDEX' in header.keys():
        subtype_name = 'SIM_TYPE_NAME'
    elif 'SNTYPE_SUBCLASS' in header.keys():
        subtype_name = 'SNTYPE_SUBCLASS'

    for snid in header[id_name].values:      

        # load individual light curves
        lc = LightCurve()                       
        lc.load_resspect_lc(path_photo_file, snid)

        # fit all bands                
        lc.fit_bazin_all()

        # get model name 
        lc.redshift = header[z_name][header[lc.id_name] == snid].values[0]
        lc.sntype = header[type_name][header[lc.id_name] == snid].values[0]
        lc.sncode = header[subtype_name][header[lc.id_name] == snid].values[0]
        lc.sample = sample

        # append results to the correct matrix
        if 'None' not in lc.bazin_features:
            count_surv = count_surv + 1
            print('Survived: ', count_surv)

            # save features to file
            with open(output_file, 'a') as param_file:
                param_file.write(str(lc.id) + ' ' + str(lc.redshift) + ' ' + str(lc.sntype) + ' ')
                param_file.write(str(lc.sncode) + ' ' + str(lc.sample) + ' ')
                for item in lc.bazin_features:
                    param_file.write(str(item) + ' ')
                param_file.write('\n')

    param_file.close()


def fit_plasticc_bazin(path_photo_file: str, path_header_file:str,
                       output_file: str, sample=None):
    """Perform Bazin fit to all objects in a given PLAsTiCC data file.

    Parameters
    ----------
    path_photo_file: str
        Complete path to light curve file.
    path_header_file: str
        Complete path to header file.
    output_file: str
        Output file where the features will be stored.
    sample: str
	'train' or 'test'. Default is None.
    """
    types = {90: 'Ia', 67: '91bg', 52:'Iax', 42:'II', 62:'Ibc', 
             95: 'SLSN', 15:'TDE', 64:'KN', 88:'AGN', 92:'RRL', 65:'M-dwarf',
             16:'EB',53:'Mira', 6:'MicroL', 991:'MicroLB', 992:'ILOT', 
             993:'CART', 994:'PISN',995:'MLString'}

    # count survivers
    count_surv = 0

    # read header information
    if '.tar.gz' in path_header_file:
        tar = tarfile.open(path_header_file, 'r:gz')
        fname = tar.getmembers()[0]
        content = tar.extracfile(fname).read()
        header = pd.read_csv(io.BytesIO(content))
        tar.close()
    else:
        header = pd.read_csv(path_header_file, index_col=False)

        if ' ' in header.keys()[0]:
            header = pd.read_csv(path_header_file, sep=' ', index_col=False)

    # add headers to files
    with open(output_file, 'w') as param_file:
        param_file.write('id redshift type code orig_sample uA uB ut0 utfall ' +
                         'utrise gA gB gt0 gtfall gtrise rA rB rt0 rtfall ' +
                         'rtrise iA iB it0 itfall itrise zA zB zt0 ztfall ' + 
                         'ztrise YA YB Yt0 Ytfall Ytrise\n')

    # check id flag
    if 'SNID' in header.keys():
        id_name = 'SNID'
    elif 'snid' in header.keys():
        id_name = 'snid'
    elif 'objid' in header.keys():
        id_name = 'objid'

    for snid in header[id_name].values:      

        # load individual light curves
        lc = LightCurve()                       
        lc.load_plasticc_lc(path_photo_file, snid) 
        lc.fit_bazin_all()

        # get model name 
        lc.redshift = header['true_z'][header[lc.id_name] == snid].values[0]
        lc.sntype = types[header['true_target'][header[lc.id_name] == snid].values[0]]            
        lc.sncode = header['true_target'][header[lc.id_name] == snid].values[0]
        lc.sample = sample

        # append results to the correct matrix
        if 'None' not in lc.bazin_features:
            count_surv = count_surv + 1
            print('Survived: ', count_surv)

            # save features to file
            with open(output_file, 'a') as param_file:
                param_file.write(str(lc.id) + ' ' + str(lc.redshift) + ' ' + str(lc.sntype) + ' ')
                param_file.write(str(lc.sncode) + ' ' + str(lc.sample) + ' ')
                for item in lc.bazin_features:
                    param_file.write(str(item) + ' ')
                param_file.write('\n')

    param_file.close()


def main():
    return None


if __name__ == '__main__':
    main()
