import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
from scipy.interpolate import interp1d

class DeltaMaps:
    """
    Handle LSST maglim maps (FITS) and apply them to catalogs.
    """

    def __init__(self, maglim_fits):
        """
        Parameters
        ----------
        maglim_fits : dict
            {band: fits_filename}
            e.g. {"r": "ECDFS_maglim_r.fits", "i": "ECDFS_maglim_i.fits"}
        """
        self.maps = {}

        for band, fname in maglim_fits.items():
            with fits.open(fname) as hdul:
                data = hdul[0].data
                wcs = WCS(hdul[0].header)

            self.maps[band] = {
                "data": data,
                "wcs": wcs,
                "shape": data.shape,
            }

    def bands(self):
        """Return available bands."""
        return list(self.maps.keys())

    def get_map(self, band):
        """Return (data, wcs) for a given band."""
        if band not in self.maps:
            raise ValueError(f"Band '{band}' not loaded.")
        return self.maps[band]["data"], self.maps[band]["wcs"]

    def get_maglim_at_radec(self, band, ra, dec):
        """
        Return maglim values at given RA/Dec positions.

        Parameters
        ----------
        band : str
        ra, dec : array-like (deg)

        Returns
        -------
        maglim : ndarray
        """
        data, wcs = self.get_map(band)

        x, y = wcs.world_to_pixel_values(ra, dec)
        # print(x,y)
        x = np.round(x).astype(int)
        y = np.round(y).astype(int)
        ny, nx = data.shape
        good = (x >= 0) & (x < nx) & (y >= 0) & (y < ny)

        maglim = np.full(len(ra), np.nan)
        maglim[good] = data[y[good], x[good]]

        return maglim

    def add_delta_to_catalog(self, cat, mag_columns, radec_names=["ra","dec"]):
        """
        Add maglim_band and delta_band columns to a catalog.

        Parameters
        ----------
        cat : pandas.DataFrame
            Must contain 'ra', 'dec' and magnitude columns.
        mag_columns : dict
            {band: mag_column_name}

        Returns
        -------
        cat : pandas.DataFrame
        """
        for band, mag_col in mag_columns.items():
            if band not in self.maps:
                raise ValueError(f"Band '{band}' not loaded.")

            maglim = self.get_maglim_at_radec(
                band,
                cat[f"{radec_names[0]}"].values,
                cat[f"{radec_names[1]}"].values,
            )
            cat[f"maglim_{band}"] = maglim
            cat[f"delta_{band}"] = cat[mag_col] - maglim
        # print(cat[f"maglim_{band}"])
        return cat

    def plot_map(
        self,
        band,
        vmin=None,
        vmax=None,
        cmap="viridis",
        figsize=(8, 5),
    ):
        """
        Plot the maglim map for a given band using imshow.
        """
        data, wcs = self.get_map(band)

        plt.figure(figsize=figsize)

        im = plt.imshow(
            data,
            origin="lower",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            aspect="auto",
        )

        plt.colorbar(im, label=f"PSF mag_lim (5σ, {band}-band)")
        plt.xlabel("Pixel X")
        plt.ylabel("Pixel Y")
        plt.title(f"Maglim map – {band}-band")

        plt.tight_layout()
        plt.show()



class PhotometricErrorModel:
    """
    Conditional log-normal model for photometric errors:
        p(sigma | x)
    """

    # 1. Fit parameters
    def fit_params(self, x, sigma, x_bins=15, min_per_bin=30):
        """
        Estimate log-normal parameters of photometric errors
        conditioned on a variable x.

        Parameters
        ----------
        x : array-like
            Conditioning variable (e.g. delta = mag - mag_lim, or magnitude).
        sigma : array-like
            Observed photometric uncertainties (> 0).
        x_bins : int or array-like, optional
            If int, number of quantile bins in x.
            If array-like, explicit bin edges.
        min_per_bin : int, optional
            Minimum number of data points required per bin.

        Returns
        -------
        bin_centers : ndarray
            Centers of the valid x bins.
        mu : ndarray
            Mean of ln(sigma) in each bin.
        tau : ndarray
            Standard deviation of ln(sigma) in each bin.
        """

        x = np.asarray(x)
        sigma = np.asarray(sigma)

        good = np.isfinite(x) & np.isfinite(sigma) & (sigma > 0)
        x = x[good]
        sigma = sigma[good]

        if isinstance(x_bins, int):
            x_bins = np.quantile(x, np.linspace(0, 1, x_bins + 1))

        bin_centers, mu, tau = [], [], []

        for x0, x1 in zip(x_bins[:-1], x_bins[1:]):
            sel = (x >= x0) & (x < x1)

            if sel.sum() < min_per_bin:
                continue

            ln_sigma = np.log(sigma[sel])
            bin_centers.append(0.5 * (x0 + x1))
            mu.append(np.mean(ln_sigma))
            tau.append(np.std(ln_sigma, ddof=1))

        return (
            np.asarray(bin_centers),
            np.asarray(mu),
            np.asarray(tau),
        )

    # 2. Build interpolated model
    def fit(self, bin_centers, mu, tau, interp_kind="linear"):
        """
        Build interpolated conditional model from fitted parameters.
        """

        self.mu_interp = interp1d(
            bin_centers, mu,
            kind=interp_kind,
            fill_value="extrapolate"
        )

        self.tau_interp = interp1d(
            bin_centers, tau,
            kind=interp_kind,
            fill_value="extrapolate"
        )

        return self

    # 3. Prediction
    def predict(self, x, statistic="median"):
        """
        Predict expected sigma for given x.
        """
        x = np.asarray(x)

        mu = self.mu_interp(x)
        tau = self.tau_interp(x)

        if statistic == "median":
            return np.exp(mu)
        elif statistic == "mean":
            return np.exp(mu + 0.5 * tau**2)
        else:
            raise ValueError("statistic must be 'median' or 'mean'")

    # 3bis. Sampling
    def sample(self, x):
        """
        Sample sigma from p(sigma | x).
        """
        x = np.asarray(x)

        mu = self.mu_interp(x)
        tau = self.tau_interp(x)

        return np.random.lognormal(mean=mu, sigma=tau)

    # 4. Store models in a collection for re-use
    def model_collection(self, x_lists, sigma_lists, model_names,
        x_bins=15, min_per_bin=30, interp_kind="linear", fit_return="fit"):
        """
        Build a collection of photometric error models.

        Parameters
        ----------
        x_lists : array-like (1D or 2D)
            Conditioning variables. Shape (Nmodels, Npoints) or (Npoints,).
        sigma_lists : array-like (1D or 2D)
            Photometric uncertainties corresponding to x_lists.
        model_names : str or list of str
            Names of the models (e.g. band names).
        x_bins : int or array-like, optional
            Binning definition for x.
        min_per_bin : int, optional
            Minimum number of points per bin.
        interp_kind : str, optional
            Interpolation kind.
        fit_return : {"fit", "fit_params"}, optional
            If "fit", return fitted models.
            If "fit_params", return (bin_centers, mu, tau).

        Returns
        -------
        models : dict
            Dictionary of models or fitted parameters.
        """

        # manage entries
        if hasattr(x_lists, "values"):
            x_lists = x_lists.values.T
        else:
            x_lists = np.asarray(x_lists)
        if hasattr(sigma_lists, "values"):
            sigma_lists = sigma_lists.values.T
        else:
            sigma_lists = np.asarray(sigma_lists)

        x_lists = np.atleast_2d(x_lists)
        sigma_lists = np.atleast_2d(sigma_lists)


        if isinstance(model_names, str):
            model_names = [model_names]

        if len(model_names) != x_lists.shape[0]:
            raise ValueError("model_names must match number of x_lists")

        models = {}

        for i, name in enumerate(model_names):
            x = x_lists[i]
            sigma = sigma_lists[i]

            bin_centers, mu, tau = self.fit_params(
                x, sigma,
                x_bins=x_bins,
                min_per_bin=min_per_bin
            )

            if fit_return == "fit_params":
                models[name] = (bin_centers, mu, tau)

            elif fit_return == "fit":
                model = PhotometricErrorModel()
                model.fit(
                    bin_centers, mu, tau,
                    interp_kind=interp_kind
                )
                models[name] = model

            else:
                raise ValueError("fit_return must be 'fit' or 'fit_params'")

        return models

    def plot_models(self, models, x_lists, sigma_lists, model_names=None,
        show_scatter=True, show_model=True, statistic="median", subsample=None):
        """
        Plot photometric error models.

        Parameters
        ----------
        models : dict
            Dictionary of PhotometricErrorModel objects.
        x_lists : array-like or DataFrame
            Conditioning variables used to fit the models.
        sigma_lists : array-like or DataFrame
            Observed photometric uncertainties.
        model_names : list of str or None
            Names of models to plot. If None, plot all.
        show_scatter : bool, optional
            Whether to show observed data scatter.
        show_model : bool, optional
            Whether to show model prediction.
        statistic : {"median", "mean"}, optional
            Statistic to plot for the model.
        subsample : int or None, optional
            Random subsampling factor for scatter points.
        """

        # --- Handle DataFrame / array ---
        if hasattr(x_lists, "values"):
            x_lists = x_lists.values.T
        else:
            x_lists = np.asarray(x_lists)

        if hasattr(sigma_lists, "values"):
            sigma_lists = sigma_lists.values.T
        else:
            sigma_lists = np.asarray(sigma_lists)

        x_lists = np.atleast_2d(x_lists)
        sigma_lists = np.atleast_2d(sigma_lists)

        if model_names is None:
            model_names = list(models.keys())

        for i, name in enumerate(model_names):
            model = models[name]
            x = x_lists[i]
            sigma = sigma_lists[i]

            if subsample is not None and subsample < len(x):
                idx = np.random.choice(len(x), subsample, replace=False)
                x = x[idx]
                sigma = sigma[idx]

            plt.figure(figsize=(6, 4))

            if show_scatter:
                plt.scatter(x, sigma, s=5, alpha=0.3, label="Observed")

            if show_model:
                x_grid = np.linspace( np.nanmin(x), np.nanmax(x), 300)
                sigma_model = model.predict(x_grid, statistic=statistic)

                plt.plot(
                    x_grid,
                    sigma_model,
                    color="black",
                    lw=2,
                    label=f"Model ({statistic})"
                )

            plt.xlabel("x")
            plt.ylabel(r"$\sigma_{\rm mag}$")
            plt.title(f"Photometric error model – {name}")
            plt.legend()
            plt.tight_layout()
            plt.show()

