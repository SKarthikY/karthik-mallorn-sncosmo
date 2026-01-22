import sncosmo
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from astropy.table import Table
import os
import glob
import astropy.units as u
import astropy.constants as co
import astropy.cosmology.units as cu
from scipy.interpolate import CubicSpline
from scipy.integrate import cumulative_trapezoid as cumtrapz

DAY_CGS = u.day
M_SUN_CGS = co.M_sun
C_CGS = co.c
beta = 13.7
KM_CGS = u.km

STEF_CONST = 4. * np.pi * co.sigma_sb
ANG_CGS = u.Angstrom
MPC_CGS = u.Mpc

DIFF_CONST = 2.0 * M_SUN_CGS / (beta * C_CGS * KM_CGS)
TRAP_CONST = 3.0 * M_SUN_CGS / (4. * np.pi * KM_CGS ** 2)
FLUX_CONST = 4.0 * np.pi * (
        2.0 * co.h * co.c ** 2 * np.pi) * u.Angstrom
X_CONST = (co.h * co.c / co.k_B)

class ArnettSource(sncosmo.Source):

    _param_names = ['texp', 'mej', 'fni', 'vej']
    param_names = ['texp', 'mej', 'fni', 'vej']

    param_names_latex = ['t_{exp}', 'M_{ej}', 'f_{Ni}', 'v_{ej}']   # used in plotting display

    def __init__(self, phase, wave, redshift,  params=None, name=None, version=None):
        
        self.name = name
        self.version = version
        self._phase = phase
        self._wave = wave
        self._z = redshift
        # self._texp = texp
        self._tfloor = 3000 * u.K
        if params is not None:
            self._parameters = params
        else:
            self._parameters = [0, 0, 0, 0]

    def _blackbody_flux(self, temperature, radius, wavelength):
        # Convert wavelength from Angstrom to cm

        all_fluxes = np.zeros((len(temperature), len(wavelength))) * (u.erg / (u.s * u.AA)).decompose()
        for i in range(len(temperature)):
            temp = temperature[i]
            rad = radius[i]

            # Planck function numerator: 2hc^2 / λ^5
            numerator = (2 * co.h * co.c**2 / wavelength**5).decompose()

            # Exponent of the Planck function
            exponent = (co.h * co.c / (wavelength * co.k_B * temp)).decompose()

            # Compute denominator, stripping unit to pass into np.exp safely
            denominator = np.exp(exponent.value) - 1

            # Spectral radiance: Planck's law in SI units (W / (m^2 m))
            flux_density = numerator / denominator

            # Convert to flux at the source: multiply by 4πR²
            flux_final = flux_density * (4 * np.pi * rad**2)

            # Assign to all_fluxes
            all_fluxes[i, :] = flux_final
        return all_fluxes 


    def _gen_arnett_model(self, t, wvs, redshift, theta):
        texp, mej, fni, vej = theta
        mej = (mej * u.Msun).to(u.g)
        vej = vej * u.km/u.s
        t = t * u.day
        wvs = wvs * u.AA
        tfloor = self._tfloor
        mni = mej * fni
        # Convert velocity to cm/s
        vej = vej.to(u.cm / u.s)

        tni = 8.8 * u.day  # days
        tco = 111.3  * u.day # days
        epco = 6.8e9 * u.erg / u.g / u.s # erg/g/s
        epni = 3.9e10 * u.erg / u.g / u.s # erg/g/s
        opac = 0.1 *u.cm * u.cm/u.g        
        texp = texp * u.day

        # print("hello")
        # Diffusion timescale in days
        td = np.sqrt(2 * opac * mej / (13.7 * co.c * vej)).to(u.day)   # convert seconds to days
 
        # use a denser time array for better integration
        t_to_integrate = np.linspace(0, np.max(t - texp), 1000)
        
        # td = np.sqrt(2 * opac * mej / (13.7 * C_CGS * vej)) / 86400  # convert seconds to days
        integrand1 = (t_to_integrate / td) * np.exp(t_to_integrate**2 / td**2 - t_to_integrate / tni)
        # print(np.exp(t_to_integrate**2 / td**2 - t_to_integrate / tni))
        integrand2 = (t_to_integrate / td) * np.exp(t_to_integrate**2 / td**2 - t_to_integrate / tco)
        # print(np.exp(t_to_integrate**2 / td**2 - t_to_integrate / tco))
        # print("a")
        # print(td)
        # print(t_to_integrate)
        # print("1", integrand1)
        # print("2", integrand2)


        # evaluate np.exp(-t_to_integrate**2 / td**2) and set to zero if too small

        # Luminosity calculation
        dense_luminosities = 2 * mni / (td) * np.exp(-t_to_integrate**2 / td**2) * \
              (((epni - epco) * cumtrapz(integrand1, t_to_integrate, initial=0) + 
               epco * cumtrapz(integrand2, t_to_integrate, initial=0)))*u.day # these should be erg/s
        # print("b")
        # print(dense_luminosities)
        spline = CubicSpline(t_to_integrate, dense_luminosities, extrapolate = False)
        # print("c")
        luminosities = spline(t - texp) * u.erg / u.s # interpolate back from dense time array to original time points
        # print(luminosities)
        #Do BB calculation
        radius = (vej * ((t - texp) * ((t-texp)>=0))).to(u.cm)

        temperature = ((luminosities / (STEF_CONST * radius**2))**0.25).to(u.K)# * (1e52)**0.25
        # gind = (temperature < tfloor) | np.isnan(temperature)
        # temperature = np.nan_to_num(temperature)
        # notgind = np.invert(gind)
        # temperature = (0. * temperature) + (temperature * notgind) + (tfloor * gind)

                # Set temperature floor before radius calculation
        temperature = np.maximum(temperature, tfloor)
        
        # print(temperature)

        radius = np.sqrt(luminosities / (STEF_CONST * temperature**4))
        radius = radius.to(u.cm)

        # fluxes = self._blackbody_flux(temperature, radius, wvs) # this is a luminosity density
        
        fluxes = self._blackbody_flux(temperature, radius, wvs / (1 + redshift)) # this is a luminosity density

        fluxes[t < texp,:] = 0.* u.kg *u.m / u.s**3
        fluxes[t < 0,:] = 0.* u.kg *u.m / u.s**3
        fluxes[np.isnan(fluxes)] = 0.* u.kg *u.m / u.s**3

        
        # divide luminosity density by 4pir^2 to get flux
        z = redshift * cu.redshift
        d_cm = z.to(u.cm, cu.redshift_distance(cosmo, kind="luminosity"))
        flux_density = fluxes / (4 * np.pi * d_cm**2) 
        return flux_density / (1 + redshift)


    def _flux(self, phase, wave):
        return self._gen_arnett_model(phase, wave, self._z, self._parameters)




times = np.linspace(0.1,100,100) # days
wavelengths = np.linspace(2000,12000,10) # Angstroms
arnett_source = ArnettSource(times, wavelengths, true_z) # initial parameter values? not sure why these are needed...

to_fit = 100  #number of lightcurves to fit
fitted = 0
fitted_params = {'z':[], 't0':[], 'x0':[], 'x1':[], 'c':[], 'chi_sq':[]}
spec_types = []
dir = '/n/home07/kyadavalli/kyadavalli/kaggle_challenge/train_SNe/'
files = glob.glob(dir+'*_lightcurve.csv')
for file in files:
    if fitted > to_fit:
        continue
    fitted += 1

    df=pd.read_csv(file)
    obj_name = file.split('/')[-1].replace('_lightcurve.csv','')
    print("obj name: "+str(obj_name))


    # Use dereddened fluxes; if you prefer raw flux, swap to "Flux"
    df["time"]    = df["Time (MJD)"]
    df["flux"]    = df["Flux_dered"]
    df["fluxerr"] = df["Flux_err_dered"]

    # Band names: sncosmo expects something like "sdssg", "sdssr", etc.
    # If your 'g','r','i','z' correspond to SDSS, do:
    band_map = {
        "g": "sdssg",
        "r": "sdssr",
        "i": "sdssi",
        "z": "sdssz"
    }
    df["band"] = df["Filter"].map(band_map)

    # Constant AB zeropoint (from your mag/flux relation)
    df["zp"] = 23.9
    df["zpsys"] = "ab"

    # Keep only needed columns for sncosmo
    df_sncosmo = df[["time", "band", "flux", "fluxerr", "zp", "zpsys"]].copy()

    # Drop any rows where band map failed (NaN) just in case
    df_sncosmo = df_sncosmo.dropna(subset=["band"])

    # Convert to astropy Table
    data = Table.from_pandas(df_sncosmo)



    # create a model
    #model = sncosmo.Model(source='salt2')
    model = sncosmo.Model(source=arnett_source)

    # run the fit
    result, fitted_model = sncosmo.fit_lc(
        data, model,
        ['z', 't0', 'x0', 'x1', 'c'],  # parameters of model to vary
        bounds={'z':(0.3, 0.7)})  # bounds on parameters (if any)


    
    fitted_params['z'].append(result.parameters[0])
    fitted_params['t0'].append(result.parameters[1])
    fitted_params['x0'].append(result.parameters[2])
    fitted_params['x1'].append(result.parameters[3])
    fitted_params['c'].append(result.parameters[4])
    fitted_params['chi_sq'].append(result.chisq)

    #read in train_log.csv file
    log_df = pd.read_csv('/n/home07/kyadavalli/kyadavalli/kaggle_challenge/data/train_log.csv')

    #find where object_id == obj_name in log_df
    spec_type = log_df[log_df['object_id'] == obj_name]['SpecType'].values[0]
    spec_types.append(spec_type)


    #plot the lightcurve and the fit
    os.makedirs('lightcurve_fits/', exist_ok=True)
    sncosmo.plot_lc(data, model=fitted_model, fname=f'lightcurve_fits/{obj_name}_fit.pdf')


unique_spec_types = list(set(spec_types))
os.makedirs('fitted_param_histograms/', exist_ok=True)


for key in fitted_params:
    for unique_type in unique_spec_types:

        type_idx = np.where(np.array(spec_types) == unique_type)[0]
        plt.hist(np.array(fitted_params[key])[type_idx], alpha=0.5, label=unique_type, density=True, bins=30)
        print("For spec type "+str(unique_type)+", mean "+str(key)+": "+str(np.mean(np.array(fitted_params[key])[type_idx]))+", std: "+str(np.std(np.array(fitted_params[key])[type_idx])))
    plt.title(key)
    plt.legend()
    plt.savefig(f'fitted_param_histograms/{key}_hist.pdf', bbox_inches='tight')
    plt.close()