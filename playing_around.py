import sncosmo
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from astropy.table import Table
import os
import glob

data = sncosmo.load_example_data()


#need to read in the real data

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
    model = sncosmo.Model(source='salt2')

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