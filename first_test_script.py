import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from extinction import fitzpatrick99
from astropy.cosmology import Planck18
import os

# Defining function to de-extinct a set of flux values
def jurassic_park (flux, eff_wl, ebv):
    eff_wl_array = np.array([eff_wl]) if np.isscalar(eff_wl) else eff_wl
    A_lambda = fitzpatrick99(eff_wl_array, ebv * 3.1) #3.1 = Standard Milky Way value
    flux_ext = flux * 10**((A_lambda[0])/2.5)
    return flux_ext, A_lambda[0]

u_eff_wl = np.array([3641]); g_eff_wl = np.array([4704]); r_eff_wl = np.array([6155])
i_eff_wl = np.array([7504]); z_eff_wl = np.array([8695]); y_eff_wl = np.array([10056])

# Filter wavelength map
filter_wl_map = {
    'u': u_eff_wl[0],
    'g': g_eff_wl[0],
    'r': r_eff_wl[0],
    'i': i_eff_wl[0],
    'z': z_eff_wl[0],
    'y': y_eff_wl[0]
}

AB_ZEROPOINT_MUJY = 3.631e9  # microjansky

def extract_features_for_object(object_row, lightcurve_data):
    """Extract features for a single object"""
    object_id = object_row['object_id']
    ebv = object_row['EBV']
    redshift = object_row['Z']
    
    # Filter light curve for this object
    lightcurve = lightcurve_data[lightcurve_data['object_id'] == object_id].copy()
    
    if len(lightcurve) == 0:
        return None
    
    # Deredden the flux based on filter
    def deredden_row(row):
        filter_name = row['Filter']
        eff_wl = filter_wl_map[filter_name]
        flux_dered, A_lambda = jurassic_park(row['Flux'], eff_wl, ebv)
        # Deredden flux error by same factor
        flux_err_dered = row['Flux_err'] * 10**((A_lambda)/2.5)
        return flux_dered, A_lambda, flux_err_dered
    
    lightcurve[['Flux_dered', 'A_lambda', 'Flux_err_dered']] = lightcurve.apply(
        lambda row: pd.Series(deredden_row(row)), axis=1
    )
    
    # Convert flux to magnitude
    positive_flux_mask = lightcurve['Flux_dered'] > 0
    lightcurve['Magnitude'] = np.nan
    lightcurve['Magnitude_err'] = np.nan
    lightcurve.loc[positive_flux_mask, 'Magnitude'] = -2.5 * np.log10(lightcurve.loc[positive_flux_mask, 'Flux_dered'] / AB_ZEROPOINT_MUJY)
    lightcurve.loc[positive_flux_mask, 'Magnitude_err'] = (2.5 / np.log(10)) * (lightcurve.loc[positive_flux_mask, 'Flux_err_dered'] / lightcurve.loc[positive_flux_mask, 'Flux_dered'])
    
    # Calculate SNR
    lightcurve['SNR'] = np.abs(lightcurve['Flux_dered'] / lightcurve['Flux_err_dered'])
    
    # Calculate distance modulus and convert to absolute magnitudes
    distmod = Planck18.distmod(redshift).value
    lightcurve['Abs_Magnitude'] = lightcurve['Magnitude'] - distmod
    
    # Apply k-correction
    k_correction = 2.5 * np.log10(1 + redshift)
    lightcurve['Abs_Magnitude_k'] = lightcurve['Abs_Magnitude'] + k_correction
    
    # Convert to rest-frame time
    lightcurve['Time_rest'] = lightcurve['Time (MJD)'] / (1 + redshift)
    
    # Extract features for SNR > 3 points
    high_snr_mask = (lightcurve['SNR'] > 3) & lightcurve['Abs_Magnitude_k'].notna()
    high_snr_data = lightcurve[high_snr_mask].copy()
    
    if len(high_snr_data) == 0:
        return None
    
    # Peak absolute magnitude (minimum magnitude = brightest)
    peak_abs_mag = high_snr_data['Abs_Magnitude_k'].min()
    
    # Duration: max time - min time (rest-frame)
    duration_max_min = high_snr_data['Time_rest'].max() - high_snr_data['Time_rest'].min()
    
    # Time to fall by 1 mag
    peak_time = high_snr_data.loc[high_snr_data['Abs_Magnitude_k'].idxmin(), 'Time_rest']
    post_peak = high_snr_data[high_snr_data['Time_rest'] > peak_time]
    target_mag = peak_abs_mag + 1.0
    time_to_fall_1mag = np.nan
    if len(post_peak) > 0:
        fallen_points = post_peak[post_peak['Abs_Magnitude_k'] >= target_mag]
        if len(fallen_points) > 0:
            time_to_fall_1mag = fallen_points['Time_rest'].min() - peak_time
        else:
            time_to_fall_1mag = post_peak['Time_rest'].max() - peak_time
    
    # Get redshift error
    redshift_err = object_row.get('Z_err', np.nan)
    if pd.isna(redshift_err) or redshift_err == '':
        redshift_err = np.nan

    os.makedirs('test_SNe/', exist_ok=True)
    #write the light curve for this supernova to this directory
    if not os.path.exists(f'test_SNe/{object_id}_lightcurve.csv'):
        high_snr_data.to_csv(f'test_SNe/{object_id}_lightcurve.csv', index=False)

    return {
        'object_id': object_id,
        'peak_abs_mag': peak_abs_mag,
        'duration_max_min': duration_max_min,
        'time_to_fall_1mag': time_to_fall_1mag,
        'redshift': redshift,
        'redshift_err': redshift_err,
        'target': object_row['target']
    }

# Read train_log.csv
train_log = pd.read_csv('data/test_log.csv')
print(f"Processing {len(train_log)} objects...")

# Group by split to avoid loading lightcurve files multiple times
all_features = []
for split_name in train_log['split'].unique():
    split_objects = train_log[train_log['split'] == split_name]
    print(f"\nProcessing split {split_name} ({len(split_objects)} objects)...")
    
    # Load lightcurve data for this split
    lightcurve_path = f'data/{split_name}/test_full_lightcurves.csv'
    lightcurves = pd.read_csv(lightcurve_path)
    
    # Process each object in this split
    for idx, object_row in split_objects.iterrows():
        features = extract_features_for_object(object_row, lightcurves)
        if features is not None:
            all_features.append(features)
        
        # Progress update every 100 objects
        if len(all_features) % 100 == 0:
            print(f"  Processed {len(all_features)} objects...")

# Convert to DataFrame and save
features_df = pd.DataFrame(all_features)
features_df.to_csv('test_lightcurve_features.csv', index=False)

print(f"\n✓ Successfully extracted features for {len(all_features)} objects")
print(f"✓ Features saved to lightcurve_features.csv")
print(f"\nFeatures summary:")
print(features_df.describe())

