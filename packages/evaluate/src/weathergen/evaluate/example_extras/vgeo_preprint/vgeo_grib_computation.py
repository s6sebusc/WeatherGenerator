# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "cfgrib>=0.9.15.1",
#     "eccodes>=2.44.0",
#     "matplotlib",
#     "metpy>=1.7.1",
#     "netcdf4>=1.7.4",
#     "numpy~=2.2",
#     "scipy",
#     "tqdm",
#     "xarray>=2025.6.1",
#     "joblib",
# ]
# ///
import xarray as xr
import numpy as np
from utils import regularise_dataset
from metpy.calc import geostrophic_wind, ageostrophic_wind
from metpy.units import units
from joblib import Parallel, delayed
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path
from itertools import product

def geo_ageo(ds):
    z = ds.z / 9.81 * units.m
    u = ds.u * units.m / units.s
    v = ds.v * units.m / units.s
    ug, vg = geostrophic_wind(z)
    uag, vag = ageostrophic_wind(z, u, v)
    Vg = np.sqrt(ug**2+vg**2).where( (np.abs(ug.lat)>30) & (np.abs(ug.lat)<85) )
    Vag = np.sqrt(uag**2+vag**2).where( (np.abs(ug.lat)>30) & (np.abs(ug.lat)<85) )
    return Vg, Vag

def compute_time_step(dat, time_index, step_index):
    # go from reduced to full Gaussian
    dat_r = regularise_dataset(
        dat.isel(time=time_index, step=step_index), gridpointdim="values"
    ).drop_vars("valid_time")
    # compute geostrophic and ageostrophic wind speeds
    Vg, Vag = geo_ageo(dat_r)
    w = np.cos(np.deg2rad(Vg.lat))
    profile_Vg = Vg.weighted(w).mean(["lon", "lat"])
    profile_Vag = Vag.weighted(w).mean(["lon", "lat"])
    return time_index, step_index, profile_Vg, profile_Vag

basepath = Path("/p/scratch/weatherai/shared/weather_generator_data/")
outpath = Path("./data/")
outpath.mkdir(exist_ok=True, parents=True)
ranks = [0,1]
selsteps = [24, 120, 240]
selsteps = [ np.timedelta64(s, "h") for s in selsteps ]
n_jobs = 3
nst = len(selsteps)

for rank in ranks:
    infile = basepath / f"prediction_pl_nh0_ja7f_rank{rank:04}.grib" #"/p/scratch/weatherai/shared/weather_generator_data/.grib"
    outfile = outpath / f"geostrophic_wind_profile_rank{rank:04}.nc"
    if outfile.exists():
        print(f"result already present at {outfile}")
        continue
    
    dat = xr.open_dataset(infile)[["u","v","z"]]
    dat = dat.sel(step=selsteps)
    times = dat.time
    nt = len(times)
    nlev = dat.isobaricInhPa.size

    # apply_ufunc was slow for this calculation, so parallelise each
    # independent time-step pair.
    all_Vg, all_Vag = np.zeros((nt, nst,nlev)), np.zeros((nt, nst,nlev))
    
    jobs = list(product(range(nt), range(nst)))
    results = Parallel(n_jobs=n_jobs, backend="threading")(
        delayed(compute_time_step)(dat, i, j)
        for i, j in tqdm(jobs, desc=f"rank {rank}")
    )
    for i, j, profile_Vg, profile_Vag in results:
        all_Vg[i, j, :] = profile_Vg
        all_Vag[i, j, :] = profile_Vag
        
    # store the results as netcdf
    res = xr.Dataset(
        data_vars=dict(
            Vg=(["time","step","isobaricInhPa"], all_Vg),
            Vag=(["time","step","isobaricInhPa"], all_Vag),
        ),
        coords=dict(
            time=dat.time,
            step=dat.step,
            isobaricInhPa=dat.isobaricInhPa,
        )
    )
    res.to_netcdf(outfile)
