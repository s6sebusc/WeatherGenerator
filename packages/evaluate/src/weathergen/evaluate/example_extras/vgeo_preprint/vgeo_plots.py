# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "dask>=2026.3.0",
#     "matplotlib>=3.10.9",
#     "netcdf4>=1.7.4",
#     "numpy>=2.4.4",
#     "xarray>=2026.4.0",
# ]
# ///
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

dat = xr.open_mfdataset("data/*.nc")

ratio = dat.Vag.mean("time") / dat.Vg.mean("time")
ratio["step"] = ratio["step"] / np.timedelta64(1, 'h')

ratio.sel(step=[24,120,240]).plot(y="isobaricInhPa", hue="step")
plt.gca().invert_yaxis()
plt.savefig("vgeo_balance.png")
plt.close()