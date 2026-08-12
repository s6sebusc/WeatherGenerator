from pathlib import Path

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from weathergen.common.io import zarrio_reader
import matplotlib.colors as mcolors

regions = dict(
    Europe = (-10, 40, 30, 60),
    Maritime_continent = (50, 150, -30, 40),
    North_America_East = (-100, -60, 25, 55),
    North_America_West = (-130, -100, 30, 60),
    South_America_Brazil = (-70, -35, -35, 5),
    Australia = (112, 155, -44, -10),
    East_Asia = (100, 145, 20, 50),
    South_Asia_India = (65, 95, 5, 35),
    Africa_Sahel_Central = (-15, 45, -10, 20),
    Southern_Africa = (10, 40, -35, -10),
    Biparjoy= (55,75,0,20),
)

def plot_tp_forecast_vs_target(
    runid,
    sample,
    forecast_step,
    stream="IMERG_ANEMOI",
    regname="Europe",
):
    """
    Load one sample/forecast step from a zarr archive and plot target vs forecast
    for channel "tp" over the Northern Indian Ocean.
    """
    store_path = Path(f"/e/scratch/weatherai/shared_work/results/{runid}/validation_chkpt00000_rank0000.zip")
    region_extent = regions.get(regname)
    with zarrio_reader(store_path) as zio:
        item = zio.get_data(sample=sample, stream=stream, forecast_step=forecast_step)

    if item.target is None:
        raise ValueError(f"No target dataset for sample={sample}, step={forecast_step}")
    if item.prediction is None:
        raise ValueError(f"No prediction dataset for sample={sample}, step={forecast_step}")

    target_da = item.target.as_xarray()
    forecast_da = item.prediction.as_xarray()

    def extract_tp(da):
        tp = da.sel(channel="tp")
        tp = tp.squeeze(drop=True)
        lon = tp["lon"].values.ravel()
        lat = tp["lat"].values.ravel()
        values = tp.values.ravel()*1000
        return lon, lat, values

    lon_t, lat_t, tp_target = extract_tp(target_da)
    lon_f, lat_f, tp_forecast = extract_tp(forecast_da)

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(16, 7),
        subplot_kw={"projection": ccrs.PlateCarree()},
       # constrained_layout=True,
    )

    plot_args = [
        (axes[0], lon_t, lat_t, tp_target, f"Target tp sample={sample} step={forecast_step}"),
        (axes[1], lon_f, lat_f, tp_forecast, f"Forecast tp sample={sample} step={forecast_step}"),
    ]
    bounds = [0, .1, 1, 3, 5, 10, 20, 50, 100]

    # 2. Build the colormap and boundary norm
    cmap = plt.get_cmap("viridis", len(bounds) - 1)  # create discrete color steps
    norm = mcolors.BoundaryNorm(boundaries=bounds, ncolors=cmap.N)
    for ax, lon, lat, values, title in plot_args:
        sc = ax.scatter(
            lon,
            lat,
            c=values,
            cmap=cmap,
            norm=norm,
            transform=ccrs.PlateCarree(),
            linewidth=0,
        )
        ax.coastlines(resolution="110m", linewidth=1, color="white")
        ax.set_extent(region_extent, crs=ccrs.PlateCarree())
        ax.set_title(title)
        ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)

    cbar = fig.colorbar(sc, ax=axes, orientation="horizontal", pad=0.05)
    cbar.set_label("tp")
    fig.suptitle(f"valid time: {target_da.valid_time.values[0]} init: {forecast_da.source_interval_end.values[0]}")

    fig.savefig(f"plots/tp_{runid}_{regname}_sample{sample}_fstep{forecast_step}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    return 0

#plot_tp_forecast_vs_target("z71y2ik8", sample=9, forecast_step=10, regname="Europe")
#plot_tp_forecast_vs_target("yz3h0kyn", sample=9, forecast_step=10, regname="Europe")
plot_tp_forecast_vs_target("z71y2ik8", sample=5, forecast_step=20, regname="Maritime_continent")
plot_tp_forecast_vs_target("yz3h0kyn", sample=5, forecast_step=20, regname="Maritime_continent")
#plot_tp_forecast_vs_target("z71y2ik8", sample=5, forecast_step=20, regname="Biparjoy")
#plot_tp_forecast_vs_target("yz3h0kyn", sample=5, forecast_step=20, regname="Biparjoy")
