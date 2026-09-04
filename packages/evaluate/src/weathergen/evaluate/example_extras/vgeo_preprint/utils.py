import numpy as np
from scipy.interpolate import interp1d
import xarray as xr
from functools import cached_property

def regularise_dataset(ds: xr.Dataset, gridpointdim: str = "ncells") -> xr.Dataset:
    """
    Tries to guess the grid and interpolate a dataset from reduced to regular gaussian.
    """
    is_dask = (len(ds.chunks) > 0)
    n_points = ds[gridpointdim].size
    if "longitude" in ds.coords:
        lonname = "longitude"
        latname = "latitude"
    else:
        lonname = "lon"
        latname = "lat"
    if n_points == 542080:
        grid = reduced_grid(320, ds[lonname].values, ds[latname].values, mode="N")
    elif n_points == 40320:
        grid = reduced_grid(96, ds[lonname].values, ds[latname].values, mode="O")
    else:
        raise NotImplementedError(f"grid size {n_points} fits neither N320 nor O96")
    reslon, reslat =  grid.regular_lons, grid.regular_lats
    res = xr.apply_ufunc(
        lambda x: grid.reduced2regular(x),
        ds,
        input_core_dims=[[gridpointdim]],
        output_core_dims=[["lat", "lon"]],
        dask_gufunc_kwargs = dict(
            allow_rechunk=True,
            output_sizes = dict(lon=len(reslon), lat=len(reslat))
        ),
        vectorize=True,
        dask="parallelized",
        output_dtypes= float if is_dask else None
    )
    res["lon"] = reslon
    res["lat"] = reslat
    return res
    
class reduced_grid():
    """
    Handling of grid information and interpolation for reduced Gaussian grid. 

    Attributes:
        N (int): half the number of unique latitudes
        lon, lat (np.ndarray): vector of latitudes and longitudes for each gridpoint
        lat (np.ndarray): vector of latitudes for each gridpoint
        mode (str): either "O" for octahedral or "N" for the oldschool reduced grid
        n_points (int): number of gridpoints
        idx (np.ndarray): indexes that sort a vector of gridpoint values by latitude and longitude
        regular_lats, regular_lons (np.ndarray): vectors of lats and lons in the regular Gaussian grid. 
        n_lon (list of ints): list with the number of longitudes at each latitude
    
    """
    
    def __init__(self, N : int, lon: np.ndarray, lat: np.ndarray, mode: str = "N"):
        self.N   = N
        self.lon = lon
        self.lat = lat
        self.mode = mode
        self.n_points = len(self.lon)
        self.idx = np.lexsort((self.lon, self.lat))
        self.regular_lats = sorted(set(self.lat))
        self.regular_lons = np.arange(0,360,360/(4*N))
        self.n_lon = self.get_n_lon()
        assert np.sum(self.n_lon) == self.n_points
            
    def get_n_lon(self) -> list[int]:
        """
        Number of longitudes per latitude circle.
        """
        if self.mode == "N":
            n =  [  int((self.lat == l).sum()) 
                    for l in self.regular_lats ]
        elif self.mode == "O":
            n = []
            for i in range(2*self.N):
                index_from_pole = min(i+1,2*self.N-i)
                n.append( 4*index_from_pole+16 )
        else:
            raise ValueError(f"unknown mode {self.mode}, use O or N.")
        return n

    def split_data(self, data: np.ndarray) -> list[np.ndarray]:
        """
        Takes a 1D array and splits it into a list where each entry
        contains the data for one latitude circle. 
        """
        assert len(data) == self.n_points
        k=0
        res = []
        for n in self.n_lon:
            res.append(data[self.idx[k:k+n]])
            k+=n
        assert k == self.n_points
        return res
    
    def reduced2regular(self, data: np.ndarray) -> np.ndarray:
        """
        Interpolate a 1D np.array from reduced to regular Gaussian. 
        """
        res = np.zeros((2*self.N, 4*self.N))
        for i,values in enumerate(self.split_data(data)):
            lon = np.arange(0,360,360/self.n_lon[i])
            res[i,:] = np.interp(self.regular_lons,lon,values) 
        return res