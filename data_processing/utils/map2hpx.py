import sys 
import argparse
import sys
import os
import numpy as np
import xarray as xr
import healpy as hp
from tqdm import tqdm
import multiprocessing
import matplotlib.pyplot as plt

from data_processing.remap.healpix import HEALPixRemap
from data_processing.remap.cubesphere import to_chunked_dataset

EXAMPLE_PARAMS = {
    'file_name' : '/home/quicksilver2/nacc/Data/pipeline_dev/era5_1950-2022_3h_1deg_sst-ti.nc',
    'target_variable_name' : 'sst', # this is used to extract the DataArray from the Dataset
    'file_variable_name' : 'sst', # this is how the variable will be saved in the new file 
    # this is the "prefix" for the output file. Should include the desired path to the output file
    'prefix' : '/home/quicksilver2/nacc/Data/pipeline_dev/era5_1deg_3h_HPX32_1950-2022_',
    'nside' : 32,
    'order' : 'bilinear', # order of the interpolation
    'resolution_factor' : 1.0,
    'visualize':False, # This determines whether to visualize remap. Warning: buggy
}
EXAMPLE_PARAMS_CONSTANT = {
    'file_name' : '/home/quicksilver2/nacc/Data/pipeline_dev/era5_1950-2022_3h_1deg_lsm.nc',
    'target_variable_name' : 'lsm', 
    'file_variable_name' : 'lsm', 
    'prefix' : '/home/quicksilver2/nacc/Data/pipeline_dev/era5_1deg_3h_HPX32_1950-2022_',
    'nside' : 32,
    'order' : 'bilinear', 
    'resolution_factor' : 1.0,
    'visualize':False,
}

def main(params):
    
    print('checking parameters...')
    if not os.path.isfile(params['file_name']):
        print(f'source file ~{params["file_name"]}~ not found. Aborting.')
        return
    if os.path.isfile(params['prefix']+params['file_variable_name']+'.nc'):
        print(f'target file ~{params["prefix"]+params["file_variable_name"]+".nc"}~ already exists. Aborting.')
        return
    # create namespace object for quick attribute referencing
    args = argparse.Namespace(**params)
 
    # save for later 
    nside = args.nside
    prefix = args.prefix

    assert f"HPX{nside}" in prefix, (f"'HPX{nside}' could not be found in the prefix '{prefix}'. Please make sure "
                                     f"that the nside={nside} parameter and the prefix provided to this function "
                                     f"match.")

    # Load .nc file in latlon format to extract latlon information and to initialize the remapper module
    ds_ll = xr.open_dataset(args.file_name).rename({"time": "sample"}).squeeze()
    latitudes, longitudes = ds_ll.dims["latitude"], ds_ll.dims["longitude"]
    mapper = HEALPixRemap(
        latitudes=latitudes,
        longitudes=longitudes,
        nside=nside,
        resolution_factor=args.resolution_factor,
        order=args.order
        )

    # Determine whether a "constant" or "variable" is processed
    const = False if "sample" in list(ds_ll.dims.keys()) else True
    vname = list(ds_ll.keys())[0]

    # Set up coordinates and chunksizes for the HEALPix dataset
    coords = {}
    if not const:
        coords["sample"] = ds_ll.coords["sample"]
    coords["face"] = np.array(range(12), dtype=np.int64)
    coords["height"] = np.array(range(nside), dtype=np.int64)
    coords["width"] = np.array(range(nside), dtype=np.int64)

    # Map the "constant" or "variable" to HEALPix
    if const:
        data_hpx = mapper.ll2hpx(ds_ll.variables[vname].values, visualize=args.visualize)
        ds_mean = ds_ll.variables[vname].mean()
        ds_std = ds_ll.variables[vname].std()
    else:
        dims = []
        for coord in coords: 
            dims.append(len(coords[coord]))# if coord!='varlev' else 1)

        # Sequential sample mapping via for-loop
        # Allocate a (huge) array to store all samples (time steps) of the projected data
        data_hpx = np.zeros(dims, dtype=ds_ll.variables[vname])
        # Iterate over all samples and levels, project them to HEALPix and store them in the predictors array
        pbar = tqdm(ds_ll.coords["sample"])
        for s_idx, sample in enumerate(pbar):
            pbar.set_description("Remapping time steps")
            data_hpx[s_idx] = mapper.ll2hpx(data=ds_ll.variables[vname][s_idx].values,
                                               visualize=args.visualize)
    # Determine latitude and longitude values for the HEALPix faces
    hpxlon, hpxlat = hp.pix2ang(mapper.nside, range(mapper.npix), nest=True, lonlat=True)
    data_lat = mapper.hpx1d2hpx3d(hpx1d=hpxlat, dtype=np.float32)
    data_lon = mapper.hpx1d2hpx3d(hpx1d=hpxlon, dtype=np.float32)

    # Build HEALPix dataset and write it to file
    ds_hpx = xr.Dataset(
        coords=coords,
        data_vars={
            "lat": (["face", "height", "width"], data_lat),
            "lon": (["face", "height", "width"], data_lon),
            args.target_variable_name: (list(coords.keys()), data_hpx),
            },
        attrs=ds_ll.attrs
        )
    # define chunksizes for dataarray 
    chunksizes = {}
    for coord in coords: 
        chunksizes[coord] = len(coords[coord]) if coord in ['face','height','width'] else 1
    ds_hpx = to_chunked_dataset(ds=ds_hpx, chunking=chunksizes)
    print(f"Dataset sucessfully built. Writing data to file {prefix + args.file_variable_name + '.nc'}...")
    ds_hpx.to_netcdf(prefix + args.file_variable_name + ".nc")
    ds_reload = xr.open_dataset(prefix + args.file_variable_name + ".nc")
    print(ds_reload)

if __name__ == "__main__":
     
    main(EXAMPLE_PARAMS)
    main(EXAMPLE_PARAMS_CONSTANT)
