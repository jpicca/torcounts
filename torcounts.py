from processor import TorProbSim

import pathlib
import argparse
import json

import numpy as np
from scipy import interpolate as I
from shapely.geometry import shape

import pygrib as pg

from pygridder import pygridder as pgrid
import pyproj

import os


### CLI Parser ###
forecast_file_help = "The tornado coverage probabilities grib file"
geo_file_help = "The conditional intensity geojson file"
out_path_help = "Absolute path to the directory for writing the images"
ndfd_file_help = "NPZ including grid lat/lons and projection string"

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--forecastfile", required=True, help=forecast_file_help)
parser.add_argument("-gf", "--geofile", required=True, help=geo_file_help)
parser.add_argument("-o", "--outpath", required=False, default=os.getcwd(), help=out_path_help)
parser.add_argument("-n", "--ndfdfile", required=False, default=pathlib.Path('./assets/ndfd.npz'), help=ndfd_file_help)

args = parser.parse_args()


forecast_file = pathlib.Path(args.forecastfile)
geo_file = pathlib.Path(args.geofile)
out_file = pathlib.Path(args.outpath)
ndfd_file = pathlib.Path(args.ndfdfile)

# Function to read forecast grib file
def read_ndfd_grib_file(grbfile):
    """ Read an SPC Outlook NDFD Grib2 File """
    with pg.open(grbfile.as_posix()) as GRB:
        try:
            vals = GRB[1].values.filled(-1)
        except AttributeError:
            vals = GRB[1].values
    return vals

# Get tornado coverage prob grid
torn = read_ndfd_grib_file(forecast_file)

# Get NDFD lat / lons and proj string
with np.load(ndfd_file.as_posix()) as NPZ:
    X = NPZ["X"]
    Y = NPZ["Y"]
    ndfd_lons = NPZ["lons"]
    ndfd_lats = NPZ["lats"]
    proj = pyproj.Proj(NPZ["srs"].item())

# Make gridder object
G = pgrid.Gridder(X, Y, dx=4000)

# Read geojson and create CIG grid
with open(geo_file) as f:
    geo_data = json.load(f)

sig = G.make_empty_grid().astype(float)

lons = []
lats = []
cats = []

for feature in geo_data['features']:
    
    geometry = shape(feature['geometry'])

    try:
        lon,lat = geometry.exterior.xy
        lon,lat = proj(lon,lat)
        lons.append(lon)
        lats.append(lat)
        cats.append(feature['properties']['LABEL'])
    except AttributeError:
        for g in geometry.geoms:
            lon = g.exterior.xy[0]
            lat = g.exterior.xy[1]
            lon, lat = proj(lon, lat)
            lons.append(lon)
            lats.append(lat)
            cats.append(feature['properties']['LABEL'])
            
polys = G.grid_polygons(lons, lats)

for poly, cat in zip(polys, cats):

    if cat == 'SIGN':
        sig[poly] = 0.1
    elif cat == 'SIGD':
        sig[poly] = 0.2
    elif cat == 'SIGT':
        sig[poly] = 0.3

sig = sig.astype(float)*100

# Check torn and sig grids for 0s / inconsistencies
if not np.count_nonzero(torn):

    # If there are CIG contours
    if np.count_nonzero(sig):
        print('''
        **Error Detected** 
        Tornado coverage probabilities less than 2%, but conditional intensity contours provided
        ''')

    # Else if there are no CIG contours
    else:
        print(f'Coverage probabilities less than 2%. No tornado count ranges generated.')

    # Exit script
    import sys
    sys.exit(0)

# Otherwise, create torprobsim object and counts
counter = TorProbSim(torn,sig,ndfd_lons,ndfd_lats)
counter.calcCounts(out_file,graphic=True)
