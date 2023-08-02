from processor import TorProbSim

import pathlib
import argparse

import numpy as np
from scipy import interpolate as I

import pygrib as pg

import pygridder as pgrid
import pyproj


### CLI Parser ###
forecast_file_help = "The tornado coverage probabilities grib file"
geo_file_help = "The conditional intensity geojson file"
ndfd_file_help = "NPZ including grid lat/lons and projection string"

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--forecastfile", required=True, help=forecast_file_help)
parser.add_argument("-gf", "--geofile", required=True, help=geo_file_help)
parser.add_argument("-n", "--ndfdfile", required=False, default=pathlib.Path('./assets/ndfd.npz'), help=ndfd_file_help)

args = parser.parse_args()


forecast_file = pathlib.Path(args.forecastfile)
geo_file = pathlib.Path(args.geofile)
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
    proj = pyproj.Proj(NPZ["srs"].item())


# Read geojson and grid on ndfd grid


# Create torprobsim object


