import os
import geopandas as gpd

SVM_PARAMS = {
    'kernel': [
               'linear', 
               # 'poly', 
               'rbf', 
               'sigmoid', 
               # 'precomputed'
               ],
    'gamma': ['scale', 
              'auto', 
              # 0.5, 
              # 2.5, 
              # 5.0
              ],
    'nu': [
           # 0.01,
           0.15,
           0.16,
           # 0.17,
           # 0.18,
           0.19,
           0.20,
           0.25,
           # 0.5,
           # 0.75,
           # 0.99
           ]
}

YEARS = ['2018', '2019']
BASE_YEAR = '2017'
LIVING_YEAR = '2022'


# Phenology variables
PHEN_VARS = [
    'AMPL',
    'EOSD', 
    'EOSV', 
    'LENGTH', 
    'LSLOPE', 
    'MAXD', 
    'MAXV', 
    'MINV', 
    'RSLOPE', 
    'SOSD', 
    'SOSV',
    'SPROD',
    'TPROD'
]

TO_DROP = [
    'fid',
    'TYPE',
    'IDENT',
    'COMMENT',
    'DISPLAY',
    'SYMBOL',
    'UNUSED1',
    'DIST',
    'PROX_INDEX',
    'COLOR',
    'ALTITUDE',
    'DEPTH',
    'TEMP',
    'TIME',
    'WPT_CLASS',
    'SUB_CLASS',
    'ATTRIB',
    'LINK',
    'STATE',
    'COUNTRY',
    'CITY',
    'ADDRESS',
    'FACILITY',
    'CROSSROAD',
    'UNUSED2',
    'ETE',
    'DTYPE',
    'MODEL',
    'FILENAME',
    'LTIME',
    'OBJECTID_1'
]

# validation shape files -- living trees
VALIDATION_POINTS = ''

# validation geodataframe
VALIDATION_GDF = gpd.read_file(VALIDATION_POINTS)

# working directories
MAIN_DIR = ''
# change name to RASTER_DIR
WORKING_DIR = 'hrvpp-snd/'
SHAPE_DIR = 'shapes/'
RASTERS = os.listdir(MAIN_DIR+WORKING_DIR)
