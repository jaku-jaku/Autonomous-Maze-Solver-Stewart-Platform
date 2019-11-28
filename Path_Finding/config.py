# sys config
ENABLE_DEBUG  = 1
ENABLE_DPRINT = 1
ENABLE_EPRINT = 1
ENABLE_SPRINT = 1
ANIMATION_FPS = 60

# Heat map factors
GRADIENT_FACTOR = 8
MAX_HEAT_MAP_VALUE = 210 
MAX_HEAT_MAP_POWER = 1
MAX_HEAT_MAP_WEIGHT = 50

# MAP Config
GRID_SIZE_PERCENT = 0.018


# Feature
FEATURE_TARGET =[   { 'tag': 'blue_mark', 'lower':[41,0,93], 'upper':[106,255,150], 'minArea':1000, 'maskSize':50 },
                    { 'tag': 'red_mark','lower':[2,0,0],  'upper':[14, 255, 255], 'minArea':1000, 'maskSize':60 },
                    { 'tag': 'ball', 'lower':[0,0,230], 'upper':[255,255,255], 'minArea':1000, 'maskSize':60 },
                    { 'tag': 'green_mark', 'lower': [32, 32, 35], 'upper': [73, 160, 161], 'minArea': 1000, 'maskSize': 55 }
                ]

# INTERFACE Config
COUNT_DELAY = 0.1 #s
COUNT_DELAY_REPEAT = 0.01 #0.01 #s


#
COLOR_STRIP = [(192, 141, 240), (136, 230, 64), (64, 230, 213), (64, 133, 230), (230, 64, 161), (177, 230, 64)]