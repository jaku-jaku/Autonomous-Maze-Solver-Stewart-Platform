# sys config
ENABLE_DEBUG  = 1
ENABLE_DPRINT = 1
ENABLE_EPRINT = 1
ENABLE_SPRINT = 1
ANIMATION_FPS = 60

# Heat map factors
GRADIENT_FACTOR = 6
MAX_HEAT_MAP_VALUE = 210 
MAX_HEAT_MAP_POWER = 3
MAX_HEAT_MAP_WEIGHT = 50

# MAP Config
GRID_SIZE_PERCENT = 0.018

# Feature
FEATURE_TARGET =[   { 'tag': 'blue_mark', 'lower':[89,0,0], 'upper':[131,255,158], 'minArea':1000, 'maskSize':50 },
                    { 'tag': 'red_mark','lower':[2,0,0],  'upper':[14, 255, 255], 'minArea':1000, 'maskSize':55 },
                    { 'tag': 'ball', 'lower':[0,0,230], 'upper':[255,255,255], 'minArea':1000, 'maskSize':58 },
                    { 'tag': 'green_mark', 'lower': [32, 32, 35], 'upper': [73, 160, 161], 'minArea': 1000, 'maskSize': 53 }
                ]

# INTERFACE Config
COUNT_DELAY = 0.08 #s		
COUNT_DELAY_REPEAT = 0.015 #0.01 #s
END_OF_CMD_DELAY = 2 #s

#
COLOR_STRIP = [(192, 141, 240), (136, 230, 64), (64, 230, 213), (64, 133, 230), (230, 64, 161), (177, 230, 64)]