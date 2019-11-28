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
FEATURE_TARGET =[   { 'tag': 'blue_mark', 'lower':[103,0,0], 'upper':[138,255,117], 'minArea':1000, 'maskSize':50 },
                    { 'tag': 'red_mark','lower':[2,0,0],  'upper':[14, 255, 255], 'minArea':1000, 'maskSize':55 },
                    { 'tag': 'ball', 'lower':[0,0,230], 'upper':[255,255,255], 'minArea':1000, 'maskSize':58 },
                    { 'tag': 'green_mark', 'lower': [32, 32, 35], 'upper': [73, 160, 161], 'minArea': 1000, 'maskSize': 53 }
                ]

# INTERFACE Config
TRANSITION_DELAY = 0.3 #s
SAME_CMD_DELAY   = 0.1 #0.01 #s
SAME_CMD_LDELAY  = 0.05 #0.01 #s
SAME_CMD_DELAY_EFF_TICK = 4 #s
END_OF_CMD_DELAY = 1.5 #s

#
COLOR_STRIP = [(192, 141, 240), (136, 230, 64), (64, 230, 213), (64, 133, 230), (230, 64, 161), (177, 230, 64)]