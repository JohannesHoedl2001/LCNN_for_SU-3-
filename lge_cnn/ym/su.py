"""
Select suitable SU group.
"""
import os

su_group = os.environ.get('GAUGE_GROUP', 'su3').lower()

if su_group == 'su2':
    print("Using SU(2)")
    from lge_cnn.ym.su2 import *
    NC = 2
if su_group == 'su3':
    print("Using SU(3)")
    from lge_cnn.ym.su3 import *
    NC = 3
else:
    print("Unsupported gauge group: " + su_group)
    exit()
