import sys
import os

# Ensure python-ml/ is always on the path regardless of where pytest is run from
sys.path.insert(0, os.path.dirname(__file__))