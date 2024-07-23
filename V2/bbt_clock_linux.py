# -*- coding: utf-8 -*-

import sys
import time

class Clock:
    """A class to get standard bbt timestamps for linux"""

    def __init__(self):
        pass

    def timestamp(self):
        return int(time.monotonic_ns()/1000)
