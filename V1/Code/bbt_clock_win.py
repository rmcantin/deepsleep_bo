# -*- coding: utf-8 -*-

import sys
import time
import numpy
from ctypes import windll, c_longlong, byref, Structure

class Clock:
    """A class to get standard bbt timestamps for windows"""
    class LARGE_INTEGER(Structure):
        _fields_ = [("QuadPart", c_longlong)]

    query_performance_frequency = windll.kernel32.QueryPerformanceFrequency
    query_performance_counter = windll.kernel32.QueryPerformanceCounter

    def __init__(self):
        self.frequency = self.LARGE_INTEGER(0)
        Clock.query_performance_frequency(byref(self.frequency))

    def timestamp(self):
        precision_time = self.LARGE_INTEGER(0)
        Clock.query_performance_counter(byref(precision_time))
        return int(precision_time.QuadPart*1000000/self.frequency.QuadPart)
