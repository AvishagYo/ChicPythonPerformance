import time

from timeitt import timeitt


@timeitt
def slow_function():
    time.sleep(1)

slow_function()