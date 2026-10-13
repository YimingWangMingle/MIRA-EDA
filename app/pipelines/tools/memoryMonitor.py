import os
from time import sleep

import psutil


class MemoryMonitor:
    def __init__(self):
        self.keep_measuring = True
        self.process = psutil.Process(os.getpid())

    def measure_usage(self):
        max_usage = 0
        while self.keep_measuring:
            max_usage = max(
                max_usage,
                self.process.memory_info().rss
            )
            sleep(0.1)

        return max_usage
