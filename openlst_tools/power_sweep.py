from openlst_tools.openlst import OpenLst

import numpy as np
import time

class Sweeper:
    def __init__(self, port, hwid):
        self.lst = OpenLst(port, hwid, quiet=True)

        self.lst.open()
        self.lst.clean_packets()
        self.lst.reboot()
        time.sleep(1)

    def __del__(self):
        self.lst.close()
        del self.lst
    
    def set_freq(self, freq: float) -> float:
        # Use max pwoer
        ret = self.lst.set_rf_params(frequency=freq, power=0xC0)

        return ret[0]

    def sweep_range(self, start=400e6, stop=450e6, step=10e3, timeout=0.2):
        freqs = np.arange(start, stop, step)

        for f in freqs:
            self.set_freq(f)

            start = time.time()

            self.lst.transmit(b"0"*83)
            time.sleep(0.2)


def main():
    sweep = Sweeper("/dev/ttyUSB1", 0x0171)

    sweep.sweep_range(start=425e6, stop=450e6, step=15e3)

if __name__ == "__main__":
    main()
