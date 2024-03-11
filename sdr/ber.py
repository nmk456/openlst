#!/usr/bin/env python

import sdr
from openlst_tools.openlst import OpenLst

import matplotlib.pyplot as plt
import numpy as np

import time


class BerTester:
    def __init__(self, port, hwid=0x71):
        self.sdr = sdr.SDR()
        self.lst = OpenLst(port=port, hwid=hwid, rtscts=True, quiet=True)

        self.lst.open()

    def __del__(self):
        del self.sdr
        del self.lst

    def packet_error_rate(self, count, gain, size=255, rate=7416):
        # Returns PER

        assert count % 10 == 0, "Count must be a multiple of 10"

        _, bw, drate, dev = self.lst.set_rf_params(drate=rate, deviation=rate/2, chan_bw=2*rate/0.8)

        print(f"Using datarate={int(drate)}, deviation={int(dev)}, bandwidth={int(bw)}")

        # Use actual symbol rate
        self.sdr.set_symbol_rate(drate)
        self.sdr.set_gain(gain)

        self.lst.reboot()

        # Wait for 15 packets worth of time to make uptime valid
        time.sleep(max(size*8*15/(rate/2), 1))

        self.lst.set_rf_params(drate=rate, deviation=rate/2, chan_bw=2*rate/0.8)

        last_uptime = self.lst.get_telem()["uptime"]
        last_packets = 0
        good_packets = 0

        i = 0

        while i < count:
            # print("Transmitting")
            for _ in range(10):
                self.sdr.transmit(0x11, b'0'*(size-10))

            # Wait for packet to finish transmitting
            time.sleep(1.5*size*8/(rate/2))

            telem = self.lst.get_telem()

            if telem["uptime"] >= last_uptime:
                i += 10
                new_packets = telem["packets_good"] - last_packets
                good_packets += new_packets
            else:
                print("Uptime decreased, unexpected reboot")

            last_uptime = telem["uptime"]
            last_packets = telem["packets_good"]
            self.lst.clean_packets()

        time.sleep(1)

        return 1 - good_packets / count

def vary_gain(hwid=0x00, num=4000, min=20, max=80, step=10, size=255):
    tester = BerTester('/dev/serial/by-id/usb-FTDI_FT232R_USB_UART_B001JRNT-if00-port0')

    gains = np.arange(min, max+step, step)
    pers = np.zeros(len(gains), dtype=np.float32)

    for i in range(len(gains)):
        per = tester.packet_error_rate(num, gains[i], size)

        pers[i] = per

        print(f"G={gains[i]}, PER={per}")

    final_data = np.array([gains, per]).T
    np.savetxt(f'gain_vs_received_n={num}_min={min}_max={max}.csv', final_data, delimiter=',')

    plt.semilogy(gains, per)
    plt.xlabel("SDR Gain (dB)")
    plt.ylabel("PER")
    plt.title("Packet Error Rate")

    plt.savefig(f'gain_vs_received_n={num}_min={min}_max={max}.png')

if __name__ == "__main__":
    vary_gain(num=1000, min=20, max=50, step=5)
