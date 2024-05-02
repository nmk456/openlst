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
        self.hwid = hwid

        self.lst.open()

        self.lst.clean_packets()
        self.lst.reboot()
        time.sleep(1)

    def __del__(self):
        del self.sdr
        del self.lst

    def reboot(self, size):
        self.lst.reboot()

        # Wait for 15 packets worth of time to make uptime valid
        time.sleep(max(size*8*15/(self.drate/2), 1))

        self.lst.set_rf_params(drate=self.drate, deviation=self.dev, chan_bw=self.bw)

    def packet_error_rate(self, count, gain, size=255, rate=7416):
        # Returns PER

        PACKET_STEP = 10
        assert count % PACKET_STEP == 0, "Count must be a multiple of 10"

        _, self.bw, self.drate, self.dev = self.lst.set_rf_params(drate=rate, deviation=rate/2, chan_bw=2*rate/0.8)

        print(f"Using datarate={int(self.drate)}, deviation={int(self.dev)}, bandwidth={int(self.bw)}")

        # Use actual symbol rate
        self.sdr.set_symbol_rate(self.drate)
        self.sdr.set_gain(gain)

        self.reboot(size)

        last_uptime = self.lst.get_telem()["uptime"]
        last_packets = 0
        good_packets = 0

        i = 0

        while i < count:
            # print("Transmitting")
            for _ in range(PACKET_STEP):
                # If we send it to the HWID of the OpenLST, it will just drop
                # the message instead of forwarding it. This might help reduce
                # errors.
                self.sdr.transmit(0x11, b'0'*(size-10), dest_hwid=self.hwid)

            # Wait for packet to finish transmitting
            time.sleep(2)

            try:
                telem = self.lst.get_telem()
            except (AssertionError, TimeoutError):
                telem = None

            if telem is not None and telem["uptime"] >= last_uptime:
                i += PACKET_STEP
                new_packets = telem["packets_good"] - last_packets
                good_packets += new_packets

                last_uptime = telem["uptime"]
                last_packets = telem["packets_good"]
            else:
                print("Uptime decreased, unexpected reboot")
                self.reboot(size)

                last_uptime = 0
                last_packets = 0

            self.lst.clean_packets()

        time.sleep(1)

        return 1 - good_packets / count

def vary_gain(hwid=0x00, num=4000, min=20, max=80, step=10, size=255, rate=7416):
    tester = BerTester('/dev/ttyUSB1', hwid=hwid)

    gains = np.arange(min, max+step, step)
    pers = np.zeros(len(gains), dtype=np.float32)

    for i in range(len(gains)):
        per = tester.packet_error_rate(num, gains[i], size, rate=rate)

        pers[i] = per

        print(f"G={gains[i]}, PER={per}")

    final_data = np.array([gains, pers]).T
    np.savetxt(f'gain_vs_received_n={num}_min={min}_max={max}_r={int(rate)}.csv', final_data, delimiter=',')

    plt.semilogy(gains, pers)
    plt.xlabel("SDR Gain (dB)")
    plt.ylabel("PER")
    plt.title(f"Packet Error Rate N={num}, R={int(rate)}, S={size}")

    plt.savefig(f'gain_vs_received_n={num}_min={min}_max={max}_r={int(rate)}.png')

if __name__ == "__main__":
    vary_gain(hwid=0x0171, num=1000, min=30, max=70, step=10, size=20, rate=38.4e3)

    # tester = BerTester('/dev/serial/by-id/usb-FTDI_FT232R_USB_UART_B001JRNT-if00-port0')
    # print(tester.packet_error_rate(200, gain=50))

    # time.sleep(3)

    # print(tester.lst.get_telem()["packets_good"])
