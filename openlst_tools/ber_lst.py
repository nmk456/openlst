#!/usr/bin/env python

from openlst_tools.openlst import OpenLst

import matplotlib.pyplot as plt
import numpy as np

import time


class BerTester:
    def __init__(self, port_t, port_r, hwid_t=0x71, hwid_r=0x72):
        print('receive:', port_r, hex(hwid_r))
        print('transmit:', port_t, hex(hwid_t))
        self.lst_r = OpenLst(port=port_r, hwid=hwid_r, rtscts=False, quiet=True)
        self.hwid_r = hwid_r
        print(self.lst_r.ser.is_open, self.lst_r.ser.port)

        self.lst_r.open()
        self.lst_r.clean_packets()
        self.lst_r.reboot()
        
        print(hex(self.lst_r.hwid))

        time.sleep(1)

        self.lst_t = OpenLst(port=port_t, hwid=hwid_t, rtscts=True, quiet=True)
        self.hwid_t = hwid_t
        print(self.lst_t.ser.is_open, self.lst_t.ser.port)
        
        print(hex(self.lst_t.hwid))

        self.lst_t.open()
        self.lst_t.clean_packets()
        self.lst_t.reboot()
        time.sleep(1)

    def __del__(self):
        del self.lst_r
        del self.lst_t

    def reboot(self, size, receive=True, transmit=True):
        if receive:
            self.lst_r.reboot()
        
        if transmit:
            self.lst_t.reboot()

        # Wait for 15 packets worth of time to make uptime valid
        # time.sleep(max(size*8*15/(self.drate/2), 1))
        time.sleep(1.5)
        # lst.set_rf_params(drate=self.drate, deviation=self.dev, chan_bw=self.bw)

    def set_params(self, rate, gain):
        _, self.bw, self.drate, self.dev = self.lst_r.set_rf_params(drate=rate)
        self.lst_t.set_rf_params(drate=rate, power=gain)

        # Bypass PA on TX LST
        self.lst_t.set_bypass(True)

    def packet_error_rate(self, count, gain, size=255, rate=7416):
        # Returns PER

        PACKET_STEP = 10
        assert count % PACKET_STEP == 0, "Count must be a multiple of 10"

        self.reboot(size)
        self.set_params(rate, gain)

        print(f"Using datarate={int(self.drate)}, deviation={int(self.dev)}, bandwidth={int(self.bw)}")

        last_uptime = self.lst_r.get_telem()["uptime"]
        last_packets = 0
        good_packets = 0

        i = 0

        while i < count:
            for _ in range(PACKET_STEP):
                # If we send it to the HWID of the OpenLST, it will just drop
                # the message instead of forwarding it. This might help reduce
                # errors.
                self.lst_t.transmit(b'0'*(size-10), self.hwid_r)
                time.sleep(5*size*8*2/self.drate)

            # Wait for packet to finish transmitting
            time.sleep(10)

            try:
                telem = self.lst_r.get_telem()
            except (AssertionError, TimeoutError):
                telem = None

            print(telem["packets_good"], telem["cs_count"], telem["sfd_count"], telem["rssi_dbm"])

            if telem is not None and telem["uptime"] >= last_uptime:
                i += PACKET_STEP
                new_packets = telem["packets_good"] - last_packets
                good_packets += new_packets

                last_uptime = telem["uptime"]
                last_packets = telem["packets_good"]
            else:
                print("Uptime decreased, unexpected reboot")
                self.reboot(size)
                self.set_params(rate, gain)

                last_uptime = 0
                last_packets = 0

            self.lst_r.clean_packets()

        time.sleep(1)

        return 1 - good_packets / count

# table 72 pg 207
powers = [-30, -20, -15, -10, -5, 0, 5, 7, 10]
settings = [0x12, 0x0E, 0x1D, 0x34, 0x2C, 0x60, 0x84, 0xC8, 0xC0]

def vary_gain(port_r, port_t, hwid_r=0x00, hwid_t=0x01, num=4000, min=-30, max=10, size=255, rate=7416):
    tester = BerTester( port_r=port_r, port_t=port_t, hwid_r=hwid_r, hwid_t=hwid_t)

    gains_s = [settings[powers.index(p)] for p in powers if min <= p <= max]
    gains = [p for p in powers if min <= p <= max]

    pers = np.zeros(len(gains_s), dtype=np.float32)

    for i in range(len(gains)):
        per = tester.packet_error_rate(num, gains_s[i], size, rate=rate)

        pers[i] = per

        print(f"G={gains[i]}, PER={per}")

    final_data = np.array([gains, pers]).T
    # np.savetxt(f'gain_vs_received_n={num}_min={min}_max={max}_r={int(rate)}.csv', final_data, delimiter=',')

    # plt.semilogy(gains, pers)
    # plt.xlabel("SDR Gain (dB)")
    # plt.ylabel("PER")
    # plt.title(f"Packet Error Rate N={num}, R={int(rate)}, S={size}")

    # plt.savefig(f'gain_vs_received_n={num}_min={min}_max={max}_r={int(rate)}.png')

if __name__ == "__main__":
    vary_gain(port_t='/dev/ttyUSB0', port_r='/dev/ttyUSB1', hwid_t=0x0071, hwid_r=0x0171, num=50, min=-20, max=-10, size=20, rate=38.4e3)
