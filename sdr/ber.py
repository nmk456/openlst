#!/usr/bin/env python

import sdr
from openlst_tools.openlst import OpenLst

import matplotlib.pyplot as plt
import numpy as np

import time

def vary_gain(hwid=0x00, num=4000, min=20, max=80, step=10):
    data = []
    sdr1 = sdr.SDR()
    lst = OpenLst(port='/dev/serial/by-id/usb-FTDI_FT232R_USB_UART_B001JRNT-if00-port0', hwid=0x0071, rtscts=True)
    lst.open()
    gain = list(range(min, max+step, step))
    for g in gain:
        lst.reboot()
        time.sleep(1)
        print(f'{g} gain')
        sdr1.set_gain(g)
        for i in range(num):
            sdr1.transmit(0x11, b'0'*245)
        time.sleep(1)
        telem = lst.get_telem()
        data.append(telem["packets_good"]/num)
        print(telem["packets_good"])
    x = np.array(gain)
    y = np.array(data)

    final_data = np.array([x,y]).T
    np.savetxt(f'gain_vs_received_{num}_{min}_{max}.csv', final_data, delimiter=',')

    plt.semilogy(x, y)
    plt.savefig(f'gain_vs_received_{num}_{min}_{max}.png')

    lst.close()

if __name__ == "__main__":
    vary_gain(num=4000, max=80)
    
