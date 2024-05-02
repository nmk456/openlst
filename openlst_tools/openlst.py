#!/usr/bin/env python

import binascii
import logging
import numpy as np
import struct

from dataclasses import dataclass
from datetime import datetime

from openlst_tools.commands import OpenLstCmds, MAX_DATA_LEN
from openlst_tools.handler import LstHandler
from openlst_tools.utils import unpack_cint, pack_cint


J2000 = datetime(2000, 1, 1, 11, 58, 55, 816000)

SHELL_HEADER = """\
OpenLST shell

Commands can be accessed through the `openlst` object

Ex: `openlst.reboot()`"""


class OpenLst(LstHandler):
    def __init__(
        self,
        port: str,
        hwid: int,
        baud: int = 115200,
        rtscts: bool = False,
        timeout: float = 1,
        f_ref: float = 27e6,
        quiet: bool = False,
    ) -> None:
        """Object for communicating with OpenLST.

        Args:
            port (str): Serial port location. Looks like /dev/tty123 on Linux and COM123 on windows.
            hwid (int): HWID of connected OpenLST.
            baud (int, optional): Serial baud rate. Defaults to 115200.
            rtscts (bool, optional): Enable flow control with RTS/CTS. Defaults to False.
            timeout (int, optional): Command timeout in seconds. Defaults to 1.
        """

        self.f_ref = f_ref

        super().__init__(port, hwid, baud, rtscts, timeout, quiet)

    def __del__(self):
        self.thread.close()

    def receive(self) -> bytes:
        return self.get_packet(OpenLstCmds.ASCII)

    def transmit(self, msg: bytes, dest_hwid=0x0000):
        assert len(msg) <= MAX_DATA_LEN

        self._send(dest_hwid, OpenLstCmds.ASCII, msg)

    def reboot(self):
        self._send(self.hwid, OpenLstCmds.REBOOT)

    def bootloader_ping(self):
        seq = self._send(self.hwid, OpenLstCmds.BOOTLOADER_PING)

        return self.get_packet_timeout(seqnum=seq)

    def flash_erase(self):
        seq = self._send(self.hwid, OpenLstCmds.BOOTLOADER_ERASE)

        return self.get_packet_timeout(seqnum=seq)

    def flash_program_page(self, data: bytes, addr: int, check_resp: bool = True):
        assert len(data) == 128 or len(data) == 0

        msg = bytearray()
        msg.append(addr)
        msg.extend(data)

        seq = self._send(self.hwid, OpenLstCmds.BOOTLOADER_WRITE_PAGE, msg)
        resp = self.get_packet_timeout(seqnum=seq)

        if check_resp:
            assert resp is not None
            assert resp["command"] == OpenLstCmds.BOOTLOADER_ACK, hex(resp["command"])

        return resp

    def get_time(self):
        seq = self._send(self.hwid, OpenLstCmds.GET_TIME)
        resp = self.get_packet_timeout(seqnum=seq)

        assert resp is not None

        if resp["command"] == OpenLstCmds.NACK:
            return None

        t = {}

        t["s"] = unpack_cint(resp["data"][0:4], 4, False)
        t["ns"] = unpack_cint(resp["data"][4:8], 4, False)

        return t

    def set_time(self, seconds: int = None, nanoseconds: int = None):
        if seconds == None or nanoseconds == None:
            t = datetime.utcnow() - J2000
            seconds = int(t.total_seconds())
            nanoseconds = t.microseconds

        msg = bytearray()

        msg.extend(pack_cint(seconds, 4, False))
        msg.extend(pack_cint(nanoseconds, 4, False))

        seq = self._send(self.hwid, OpenLstCmds.SET_TIME, msg)
        resp = self.get_packet_timeout(seqnum=seq)

        assert resp is not None
        assert resp["command"] == OpenLstCmds.ACK

    def get_telem(self):
        seq = self._send(self.hwid, OpenLstCmds.GET_TELEM)
        resp = self.get_packet_timeout(seqnum=seq)

        assert resp is not None
        assert resp["command"] == OpenLstCmds.TELEM

        data = resp["data"]
        telem = {}

        telem["uptime"] = unpack_cint(data[1:5], 4, False)
        telem["uart0_rx_count"] = unpack_cint(data[5:9], 4, False)
        telem["uart1_rx_count"] = unpack_cint(data[9:13], 4, False)
        telem["rx_mode"] = unpack_cint(data[13:14], 1, False)
        telem["tx_mode"] = unpack_cint(data[14:15], 1, False)
        telem["adc"] = [unpack_cint(data[i : i + 2], 2, True) for i in range(15, 35, 2)]
        telem["last_rssi"] = unpack_cint(data[35:36], 1, False)
        telem["last_lqi"] = unpack_cint(data[36:37], 1, False)
        telem["last_freqest"] = unpack_cint(data[37:38], 1, True)
        telem["packets_sent"] = unpack_cint(data[38:42], 4, False)
        telem["cs_count"] = unpack_cint(data[42:46], 4, False)
        telem["packets_good"] = unpack_cint(data[46:50], 4, False)
        telem["packets_rejected_checksum"] = unpack_cint(data[50:54], 4, False)
        telem["packets_rejected_reserved"] = unpack_cint(data[54:58], 4, False)
        telem["packets_rejected_other"] = unpack_cint(data[58:62], 4, False)
        telem["reserved0"] = unpack_cint(data[62:66], 4, False)
        telem["reserved1"] = unpack_cint(data[66:70], 4, False)
        telem["custom0"] = unpack_cint(data[70:74], 4, False)
        telem["custom1"] = unpack_cint(data[74:78], 4, False)

        telem["sfd_count"] = telem["custom1"]

        def decode_rssi(rssi_val):
            rssi_offset = 74 # typical value for 433 MHz

            if rssi_val >= 128:
                return (rssi_val - 256) / 2 - rssi_offset
            else:
                return (rssi_val) / 2 - rssi_offset

        telem["rssi_dbm"] = decode_rssi(telem["last_rssi"])
        telem["rssi_cont_dbm"] = decode_rssi(telem["custom0"]) # Continuous RSSI

        return telem

    def set_rf_params(
        self,
        frequency: float = 437e6,
        drate: float = 7416,
        power: int = 0x12,
    ):
        """Sets CC1110 RF parameters.

        Output power setting currently controls the raw register value.
        Calibration is needed to correlate to actual output power.

        Args:
            frequency (float, optional): Carrier frequency (Hz). Defaults to 437e6.
            chan_bw (float, optional): Channel bandwidth (Hz). Defaults to 60268.
            drate (float, optional): Data rate (bps). Defaults to 7416.
            deviation (float, optional): FSK frequency deviation. Defaults to 3707.
            power (int, optional): Output power settings. Defaults to 0x12.

        Returns:
            Tuple of actual values of carrier frequency, channel bandwidth,
            data rate and deviation.
        """

        deviation = drate / 2
        chan_bw  = 2 * drate / 0.8

        # f_carrier = (f_ref / 2^16) * FREQ
        FREQ = int(2**16 * frequency / self.f_ref)
        f_actual = FREQ * self.f_ref / 2**16

        # Keep offset and IF as defaults
        FREQOFF = 0
        FREQ_IF = 6

        FSCTRL0 = FREQOFF
        FSCTRL1 = FREQ_IF

        # Channel bandwidth
        if chan_bw < 60268:
            chan_bw = 60268
        elif chan_bw > 843750:
            chan_bw = 843750

        # BW_channel = f_ref / (8 * 2^CHANBW_E * (4 + CHANBW_M))
        CHANBW_E = int(19 - np.floor(np.log2(chan_bw) + 0.25))
        CHANBW_M = int(np.round(self.f_ref / (chan_bw * 8 * 2**CHANBW_E) - 4))

        assert CHANBW_E >= 0 and CHANBW_E < 4, CHANBW_E
        assert CHANBW_M >= 0 and CHANBW_M < 4, CHANBW_M

        chanbw_actual = self.f_ref / (8 * (4 + CHANBW_M) * 2**CHANBW_E)

        # Deviation
        # f_dev = f_ref * 2^DEVIATN_E * (8 + DEVIATN_M) / 2^17
        DEVIATN_E = int(np.floor(np.log2(deviation * 2**14 / self.f_ref)))
        DEVIATN_M = int(np.round(deviation * 2**17 / (self.f_ref * 2**DEVIATN_E) - 8))

        assert DEVIATN_E >= 0 and DEVIATN_E < 8, DEVIATN_E
        assert DEVIATN_M >= 0 and DEVIATN_M < 8, DEVIATN_M

        dev_act = self.f_ref * 2**DEVIATN_E * (8 + DEVIATN_M) / 2**17

        drate = dev_act * 2

        # Data rate
        # R_DATA = f_ref * 2^DRATE_E * (256 + DRATE_M) / 2^28
        DRATE_E = int(np.floor(np.log2(drate * 2**20 / self.f_ref)))
        DRATE_M = int(np.round(drate * 2**28 / (self.f_ref * 2**DRATE_E) - 256))

        assert DRATE_E >= 0 and DRATE_E < 16, DRATE_E
        assert DRATE_M >= 0 and DRATE_M < 256, DRATE_M

        drate_actual = self.f_ref * 2**DRATE_E * (256 + DRATE_M) / 2**28

        msg = bytearray()
        msg.extend(pack_cint(FREQ, 4, False))
        msg.append(FSCTRL0)
        msg.append(FSCTRL1)
        msg.append(CHANBW_M | (CHANBW_E << 2))
        msg.append(DRATE_E)
        msg.append(DRATE_M)
        msg.append(DEVIATN_M | (DEVIATN_E << 4))
        msg.append(power & 0xFF)

        seq = self._send(self.hwid, OpenLstCmds.RF_PARAMS, msg)
        resp = self.get_packet_timeout(seqnum=seq)

        assert resp is not None
        assert resp["command"] == OpenLstCmds.ACK, hex(resp["command"])

        return f_actual, chanbw_actual, drate_actual, dev_act

    def set_bypass(self, bypass=True):
        self._send(self.hwid, OpenLstCmds.BYPASS, bytes([1 if bypass else 0]))


if __name__ == "__main__":
    import click
    import IPython
    from traitlets.config import get_config

    @click.command()
    @click.option("--port", default=None, help="Serial port")
    @click.option("--id", default=None, help="Serial interface ID")
    @click.option(
        "--rtscts", is_flag=True, default=False, help="Use RTS/CTS flow control"
    )
    @click.argument("hwid")
    def main(hwid, port, id, rtscts):
        logging.basicConfig(level="INFO")

        hwid = binascii.unhexlify(hwid)
        hwid = struct.unpack(">H", hwid)[0]

        openlst = OpenLst(port, hwid, rtscts=rtscts)

        with openlst:
            c = get_config()
            c.InteractiveShellEmbed.colors = "Linux"
            IPython.embed(header=SHELL_HEADER, config=c)

    main()
