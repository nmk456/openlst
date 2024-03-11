#!/usr/bin/env python

import struct
import uhd
import uhd.libpyuhd as lib
import numpy as np
from commpy import filters
import matplotlib.pyplot as plt
import time
import random

fec_table = [
    0, 3, 1, 2,
    3, 0, 2, 1,
    3, 0, 2, 1,
    0, 3, 1, 2
]

class UsrpWrapper:
    """Convenience wrapper for more precise control than MultiUSRP provides"""

    # TODO: multiple channels
    def __init__(self):
        self.usrp = uhd.usrp.MultiUSRP()

        st_args = lib.usrp.stream_args("fc32", "sc16")
        st_args.channels = [0]
        self.streamer = self.usrp.get_tx_stream(st_args)

        self.chan = 0 # TODO

    def __del__(self):
        del self.streamer
        del self.usrp

    def set_tx_rate(self, rate: float):
        self.usrp.set_tx_rate(rate, self.chan)
    
    def set_tx_freq(self, freq: float):
        self.usrp.set_tx_freq(lib.types.tune_request(freq), self.chan)
    
    def set_tx_gain(self, gain: float):
        self.usrp.set_tx_gain(gain, self.chan)
    
    def tx(self, samples):
        # Samples must be np.complex64

        metadata = lib.types.tx_metadata()
        metadata.end_of_burst = True

        self.streamer.send(samples, metadata)

class ModulatorDemodulator:
    def __init__(self, Fsamp: float, Fsym: float = None, sps: float = None,
                 Tsym: float = None) -> None:
        """Initialize the modulator."""
        self.Fsamp = Fsamp

        if Fsym is not None:
            self.Fsym = Fsym
            self.sps = Fsamp/Fsym
            self.Tsym = 1/Fsym
        elif sps is not None:
            self.sps = sps
            self.Fsym = Fsamp/sps
            self.Tsym = 1/self.Fsym
        elif Tsym is not None:
            self.Tsym = Tsym
            self.Fsym = 1/Tsym
            self.sps = Fsamp/Tsym

        if np.abs(self.sps - round(self.sps)) > 1e-5:
            print(f"Warning: sps={self.sps} is not an integer!")
        else:
            self.sps = int(self.sps)

        self.mod_order = 0
        self.mod_index = None

        self.filter_t = None
        self.filter_h = None

        self.N = 0

    def apply_filter(self, Nsamp: int = None, Nsps: int = 17, alpha: float = 0.35
                     ) -> None:
        """Applies a root-raised-cosine filter to the samples.

        This filter is useful for matched filtering.

        Args:
            N (int): The length of the filter, in samples. By default, this is
                set to `int(sps * Nsps + 1)`.
            Nsps (int): The length of the filter, in symbols. Set to 17 by
                default. If `N` is specified, this parameter is ignored.
            alpha (float): The roll-off factor of the filter. Set to 0.35 by
                default.
        """
        if self.filter_t is None or self.filter_h is None:
            if Nsamp is None:
                Nsamp = int(self.sps*Nsps + 1)
            if self.mod_index is not None:
                pass
                self.filter_t, self.filter_h = filters.rrcosfilter(
                    N=Nsamp, alpha=alpha, Ts=self.Tsym/(2*self.mod_index), Fs=self.Fsamp)
            else:
                self.filter_t, self.filter_h = filters.rrcosfilter(
                    N=Nsamp, alpha=alpha, Ts=self.Tsym, Fs=self.Fsamp)

        self.samples = np.convolve(
            self.samples, self.filter_h).astype(np.complex64)
        self.N = len(self.samples)

    def normalize(self, factor=1.0):
        """Normalize the samples, and multiply the by the given factor.

        Divides the samples by the value of the one with the largest magnitude.
        For example, this sets the samples to be between -2**14 and 2**14, with
        a total range of 2**15:

        `modulator.normalize(factor=2**14)`

        Args:
            factor (float): The factor to multiply the samples by. By default,
                this is set to 1.0.
        """
        self.samples /= np.max(np.abs(self.samples))
        self.samples *= factor

    def shift_frequency(self, freq_offset):
        max_shift = self.Fsamp / (2 * self.mod_order)

        if freq_offset > max_shift:
            print("Warning: Frequency offset is too high and will be clipped.")
            freq_offset = max_shift

        t = np.arange(self.N) / self.Fsamp
        self.samples *= np.exp(2.0j * np.pi * freq_offset * t)

    def get_data(self):
        return self.data

    def set_data(self, data):
        """Set data containing values for each symbol"""
        self.data = data

    def get_samples(self):
        return self.samples

    def set_samples(self, samples):
        self.samples = samples
        self.N = len(self.samples)

    def show_psd(self, unit='MHz'):
        psd = np.fft.fft(self.samples/np.max(np.abs(self.samples))) / self.N
        psd = np.fft.fftshift(psd)
        psd = 10 * np.log10(np.abs(psd))
        f = np.linspace(-self.Fsamp/2.0, self.Fsamp/2.0, len(psd))

        assert unit in ["Hz", "kHz", "MHz", "GHz"]
        f /= 10**(["Hz", "kHz", "MHz", "GHz"].index(unit) * 3)

        plt.plot(f, psd)
        plt.ylabel("Power (dBFS)")
        plt.xlabel(f"Frequency ({unit})")


class Modulator(ModulatorDemodulator):
    """Creates a modulator."""

    def __init__(self, Fsamp, Fsym=None, sps=None, Tsym=None, order=2) -> None:
        super().__init__(Fsamp, Fsym=Fsym, sps=sps, Tsym=Tsym)

        self.mod_order = order

    def modulate(self, symbols=[0.0, 1.0], premable=[]):
        bin_data = []
        if len(premable) > 0:
            bin_data.extend(premable)
        bin_data.extend(self.data)

        self.samples = np.array([symbols[b] for b in bin_data])
        self.samples = np.repeat(self.samples, int(self.sps))
        self.samples = self.samples.astype(np.complex64)
        self.N = len(self.samples)


class FSKModulator(Modulator):
    def __init__(self, Fsamp, Fsym=None, sps=None, Tsym=None, h=None, order=2) -> None:
        super().__init__(Fsamp, Fsym, sps, Tsym, order)

        if h is not None:
            self.mod_index = h
            self.Fdev = self.Fsym * h
        else:
            raise ValueError("FSK Mod Index Required")

    def modulate(self):
        q_offsets_vals = [-1, 1]
        q_offsets = [q_offsets_vals[int(b)] for b in self.data]

        sample_offsets = np.repeat(q_offsets, self.sps)

        t = np.arange(len(sample_offsets))/self.Fsamp
        self.samples = np.exp(1j*2*np.pi*t*self.Fdev)
        self.samples.imag *= sample_offsets
        self.N = len(self.samples)

class PN9Generator:
    def __init__(self):
        self.val = 0x1FF

    def __iter__(self):
        pass

    def __next__(self):
        old_val = self.val

        for i in range(8):
            new_msb = (self.val >> 0 & 1) ^ (
                self.val >> 5 & 1)  # XOR of bits 0 and 5
            self.val = (self.val >> 1) | (new_msb << 8)  # Shift and append

        return old_val & 0xFF
    
class ViterbiDecode:
    """Source: https://www.ti.com/lit/an/swra313/swra313.pdf"""
    TRELLIS_SOURCE_STATE_LUT = [[0, 4], [0, 4], [
        1, 5], [1, 5], [2, 6], [2, 6], [3, 7], [3, 7]]

    TRELLIS_TRANSITION_OUTPUT = [[0, 3], [3, 0], [
        1, 2], [2, 1], [3, 0], [0, 3], [2, 1], [1, 2]]

    TRELLIS_TRANSITION_INPUT = [0, 1, 0, 1, 0, 1, 0, 1]

    def __init__(self):
        self.nCost = [[0]*8, [0]*8]
        self.aPath = [[0]*8, [0]*8]

        for n in range(1, 8):
            self.nCost[0][n] = 100

        self.iLastBuf = 0
        self.iCurrBuf = 1
        self.nPathBits = 0

    def decodeFEC(self, pInData, nRemBytes):
        """Decodes 4 bytes of data at a time.

        Args:
            pInData (list of int): List of 4 bytes to be decoded
            nRemBytes (int): Number of bytes remaining, including current pInData bytes

        Returns:
            List of decoded bytes
        """
        pDecData = []
        nOutputBytes = 0
        nMinCost = 0
        iBit = 8 - 2

        pInIdx = 0
        pDecIdx = 0

        # De-interleave received data (and change pInData to point to de-interleaved data)
        aDeintData = [0]*4

        try:
            for iOut in range(4):
                dataByte = 0
                for iIn in range(3, -1, -1):
                    dataByte = (dataByte << 2) | (
                        (pInData[iIn] >> (2 * iOut)) & 0x03)
                aDeintData[iOut] = dataByte
        except IndexError:
            return nRemBytes, []

        pInData = aDeintData

        # Process up to 4 bytes of de-interleaved input data, processing one encoder symbol (2b) at a time
        for nIterations in range(16):
            symbol = (pInData[pInIdx] >> iBit) & 0x03

            # Find minimum cost so that we can normalize costs (only last iteration used)
            nMinCost = 0xFF

            # Get 2b input symbol (MSB first) and do one iteration of Viterbi decoding
            iBit -= 2
            if iBit < 0:
                iBit = 6
                pInIdx += 1  # Update pointer to the next byte of received data

            # For each destination state in the trellis, calculate hamming costs for both possible paths into state and select the one with lowest cost.
            for iDestState in range(8):
                nInputBit = self.TRELLIS_TRANSITION_INPUT[iDestState]

                # Calculate cost of transition from each of the two source states (cost is Hamming difference between received 2b symbol and expected symbol for transition)
                iSrcState0 = self.TRELLIS_SOURCE_STATE_LUT[iDestState][0]
                nCost0 = self.nCost[self.iLastBuf][iSrcState0]
                nCost0 += hammWeight(symbol ^
                                     self.TRELLIS_TRANSITION_OUTPUT[iDestState][0])

                iSrcState1 = self.TRELLIS_SOURCE_STATE_LUT[iDestState][1]
                nCost1 = self.nCost[self.iLastBuf][iSrcState1]
                nCost1 += hammWeight(symbol ^
                                     self.TRELLIS_TRANSITION_OUTPUT[iDestState][1])


                # Select transition that gives lowest cost in destination state, copy that source state's path and add new decoded bit
                if nCost0 <= nCost1:
                    self.nCost[self.iCurrBuf][iDestState] = bit_trim(nCost0, 8)
                    nMinCost = min(nMinCost, nCost0)
                    self.aPath[self.iCurrBuf][iDestState] = (
                        self.aPath[self.iLastBuf][iSrcState0] << 1) | nInputBit
                    self.aPath[self.iCurrBuf][iDestState] = bit_trim(
                        self.aPath[self.iCurrBuf][iDestState], 32)
                else:
                    self.nCost[self.iCurrBuf][iDestState] = bit_trim(nCost1, 8)
                    nMinCost = min(nMinCost, nCost1)
                    self.aPath[self.iCurrBuf][iDestState] = (
                        self.aPath[self.iLastBuf][iSrcState1] << 1) | nInputBit
                    self.aPath[self.iCurrBuf][iDestState] = bit_trim(
                        self.aPath[self.iCurrBuf][iDestState], 32)

            self.nPathBits += 1

            # If trellis history is sufficiently long, output a byte of decoded data
            if (self.nPathBits == 32):
                # pDecData[pDecIdx] = (self.aPath[self.iCurrBuf][0] >> 24) & 0xFF
                pDecData.append((self.aPath[self.iCurrBuf][0] >> 24) & 0xFF)
                # pDecIdx += 1
                nOutputBytes += 1
                self.nPathBits -= 8
                nRemBytes -= 1

            # After having processed 3-symbol trellis terminator, flush out remaining data
            if (nRemBytes <= 3) and (self.nPathBits == (8 * nRemBytes + 3)):
                while self.nPathBits >= 8:
                    # pDecData[pDecIdx] = (self.aPath[self.iCurrBuf][0] >> (self.nPathBits - 8)) & 0xFF
                    pDecData.append(
                        (self.aPath[self.iCurrBuf][0] >> (self.nPathBits - 8)) & 0xFF)
                    # pDecIdx += 1
                    nOutputBytes += 1
                    self.nPathBits -= 8

                return nOutputBytes, pDecData

            # Swap current and last buffers for next iteration
            self.iLastBuf = (self.iLastBuf + 1) % 2
            self.iCurrBuf = (self.iCurrBuf + 1) % 2

        # Normalize costs so that minimum cost becomes 0
        for iState in range(8):
            self.nCost[self.iLastBuf][iState] -= nMinCost

        return nOutputBytes, pDecData

    def decode(self, data):
        bytes_before = len(data)
        bytes_after = bytes_before // 4 * 2

        nBytes = bytes_after
        decoded_data = []
        arr_in_idx = 0

        while nBytes > 0:
            nBytesOut, bytesOut = self.decodeFEC(data[arr_in_idx:arr_in_idx+4], nBytes)

            nBytes -= nBytesOut
            decoded_data.extend(bytesOut)
            arr_in_idx += 4
        
        return decoded_data

def whiten(data: bytes) -> bytes:
    pn9 = PN9Generator()

    return bytes([next(pn9) ^ x for x in data])

# source: https://stackoverflow.com/a/55850496
def crc16(data : bytearray, poly=0x8005):
    crc = 0xFFFF
    for i in range(len(data)):
        crc ^= data[i] << 8
        for _ in range(0,8):
            if (crc & 0x8000) > 0:
                crc = (crc << 1) ^ poly
            else:
                crc = crc << 1
    return crc & 0xFFFF

def bit_trim(val, bits):
    return val & (2 ** bits - 1)

def hammWeight(a):
    bit_trim(a, 8)

    a = ((a & 0xAA) >> 1) + (a & 0x55)
    a = ((a & 0xCC) >> 2) + (a & 0x33)
    a = ((a & 0xF0) >> 4) + (a & 0x0F)

    bit_trim(a, 8)
    return a

# Source: https://www.ti.com/lit/an/swra113a/swra113a.pdf
def fec_encode(data: bytes):
    fec_reg = 0

    N = 2 * (len(data)//2 + 1)

    data = data + b"\x0B"*(N - len(data))

    out = []

    for i in range(N):
        fec_reg = (fec_reg & 0x700) | (data[i] & 0xFF)
        fec_out = 0

        for _ in range(8):
            fec_out = (fec_out << 2 ) | fec_table[fec_reg >> 7]
            fec_reg = (fec_reg << 1) & 0x7FF

        out.append(fec_out >> 8)
        out.append(fec_out & 0xFF)

    return bytes(out)

# Running interleave twice gets original data back
def interleave(data: bytes):
    assert len(data) % 2 == 0, "Data length must be divisible by 2"

    out = []

    for i in range(0, len(data), 4):
        int_out = 0

        for j in range(16):
            int_out = (int_out << 2) | ((data[i + (~j & 0x03)] >> (2 * ((j & 0x0C) >> 2)) & 0x03))

        out.append((int_out >> 24) & 0xFF)
        out.append((int_out >> 16) & 0xFF)
        out.append((int_out >> 8) & 0xFF)
        out.append(int_out & 0xFF)
    
    return bytes(out)


class SDR():
    def __init__(self, fsym=7416):
        self.usrp = UsrpWrapper()
        self.seq = random.randint(0, 65536)

        self.set_freq()
        self.set_gain()
        self.set_symbol_rate(fsym)

    def set_symbol_rate(self, rate=7416):
        self.Fsym = rate
        self.Fsamp = self.Fsym * 16
        # self.sps = int(self.Fsamp / self.Fsym)
        # Tsym = 1/self.Fsym
        h = 0.5

        self.usrp.set_tx_rate(self.Fsamp)
        self.mod = FSKModulator(self.Fsamp, self.Fsym, h=h)

    def set_freq(self, freq=437e6):
        self.usrp.set_tx_freq(freq)
    
    def set_gain(self, gain=30):
        self.usrp.set_tx_gain(gain)

    def transmit(self, cmd, msg, dest_hwid=0x0000):

        # command header + msg + HWID + 3
        rf_len = len(msg) + 9
        header = bytes([0x55, 0x55, 0x55, 0x55, 0xD3, 0x91, 0xD3, 0x91])

        packet = bytearray()
        packet.append(rf_len)
        packet.append(0x40) # flag ?
        packet.extend(struct.pack("<H", self.seq)) # little endian
        packet.extend([0x01, cmd & 0xFF]) # system and command
        packet.extend(msg)
        packet.extend(struct.pack("<H", dest_hwid))

        crc_data = crc16(packet)
        packet.extend([crc_data & 0xFF, crc_data >> 8])

        packet = whiten(packet)
        packet = fec_encode(packet)
        packet = interleave(packet)

        packet = header + packet
        bits = "".join([f'{x:08b}' for x in packet])

        # Modulate
        self.mod.set_data(bits)
        self.mod.modulate()
        self.mod.apply_filter()
        self.mod.normalize()

        samples = self.mod.get_samples()

        # Add a 5ms gap at the end of the packet
        samples = np.concatenate((samples, np.zeros(int(self.Fsamp/200), dtype=np.complex64)))

        # Transmit
        self.usrp.tx(samples)

        self.seq += 1
        self.seq %= 2**16



