import logging
import queue
import random
import serial
import serial.threaded
import struct
import time

from openlst_tools.commands import OpenLstCmds, MAX_DATA_LEN


class LstProtocol(serial.threaded.Protocol):
    START = b"\x22\x69"

    def __init__(self):
        self.packet = bytearray()
        self.in_packet = False
        self.transport = None

        self.byte_idx = 0
        self.packet_len = 0

        self.packet_queue = queue.Queue()

    def connection_mode(self, transport):
        self.transport = transport

    def connection_lost(self, exc):
        self.transport = None
        self.in_packet = False
        del self.packet[:]
        self.byte_idx = 0
        super(LstProtocol, self).connection_lost(exc)

    def data_received(self, data):
        # for byte in data:
        for byte in serial.iterbytes(data):
            # Handle start bytes
            if int(byte[0]) == self.START[0] and self.byte_idx == 0:
                self.byte_idx += 1

            elif int(byte[0]) == self.START[1] and self.byte_idx == 1:
                self.byte_idx += 1
                self.in_packet = True
                self.packet_len = 0

            elif self.in_packet:
                # Add all bytes to packet except start bytes
                self.packet.extend(byte)

                # Store length byte
                if self.byte_idx == 2:
                    self.packet_len = int(byte[0])

                # Finish packet
                if self.byte_idx > 2 and self.byte_idx - 2 == self.packet_len:
                    self.in_packet = False
                    self.packet_len = 0
                    self.byte_idx = 0

                    self.handle_packet(bytes(self.packet))
                    del self.packet[:]
                else:
                    self.byte_idx += 1

            else:
                self.handle_out_of_packet_data(byte)

    def handle_packet(self, packet_raw: bytes):
        packet = {}

        packet["len"] = packet_raw[0]
        packet["hwid"] = int.from_bytes(packet_raw[1:3], "big")
        packet["seq"] = int.from_bytes(packet_raw[3:5], "big")
        packet["system"] = packet_raw[5]
        packet["command"] = packet_raw[6]
        packet["data"] = packet_raw[7:]

        self.packet_queue.put_nowait(packet)

        # Print boot messages
        try:
            msg = packet["data"].decode()
        except UnicodeDecodeError:
            msg = ""

        if packet["command"] == OpenLstCmds.ASCII and "OpenLST" in msg:
            print(
                f"Boot message ({hex(packet['hwid'])}): {msg}"
            )  # TODO: figure out how to use logging without breaking ipython

    def handle_out_of_packet_data(self, data):
        print(f"Unexpected bytes: {data}")
        # pass


class LstHandler:
    def __init__(
        self,
        port: str,
        hwid: int,
        baud: int = 115200,
        rtscts: bool = False,
        timeout: float = 1,
    ) -> None:

        if port:
            self.ser = serial.Serial(port, baud, rtscts=rtscts)
        else:
            # Loop back for testing if no port is specified
            self.ser = serial.serial_for_url("loop://", baudrate=115200)

        self.timeout = timeout
        self.hwid = hwid

        # Create thread but don't start it yet
        self.thread = serial.threaded.ReaderThread(self.ser, LstProtocol)

        self.open = False
        self.protocol: LstProtocol = None

        # Initialize sequence number
        self.seq = random.randint(0, 65536)

        self.packets = []

    def __enter__(self):
        """Enter context"""
        self.thread.start()

        _, self.protocol = self.thread.connect()
        self.open = True

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context"""
        self.open = False
        self.thread.close()

    def packets_available(self):
        """Returns number of packets in queue."""

        while self.protocol.packet_queue.qsize() > 0:
            self.packets.append(self.protocol.packet_queue.get_nowait())

        return len(self.packets)

    def get_packet(self, cmd=None, seqnum=None):
        if self.packets_available() == 0:
            return None
        elif seqnum:
            for i, packet in enumerate(self.packets):
                if packet["seq"] == seqnum:
                    return self.packets.pop(i)

            return None
        elif cmd:
            for i, packet in enumerate(self.packets):
                if packet["command"] == cmd:
                    return self.packets.pop(i)

            return None
        else:
            return self.packets.pop(0)

    def get_packet_timeout(self, cmd=None, seqnum=None, timeout=None):
        start = time.time()

        if timeout == None:
            timeout = self.timeout

        while time.time() - start < timeout:
            resp = self.get_packet(cmd, seqnum)

            if resp is not None:
                return resp

    def clean_packets(self, cmd=None):
        if self.packets_available() == 0:
            return

        if cmd:
            self.packets = [x for x in self.packets if x["command"] != cmd]
        else:
            del self.packets[:]

    def _send(self, hwid: int, cmd: int, msg: bytes = bytes()):
        packet = bytearray()

        packet.extend(b"\x22\x69")
        packet.append(6 + len(msg))
        packet.extend(struct.pack(">H", hwid))
        packet.extend(struct.pack(">H", self.seq))
        packet.append(0x01)  # TODO: figure this out
        packet.append(cmd)
        packet.extend(msg)

        self.thread.write(packet)

        seq = self.seq

        self.seq += 1
        self.seq %= 2**16

        return seq


if __name__ == "__main__":
    import click
    import IPython

    @click.command()
    @click.option("--port", default=None, help="Serial port")
    @click.option(
        "--rtscts", is_flag=True, default=False, help="Use RTS/CTS flow control"
    )
    def main(port, rtscts):
        logging.basicConfig(level="INFO")

        if port:
            ser = serial.Serial(port, baudrate=115200, rtscts=rtscts)
        else:
            ser = serial.serial_for_url("loop://", baudrate=115200, timeout=1)

        thread = serial.threaded.ReaderThread(ser, LstProtocol)
        with thread as protocol:
            IPython.embed()

    main()
