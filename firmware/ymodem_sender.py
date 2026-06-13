import serial
import sys
import os

# YMODEM 协议控制字符
SOH = b'\x01'
STX = b'\x02'
EOT = b'\x04'
ACK = b'\x06'
NAK = b'\x15'
CAN = b'\x18'
C   = b'\x43'

def calc_crc(data: bytes) -> int:
    """计算 YMODEM (CRC16-CCITT)"""
    crc = 0x0000
    for byte in data:
        crc ^= (byte << 8)
        for _ in range(8):
            if crc & 0x8000:
                crc = ((crc << 1) ^ 0x1021) & 0xFFFF
            else:
                crc = (crc << 1) & 0xFFFF
    return crc

def hex_dump(data: bytes) -> str:
    """格式化打印 Hex"""
    if not data:
        return "[Empty]"
    return " ".join([f"{b:02X}" for b in data])

class YmodemDebugger:
    def __init__(self, port, baudrate, filename):
        self.port = port
        self.baudrate = baudrate
        self.filename = filename
        self.ser = None

    def tx(self, data: bytes, desc: str = ""):
        """发送数据并打印日志"""
        print(f"[TX] {desc:<20} | Length: {len(data):<4} | Data: {hex_dump(data[:16])} ...")
        self.ser.write(data)

    def rx(self, expected_len: int = 1, timeout: float = 3.0) -> bytes:
        """接收数据并打印日志"""
        self.ser.timeout = timeout
        data = self.ser.read(expected_len)
        if data:
            # 尝试解析常用的控制字符方便阅读
            ctrl_char = ""
            if len(data) == 1:
                if data == C: ctrl_char = " ('C')"
                elif data == ACK: ctrl_char = " (ACK)"
                elif data == NAK: ctrl_char = " (NAK)"
                elif data == CAN: ctrl_char = " (CAN)"

            print(f"[RX] Received            | Length: {len(data):<4} | Data: {hex_dump(data)}{ctrl_char}")
        else:
            print(f"[RX] Timeout! Expected {expected_len} bytes.")
        return data

    def send_file(self):
        try:
            self.ser = serial.Serial(self.port, self.baudrate)
            print(f"=== Serial Port {self.port} Opened at {self.baudrate} bps ===")
        except Exception as e:
            print(f"Failed to open port {self.port}: {e}")
            return

        file_size = os.path.getsize(self.filename)
        base_name = os.path.basename(self.filename).encode('utf-8')
        print(f"=== Start sending '{base_name.decode()}' ({file_size} bytes) ===")

        # ---------------- Step 1: 等待初始的 'C' ----------------
        print("\n--- Step 1: Waiting for initial 'C' from FPGA ---")
        while True:
            b = self.rx(1, timeout=1.0)
            if b == C:
                break
            elif b == b'':
                pass
            else:
                print("    (Ignored unexpected bytes)")

        # ---------------- Step 2: 发送 Block 0 (文件名和大小) ----------------
        print("\n--- Step 2: Sending Block 0 (File Info) ---")
        payload = base_name + b'\x00' + str(file_size).encode('utf-8') + b'\x00'
        payload += b'\x00' * (128 - len(payload))
        crc = calc_crc(payload)
        packet = SOH + b'\x00\xFF' + payload + crc.to_bytes(2, 'big')
        self.tx(packet, "Block 0")

        ans = self.rx(1, timeout=2.0)
        if ans != ACK:
            print("ERROR: Expected ACK for Block 0")
            return

        ans = self.rx(1, timeout=2.0)
        if ans != C:
            print("ERROR: Expected 'C' after Block 0 ACK")
            return

        # ---------------- Step 3: 发送文件数据 ----------------
        print("\n--- Step 3: Sending File Data ---")
        seq = 1
        with open(self.filename, 'rb') as f:
            while True:
                chunk = f.read(1024)
                if not chunk:
                    break

                if len(chunk) <= 128:
                    packet_size = 128
                    head = SOH
                else:
                    packet_size = 1024
                    head = STX

                if len(chunk) < packet_size:
                    chunk += b'\x1A' * (packet_size - len(chunk))

                header = head + bytes([seq & 0xFF, 0xFF - (seq & 0xFF)])
                crc = calc_crc(chunk)
                packet = header + chunk + crc.to_bytes(2, 'big')

                self.tx(packet, f"Data Block {seq}")
                ans = self.rx(1, timeout=2.0)
                if ans != ACK:
                    print(f"ERROR: Expected ACK for Data Block {seq}, but got {hex_dump(ans)}")
                    return
                seq += 1

        # ---------------- Step 4: 传输结束握手 (EOT) ----------------
        print("\n--- Step 4: End Of Transmission (EOT) ---")
        self.tx(EOT, "First EOT")
        ans = self.rx(1)
        if ans != NAK:
            print("ERROR: Expected NAK for First EOT")

        self.tx(EOT, "Second EOT")
        ans = self.rx(1)
        if ans != ACK:
            print("ERROR: Expected ACK for Second EOT")
        ans = self.rx(1)
        if ans != C:
            print("ERROR: Expected 'C' after Second EOT ACK")

        # ---------------- Step 5: 发送空 Block 0 结束会话 ----------------
        print("\n--- Step 5: Sending Empty Block 0 to close session ---")
        payload = b'\x00' * 128
        crc = calc_crc(payload)
        packet = SOH + b'\x00\xFF' + payload + crc.to_bytes(2, 'big')
        self.tx(packet, "Empty Block 0")

        ans = self.rx(1)
        if ans == ACK:
            print("\n=== SUCCESS: File transferred completely! ===")
            # 成功后，不关闭串口，直接进入终端模式
            self.terminal_mode()
        else:
            print("WARNING: Expected ACK for Empty Block 0, session might not close cleanly.")
            self.ser.close()

    def terminal_mode(self):
        """传输完成后，保持串口常开，作为终端打印后续内容"""
        print("\n=======================================================")
        print("=== Entering Terminal Mode to capture printf output ===")
        print("===           (Press Ctrl+C to exit)                ===")
        print("=======================================================\n")

        # 将读取超时设短一点，以便能及时响应 Ctrl+C
        self.ser.timeout = 0.1
        try:
            while True:
                data = self.ser.read(1024)
                if data:
                    # 使用 replace 容错解码，防止有些二进制杂音导致 Python 崩溃
                    # flush=True 确保第一时间输出到屏幕
                    print(data.decode('utf-8', errors='replace'), end='', flush=True)
        except KeyboardInterrupt:
            print("\n\n=== Exited Terminal Mode ===")
        finally:
            if self.ser and self.ser.is_open:
                self.ser.close()

if __name__ == '__main__':
    if len(sys.argv) < 4:
        print("Usage: python ymodem_sender.py <COM_PORT> <BAUDRATE> <FILE_PATH>")
        print("Example: python ymodem_sender.py COM3 115200 test.bin")
        sys.exit(1)

    port = sys.argv[1]
    baudrate = int(sys.argv[2])
    filepath = sys.argv[3]

    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        sys.exit(1)

    debugger = YmodemDebugger(port, baudrate, filepath)
    debugger.send_file()
