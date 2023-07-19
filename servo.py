""" Main servo control class."""

import logging
import sys
import termios
import time
import tty

from dynamixel_sdk import *
from dynamixel_sdk import COMM_SUCCESS, PacketHandler, PortHandler

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


class Servo:

    def __init__(
        self,
        # Dynamixel servo ID
        id=1,
        # Minimum angle of servo (in dxl units)
        min_pos=0,
        # Maximum angle of servo (in dxl units)
        max_pos=100,
        # Timeout duration for servo moving to position
        move_timeout=5,
        # DYNAMIXEL Protocol Version (1.0 / 2.0)
        # https://emanual.robotis.com/docs/en/dxl/protocol2/
        protocol_version=2.0,
        # Define the proper baudrate to search DYNAMIXELs.
        baudrate=57600,
        # Use the actual port assigned to the U2D2.
        # ex) Windows: "COM*", Linux: "/dev/ttyUSB*", Mac: "/dev/tty.usbserial-*"
        devicename='/dev/ttyUSB0',
        # MX series with 2.0 firmware update.
        addr_torque_enable=64,
        addr_goal_position=116,
        addr_present_position=132,
        # Value for enabling the torque
        torque_enable=1,
        # Value for disabling the torque
        torque_disable=0,
        # Dynamixel moving status threshold
        dxl_moving_status_threshold=10,
    ):
        self.id = id
        self.min_pos = min_pos
        self.max_pos = max_pos
        self.move_timeout = move_timeout
        self.protocol_version = protocol_version
        self.baudrate = baudrate
        self.devicename = devicename
        self.addr_torque_enable = addr_torque_enable
        self.addr_goal_position = addr_goal_position
        self.addr_present_position = addr_present_position
        self.torque_enable = torque_enable
        self.torque_disable = torque_disable
        self.dxl_moving_status_threshold = dxl_moving_status_threshold
        self.torque_enabled = False

        log.info("Starting Servo communication")
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)

        def getch():
            try:
                tty.setraw(sys.stdin.fileno())
                ch = sys.stdin.read(1)
            finally:
                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
            return ch

        # Initialize PortHandler instance
        # Set the port path
        # Get methods and members of PortHandlerLinux or PortHandlerWindows
        self.portHandler = PortHandler(self.devicename)

        # Initialize PacketHandler instance
        # Set the protocol version
        # Get methods and members of Protocol1PacketHandler or Protocol2PacketHandler
        self.packetHandler = PacketHandler(self.protocol_version)

        # Open port
        if self.portHandler.openPort():
            log.info("Succeeded to open the port")
        else:
            log.error("Failed to open the port")
            log.error("Press any key to terminate...")
            getch()
            quit()

        # Set port baudrate
        if self.portHandler.setBaudRate(self.baudrate):
            log.info("Succeeded to change the baudrate")
        else:
            log.error("Failed to change the baudrate")
            log.error("Press any key to terminate...")
            getch()
            quit()

        log.info("Servo communication started")

    def enable_torque(self):
        # Enable Dynamixel Torque
        dxl_comm_result, dxl_error = self.packetHandler.write1ByteTxRx(
            self.portHandler, self.id, self.addr_torque_enable, self.torque_enable)
        if dxl_comm_result != COMM_SUCCESS:
            log.error("%s" % self.packetHandler.getTxRxResult(dxl_comm_result))
        elif dxl_error != 0:
            log.error("%s" % self.packetHandler.getRxPacketError(dxl_error))
        else:
            log.info(f"Servo {self.id} torque enabled")
        self.torque_enabled = True

    def disable_torque(self):
        # Disable Dynamixel Torque
        dxl_comm_result, dxl_error = self.packetHandler.write1ByteTxRx(
            self.portHandler, self.id, self.addr_torque_enable, self.torque_disable)
        if dxl_comm_result != COMM_SUCCESS:
            log.error("%s" % self.packetHandler.getTxRxResult(dxl_comm_result))
        elif dxl_error != 0:
            log.error("%s" % self.packetHandler.getRxPacketError(dxl_error))
        else:
            log.info(f"Servo {self.id} torque disabled")
        self.torque_enabled = False

    def __del__(self):
        self.disable_torque()

    def get_position(self):
        """ Get servo position. """
        dxl_present_position, dxl_comm_result, dxl_error = self.packetHandler.read4ByteTxRx(
            self.portHandler, self.id, self.addr_present_position)
        if dxl_comm_result != COMM_SUCCESS:
            log.error("%s" % self.packetHandler.getTxRxResult(dxl_comm_result))
        elif dxl_error != 0:
            log.error("%s" % self.packetHandler.getRxPacketError(dxl_error))
        return dxl_present_position

    def move(self, position):
        """ Move servo to position. """
        assert 0 <= position <= 1, "Position must be between 0 and 1"

        if not self.torque_enabled:
            self.enable_torque()

        # Convert action in [0, 1] to servo position [min_pos, max_pos].
        dxl_goal_position = position * (self.max_pos - self.min_pos) + self.min_pos

        # Write goal position
        dxl_comm_result, dxl_error = self.packetHandler.write4ByteTxRx(
            self.portHandler, self.id, self.addr_goal_position, int(dxl_goal_position))
        if dxl_comm_result != COMM_SUCCESS:
            log.error(f"{self.packetHandler.getTxRxResult(dxl_comm_result)}")
        elif dxl_error != 0:
            log.error(f"{self.packetHandler.getRxPacketError(dxl_error)}")
        else:
            log.info(f"Servo {self.id} set goal position to {position}")

        # Move to goal position with timeout
        timeout_start = time.time()
        while time.time() < timeout_start + self.move_timeout:

            # Read present position
            dxl_present_position, dxl_comm_result, dxl_error = self.packetHandler.read4ByteTxRx(
                self.portHandler, self.id, self.addr_present_position)
            if dxl_comm_result != COMM_SUCCESS:
                log.error(
                    f"{self.packetHandler.getTxRxResult(dxl_comm_result)}")
            elif dxl_error != 0:
                log.error(f"{self.packetHandler.getRxPacketError(dxl_error)}")
            else:
                log.debug(f"Servo {self.id} moving to {dxl_present_position}")

            if not abs(dxl_goal_position - dxl_present_position) > self.dxl_moving_status_threshold:
                log.debug(f"Servo {self.id} reached {dxl_goal_position}")
                break