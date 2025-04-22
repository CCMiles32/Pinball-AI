import time

class ServoController:
    def __init__(self):
        # Setup serial connection to servo
  
        pass

    def send_action(self, action):
        if action == 1:  # LEFT_FLIPPER
            self.left_flipper()
        elif action == 2:  # RIGHT_FLIPPER
            self.right_flipper()
        elif action == 3:  # BOTH_FLIPPERS
            self.left_flipper()
            self.right_flipper()

    def left_flipper(self):
        print("LEFT FLIPPER ACTIVATED")
        # self.ser.write(b'L')

        time.sleep(0.1)

    def right_flipper(self):
        print("RIGHT FLIPPER ACTIVATED")
        # self.ser.write(b'R')
        
        time.sleep(0.1)

def setup_serial_connection():
    # Return an instance of ServoController
    return ServoController()