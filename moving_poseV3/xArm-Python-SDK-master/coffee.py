#!/usr/bin/env python3
import os
import time
from xarm.wrapper import XArmAPI

arm = XArmAPI('192.168.1.181')
time.sleep(0.5)
#Clean error and warn
if arm.warn_code != 0:
    arm.clean_warn()
if arm.error_code != 0:
    arm.clean_error()

#Enable the robot
arm.motion_enable(enable=True)

arm.set_mode(0)
arm.set_state(0)
oriAcc=arm.last_used_joint_acc

can_angle = [-66.3, 81, 149.2, -0.3, 66.6, -280, 0.0]
close_angle = [-66.3, 82.1, 149.1, -0.3, 65.4, 265, 0.0]

def BoillingWater():
    return

def OpenCan():
    arm.set_servo_angle(angle=[0,0,0,0,0,0], speed=80)
    arm.set_position(x=176, y=-404.9, z=217.7, roll=178.7, pitch=-0.8, yaw=1.6, speed=80,mvacc=100)
    arm.set_servo_angle(servo_id=6, angle=280, speed=600)
    arm.set_tool_position(z=19)
    arm.open_lite6_gripper()
    arm.set_pause_time(0.8)
    arm.set_servo_angle(angle=can_angle, speed=560)
    arm.set_pause_time(0.8)
    arm.set_position(z=226.5)
    arm.set_pause_time(0.8)
    arm.set_position(x=-28.8, y=-233.6, z=230.7, roll=-179.6, pitch=-2.2, yaw=-159.2, speed=150)
    arm.set_pause_time(0.8)
    arm.set_position(z=84)
    arm.set_pause_time(0.8)
    arm.close_lite6_gripper()
    arm.set_pause_time(0.8)
    arm.set_position(z=226.5)
    arm.set_servo_angle(servo_id=6, angle=0, speed=600)
    arm.set_pause_time(0.8)
    return

def CloseCan():
    # arm.set_servo_angle(servo_id=6, angle=0, speed=600)
    arm.set_position(x=-28.8, y=-233.6, z=226.7, roll=-179.6, pitch=-2.2, yaw=-159.2, speed=100)
    arm.set_pause_time(0.8)
    arm.set_position(z=82)
    arm.set_pause_time(0.8)
    arm.open_lite6_gripper()
    arm.set_pause_time(0.8)
    arm.set_position(z=226.5)
    arm.set_pause_time(0.8)
    arm.set_position(x=176, y=-404.9, z=226.5, roll=178.7, pitch=-0.8, yaw=1.6, speed=80, mvacc=100)
    arm.set_tool_position(z=20)
    arm.set_pause_time(0.8)
    arm.set_servo_angle(angle=close_angle, speed=600,wait=True)
    arm.set_pause_time(0.8)
    arm.close_lite6_gripper()
    arm.set_pause_time(0.8)
    arm.set_position(z=217.7,mvacc=30)
    arm.set_pause_time(0.8)
    arm.set_servo_angle(angle=[0,0,0,0,0,0], speed=40)
    return

def CoffeePowder():
    arm.set_position(x=250, y=-326.6, z=230.7, roll=-84, pitch=-76.8, yaw=127.9, speed=100)
    arm.set_position(z=81.2)
    arm.set_pause_time(0.8)
    arm.open_lite6_gripper()
    arm.set_pause_time(0.5)
    arm.set_tool_position(z=8)
    arm.set_pause_time(0.5)
    arm.set_tool_position(z=-8)
    arm.set_pause_time(0.3)
    arm.set_position(z=234)
    arm.set_pause_time(0.6)
    arm.set_position(x=259.5, y=-199.7, z=231.9, roll=43.9, pitch=-62.2, yaw=131.2, speed=100)
    arm.set_pause_time(0.5)
    arm.set_tool_position(yaw=-88)
    arm.set_pause_time(1.2)
    arm.set_tool_position(yaw=183)
    arm.set_tool_position(yaw=-95)
    arm.set_pause_time(0.5)
    arm.set_position(x=250, y=-326.6, z=231.2, roll=-84, pitch=-76.8, yaw=127.9)
    arm.set_pause_time(0.5)
    arm.set_position(z=82.2)
    arm.set_pause_time(0.5)
    arm.set_tool_position(y=7, z=11)
    arm.set_pause_time(0.5)
    arm.close_lite6_gripper()
    arm.set_pause_time(0.5)
    arm.set_tool_position(z=-15)
    arm.set_pause_time(0.5)
    arm.set_position(z=226.3)
    arm.set_pause_time(0.5)
    return

def Milk():
    arm.set_position(x=96.7, y=-340.9, z=218.1, roll=-19.1, pitch=-85, yaw=109.2, speed=100)
    arm.set_position(z=78.1)
    arm.open_lite6_gripper()
    arm.set_pause_time(1)
    arm.set_tool_position(x=-5, z=13, pitch=-5, speed=100)
    arm.set_pause_time(1)
    arm.set_tool_position(pitch=30)
    arm.set_position(z=228.1,speed=70)
    arm.set_pause_time(0.5)
    arm.set_position(x=355.2, y=-23.7, z=223.1, roll=-15.5, pitch=-58.3, yaw=168.9, speed=100)
    arm.set_pause_time(0.5)
    arm.set_tool_position(yaw=50)
    arm.set_pause_time(25)
    arm.set_tool_position(yaw=-50)
    arm.set_pause_time(0.5)
    arm.set_position(x=96.7,y=-340.9,z=228.1,roll=-19.1,pitch=-75,yaw=109.2,speed=200)
    arm.set_position(z=78.1,speed=70)
    arm.set_position(pitch=-85)
    arm.set_pause_time(1)
    arm.close_lite6_gripper()
    arm.set_tool_position(z=-20)
    arm.set_pause_time(0.5)
    return


OpenCan()
CoffeePowder()
Milk()
CloseCan()