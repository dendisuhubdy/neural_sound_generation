"""

Copyrigt Dendi Suhubdy, 2018
All rights reserved

"""

import csv
import numpy as np
from numpy import genfromtxt
import sys
import time

try:
    import thread
except ImportError:
    import _thread as thread

import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions.normal import Normal
from torch.distributions import kl_divergence
from torch.autograd import Function

from pca import run_pca, run_pca_np
import Leap, sys, time
from Leap import CircleGesture, KeyTapGesture, ScreenTapGesture, SwipeGesture

def load_model():
    pass


def load_pca():
    my_data = genfromtxt('./results/joint_angle_data.csv', delimiter=',')
    # remember to do a transpose on the input
    pca_matrix = run_pca(my_data.T)
    return pca_matrix


class JointAngleListener(Leap.Listener):
    finger_names = ['Thumb', 'Index', 'Middle', 'Ring', 'Pinky']
    bone_names = ['Metacarpal', 'Proximal', 'Intermediate', 'Distal']
    state_names = ['STATE_INVALID', 'STATE_START', 'STATE_UPDATE', 'STATE_END']

    def on_init(self, controller):
        self.pca_matrix = load_pca()
        model = load_model()
        print("Initialized")

    def on_connect(self, controller):
        print("Connected")

        # Enable gestures
        controller.enable_gesture(Leap.Gesture.TYPE_CIRCLE);
        controller.enable_gesture(Leap.Gesture.TYPE_KEY_TAP);
        controller.enable_gesture(Leap.Gesture.TYPE_SCREEN_TAP);
        controller.enable_gesture(Leap.Gesture.TYPE_SWIPE);

    def on_disconnect(self, controller):
        # Note: not dispatched when running in a debugger.
        print("Disconnected")

    def on_exit(self, controller):
        print("Exited")

    def on_frame(self, controller):
        # Get the most recent frame and report some basic information
        frame = controller.frame()

        # print("Frame id: %d, timestamp: %d, hands: %d, fingers: %d, tools: %d, gestures: %d" % (
              # frame.id, frame.timestamp, len(frame.hands), len(frame.fingers), len(frame.tools), len(frame.gestures())))

        # Get hands
        for hand in frame.hands:

            handType = "Left hand" if hand.is_left else "Right hand"

            # print("  %s, id %d, position: %s" % (
                # handType, hand.id, hand.palm_position))

            # Get the hand's normal vector and direction
            normal = hand.palm_normal
            direction = hand.direction

            # Calculate the hand's pitch, roll, and yaw angles
            # print("  pitch: %f degrees, roll: %f degrees, yaw: %f degrees" % (
                # direction.pitch * Leap.RAD_TO_DEG,
                # normal.roll * Leap.RAD_TO_DEG,
                # direction.yaw * Leap.RAD_TO_DEG))
            
            # Get arm bone
            arm = hand.arm
            # print("  Arm direction: %s, wrist position: %s, elbow position: %s" % (
                # arm.direction,
                # arm.wrist_position,
                # arm.elbow_position))

            joint_angle_array = [direction.pitch, normal.roll, direction.yaw]
            
            # Get fingers
            for finger in hand.fingers:

                # print("    %s finger, id: %d, length: %fmm, width: %fmm" % (
                    # self.finger_names[finger.type],
                    # finger.id,
                    # finger.length,
                    # finger.width))

                # Get bones
                prev_bone_direction = None

                for b in range(0, 4):
                    bone = finger.bone(b)
                    # Structure of the data
                    # bone.direction
                    # is the direction that we are looking for
                    # using the joint angles
                    # bone.direction is a tuple of 3 values
                    # print(f"Bone {b} : value 0")
                    # print(bone.direction)
                    # for example (-0.110091, -0.521456, -0.846146)

                    # print("      Bone: %s, start: %s, end: %s, direction: %s" % (
                        # self.bone_names[bone.type],
                        # bone.prev_joint,
                        # bone.next_joint,
                        # bone.direction))
                    
                    if prev_bone_direction is None:
                        prev_bone_direction = np.asarray([bone.direction[0], bone.direction[1], bone.direction[2]])
                        # print("prev_bone_direction")
                        # print(prev_bone_direction)
                    else:
                        curr_bone_direction = np.asarray([bone.direction[0], bone.direction[1], bone.direction[2]])
                        # print("current_bone_direction")
                        # print(curr_bone_direction)
                        joint_angle_bone = np.dot(prev_bone_direction, curr_bone_direction.T)
                        # joint angle bone must be a scalar
                        # print("joint angle")
                        # print(joint_angle_bone)
                        joint_angle_array.append(joint_angle_bone)
                        prev_bone_direction = curr_bone_direction         
                    # loop ends when all bone joint angles on each finger gets computed
            
            # model execution starts here
            # input joint angle bones into a PCA
            latent = np.dot(joint_angle_array, self.pca_matrix)
            print(latent)
            # print(latent.shape)

            # fetch latent into autoencoder 
            # result = model(latent)
            # print(result)

            # print("joint_angle_array")
            # print(joint_angle_array)            
            # with open('./joint_angle_data.csv','a') as fd:
                # writer = csv.writer(fd)
                # writer.writerow(joint_angle_array)

        # Get tools
        for tool in frame.tools:
            print("")
            # print("  Tool id: %d, position: %s, direction: %s" % (
                # tool.id, tool.tip_position, tool.direction))

        # Get gestures
        for gesture in frame.gestures():
            if gesture.type == Leap.Gesture.TYPE_CIRCLE:
                circle = CircleGesture(gesture)

                # Determine clock direction using the angle between the pointable and the circle normal
                if circle.pointable.direction.angle_to(circle.normal) <= Leap.PI/2:
                    clockwiseness = "clockwise"
                else:
                    clockwiseness = "counterclockwise"

                # Calculate the angle swept since the last frame
                swept_angle = 0
                if circle.state != Leap.Gesture.STATE_START:
                    previous_update = CircleGesture(controller.frame(1).gesture(circle.id))
                    swept_angle =  (circle.progress - previous_update.progress) * 2 * Leap.PI

                # print("  Circle id: %d, %s, progress: %f, radius: %f, angle: %f degrees, %s" % (
                        # gesture.id, self.state_names[gesture.state],
                        # circle.progress, circle.radius, swept_angle * Leap.RAD_TO_DEG, clockwiseness))

            if gesture.type == Leap.Gesture.TYPE_SWIPE:
                swipe = SwipeGesture(gesture)
                # print("  Swipe id: %d, state: %s, position: %s, direction: %s, speed: %f" % (
                        # gesture.id, self.state_names[gesture.state],
                        # swipe.position, swipe.direction, swipe.speed))

            if gesture.type == Leap.Gesture.TYPE_KEY_TAP:
                keytap = KeyTapGesture(gesture)
                # print("  Key Tap id: %d, %s, position: %s, direction: %s" % (
                        # gesture.id, self.state_names[gesture.state],
                        # keytap.position, keytap.direction ))

            if gesture.type == Leap.Gesture.TYPE_SCREEN_TAP:
                screentap = ScreenTapGesture(gesture)
                # print("  Screen Tap id: %d, %s, position: %s, direction: %s" % (
                        # gesture.id, self.state_names[gesture.state],
                        # screentap.position, screentap.direction ))

        if not (frame.hands.is_empty and frame.gestures().is_empty):
            print("")

    def state_string(self, state):
        if state == Leap.Gesture.STATE_START:
            return "STATE_START"

        if state == Leap.Gesture.STATE_UPDATE:
            return "STATE_UPDATE"

        if state == Leap.Gesture.STATE_STOP:
            return "STATE_STOP"

        if state == Leap.Gesture.STATE_INVALID:
            return "STATE_INVALID"

def main():
    # Create a sample listener and controller
    listener = JointAngleListener()
    controller = Leap.Controller()

    # Have the sample listener receive events from the controller
    controller.add_listener(listener)

    # Keep this process running until Enter is pressed
    print("Press Enter to quit...")
    try:
        sys.stdin.readline()
    except KeyboardInterrupt:
        pass
    finally:
        # Remove the sample listener when done
        controller.remove_listener(listener)


if __name__ == "__main__":
    main()
