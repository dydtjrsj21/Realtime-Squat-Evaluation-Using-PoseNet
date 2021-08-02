from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.models import load_model
import os
import sys
import cv2
import time
import argparse
import numpy as np
import keyboard
import asyncio

import pkg_resources
import tensorflow as tf
import posenet





parser = argparse.ArgumentParser()
parser.add_argument('--model', type=int, default=101)
parser.add_argument('--cam_id', type=int, default=0)
parser.add_argument('--cam_width', type=int, default=1280)
parser.add_argument('--cam_height', type=int, default=720)
parser.add_argument('--scale_factor', type=float, default=0.7125)
parser.add_argument('--file', type=str, default=None, help="Optionally use a video file instead of a live camera")
args = parser.parse_args()

stop=0
width=0
height=0
exTime=0
async def checkPressed():
    if keyboard.is_pressed('enter'):  # if key 'q' is pressed 
        global stop
        stop=1
async def counter():
    global exTime
    await asyncio.sleep(0.1)
    exTime+=0.1
    
def translate(value, leftMin, leftMax, rightMin, rightMax):
    # Figure out how 'wide' each range is
    leftSpan = leftMax - leftMin
    rightSpan = rightMax - rightMin

    # Convert the left range into a 0-1 range (float)
    valueScaled = float(value - leftMin) / float(leftSpan)

    # Convert the 0-1 range into a value in the right range.
    return rightMin + (valueScaled * rightSpan)
    
def main():
    with tf.Session() as sess:
        model = load_model('perceptron/squat_mlp_model.h5')
        model_cfg, model_outputs = posenet.load_model(args.model, sess)
        output_stride = model_cfg['output_stride']

        cap = cv2.VideoCapture(1)
        cap.set(3, args.cam_width)
        cap.set(4, args.cam_height)
        
        width=args.cam_width
        height=args.cam_height
        
        start = time.time()
        frame_count = 0
        minValue=0
        maxValue=0.9998
        count=0
        pastScore=0
        
        while True:
            loop = asyncio.get_event_loop()
            loop.run_until_complete(checkPressed())
            input_image, display_image, output_scale = posenet.read_cap(
                cap, scale_factor=args.scale_factor, output_stride=output_stride)

            heatmaps_result, offsets_result, displacement_fwd_result, displacement_bwd_result = sess.run(
                model_outputs,
                feed_dict={'image:0': input_image}
            )

            pose_scores, keypoint_scores, keypoint_coords = posenet.decode_multi.decode_multiple_poses(
                heatmaps_result.squeeze(axis=0),
                offsets_result.squeeze(axis=0),
                displacement_fwd_result.squeeze(axis=0),
                displacement_bwd_result.squeeze(axis=0),
                output_stride=output_stride,
                max_pose_detections=10,
                min_pose_score=0.15)
           
            keypoint_coords *= output_scale
            inputs=list()
            nosePoint=keypoint_coords[0][0]
            for i in keypoint_coords[0]:
                inputs.append(i[0]-nosePoint[0])
                inputs.append(i[1]-nosePoint[1])
            norm=sum([i**2 for i in inputs])
            for i in range(7,16,2):
                for j in range(2):
                    inputs.append(keypoint_coords[0][i][j]-keypoint_coords[0][i+1][j])
            inputs_normalized=np.array([])
            for i in inputs:
                inputs_normalized=np.append(inputs_normalized,i/(norm**0.5))
            inputs_normalized=np.array([inputs_normalized])
            k=model.predict(inputs_normalized)[0][0]
            if k<minValue:
                minValue=k
            if k>maxValue:
                maxValue=k    
            totalScore=translate(k,minValue,maxValue,0,10)
            global exTime
            if totalScore>8 and exTime>0.7:
                count+=1
                exTime=0
            elif totalScore>7:
                loop.run_until_complete(counter())
            else:
                exTime=0
            color=(int(totalScore*25),int(totalScore*25),int(totalScore*25))
            # TODO this isn't particularly fast, use GL for drawing and display someday...
            overlay_image = posenet.draw_skel_and_kp(
                display_image, pose_scores, keypoint_scores, keypoint_coords,
                min_pose_score=0.15, min_part_score=0.1)
            cv2.rectangle(overlay_image, (0, 0), (int(width/8), int(height)), (0, 0, 0), -1)
            cv2.rectangle(overlay_image, (int(3*width/8), 0), (int(width/2), int(height)), (0, 0, 0), -1)
            cv2.rectangle(overlay_image, (int(width/32), int(height*0.66-width/32-totalScore*10)), (int(3*width/32), int(height*0.66-width/32)), color, -1)
            cv2.putText(overlay_image,"Count : "+str(count),(int(width/32),70),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1)
            cv2.imshow('posenet', overlay_image)
            cv2.putText(overlay_image,"Score : "+("%.2f" % totalScore),(int(width/32),50),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1)
            cv2.imshow('posenet', overlay_image)
            frame_count += 1
            if (cv2.waitKey(1) & 0xFF == ord('q')) or stop:
                break

        print('Average FPS: ', frame_count / (time.time() - start))
        print(minValue,maxValue)


if __name__ == "__main__":
    main()