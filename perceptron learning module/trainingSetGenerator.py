import os
import sys
import tensorflow as tf
import cv2
import time
import argparse
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import posenet
import pickle
import numpy as np
import keyboard
import asyncio



parser = argparse.ArgumentParser()
parser.add_argument('--model', type=int, default=101)
parser.add_argument('--cam_id', type=int, default=0)
parser.add_argument('--cam_width', type=int, default=1280)
parser.add_argument('--cam_height', type=int, default=720)
parser.add_argument('--scale_factor', type=float, default=0.7125)
parser.add_argument('--file', type=str, default=None, help="Optionally use a video file instead of a live camera")
args = parser.parse_args()

keyPressed=0
stop=0

async def checkPressed():
    if keyboard.is_pressed('s'):  # if key 'q' is pressed 
        global keyPressed
        keyPressed=abs(keyPressed-1)
        print(keyPressed)
    elif keyboard.is_pressed('enter'):
        global stop
        stop=1

f = open("trainingSet/positions3.csv", 'w')

            
def main():  
    with tf.Session() as sess:
        model_cfg, model_outputs = posenet.load_model(args.model, sess)
        output_stride = model_cfg['output_stride']
        cap = cv2.VideoCapture(1)
        cap.set(3, args.cam_width)
        cap.set(4, args.cam_height)
        width=args.cam_width
        height=args.cam_height
        start = time.time()
        frame_count = 0
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
                inputs.append(i[1]-nosePoint[1]) #.....k
            norm=sum([i**2 for i in inputs])
            for i in range(7,16,2):
                for j in range(2):
                    inputs.append(keypoint_coords[0][i][j]-keypoint_coords[0][i+1][j])
            inputs_normalized = [i/(norm**0.5) for i in inputs]
            
            for i in inputs_normalized:
                f.write(str(i)+",")
            f.write(str(keyPressed))
            f.write("\n")
            
            # TODO this isn't particularly fast, use GL for drawing and display someday...
            overlay_image = posenet.draw_skel_and_kp(
                display_image, pose_scores, keypoint_scores, keypoint_coords,
                min_pose_score=0.15, min_part_score=0.1)

            cv2.imshow('posenet', overlay_image)
            frame_count += 1
            if (cv2.waitKey(1) & 0xFF == ord('q')) or stop:
                break

        print('Average FPS: ', frame_count / (time.time() - start))


if __name__ == "__main__":
    main()