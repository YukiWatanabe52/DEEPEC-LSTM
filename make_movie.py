# -*- coding: utf-8 -*-

import numpy as np

import os, sys, cv2, glob, csv, re
import pandas as pd


def make_movie(name):
    result_labels = []
    

    #csv read            
    f = open('result/cnn/Shiraishi/ep10/result.csv'.format(name), 'r')
    reader = csv.reader(f)
    labels = []
    for row in reader:
        labels.append(row[1])
    f.close()


    #movie
    video_path = 'dataset/movies/{0}.mp4'.format(name)

    cap = cv2.VideoCapture(video_path)
    frame_num = 0    
    frames = []

    while(cap.isOpened()):
        #フレームを取得
        ret, frame = cap.read()

        if frame is None:
            break

        if frame_num > 3000:
            break

        if frame_num < 1350:
            frame_num += 1
            continue


        sys.stdout.write("\r%d" % frame_num)
        sys.stdout.flush()


        frames.append(frame)
        frame_num += 1
        
    cap.release()
        
    make(frames,labels,name)
 


def make(frames,labels,name):
        
    #video_writer
    new_video_path = '{0}MJPG.avi'.format(name)
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    writer = cv2.VideoWriter(new_video_path, fourcc, 30.0, (1280, 720))

    for i in range(len(frames)):
        frame = frames[i]
        sys.stdout.write("\rwriting... %d / %d" % (i, len(frames)))
        sys.stdout.flush()

        if i < len(labels) :
            if float(labels[i+1350]) > 0.4 :
                #cv2.putText(frame,'0',(100,100),cv2.FONT_HERSHEY_PLAIN, 10, (50,50,50),10,cv2.LINE_AA)
                cv2.putText(frame,'Eye Contacting!!',(100,100),cv2.FONT_HERSHEY_PLAIN, 7, (255,0,0),10,cv2.LINE_AA)

            #else:
                #cv2.putText(frame,'*',(100,100),cv2.FONT_HERSHEY_PLAIN, 10, (50,50,50),10,cv2.LINE_AA)
                #cv2.putText(frame,'1',(200,100),cv2.FONT_HERSHEY_PLAIN, 10, (50,50,50),10,cv2.LINE_AA)
        #else:
                #cv2.putText(frame,'0',(100,100),cv2.FONT_HERSHEY_PLAIN, 10, (255,255,255),10,cv2.LINE_AA)
                #cv2.putText(frame,'1',(200,100),cv2.FONT_HERSHEY_PLAIN, 10, (50,50,50),10,cv2.LINE_AA)

        #cv2.putText(frame,'{0}'.format(i+3000),(50,600),cv2.FONT_HERSHEY_PLAIN, 5, (255,255,255),5,cv2.LINE_AA)

        writer.write(frame)
      


if __name__ == '__main__':
    modelStr = 'DEEPEC-NP'

    make_movie('Shiraishi')


    
