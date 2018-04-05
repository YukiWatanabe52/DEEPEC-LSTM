import cv2
import sys
import csv
import os.path
import numpy as np

def writer(video_path,csv_path):   

    #動画のパス
    if not os.path.exists(video_path): 
        print('Video file is not found')
        sys.exit()
    root, video_name = os.path.split(video_path)
    video_root, ext = os.path.splitext(video_name)

    #csvがあれば読み込み
    if os.path.exists(csv_path):      
        f = open(csv_path, 'r')
        reader = csv.reader(f)
        labels = []
        for row in reader:
            for i in row:
                labels.append(int(float(i)))
    else:
        print('Csv file is not found.')
        sys.exit()

    #video_writer
    fourcc = cv2.VideoWriter_fourcc('M','P','4','V')
    writer = cv2.VideoWriter('dataset/movies/labeled/{0}.avi'.format(video_root), fourcc, 30.0, (1280, 720))


    #動画読み込み
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_num = 0
    while(cap.isOpened()):
        ret, frame = cap.read()

        if frame is None:
            break

        frame = cv2.resize(frame,(1080, 720))
         
        if labels[frame_num] == 0:
            cv2.putText(frame,'0',(50,100),cv2.FONT_HERSHEY_PLAIN, 5, (255,255,255),5,cv2.LINE_AA)
            cv2.putText(frame,'1',(120,100),cv2.FONT_HERSHEY_PLAIN, 5, (50,50,50),5,cv2.LINE_AA)
        else:
            cv2.putText(frame,'0',(50,100),cv2.FONT_HERSHEY_PLAIN, 5, (50,50,50),5,cv2.LINE_AA)
            cv2.putText(frame,'1',(120,100),cv2.FONT_HERSHEY_PLAIN, 5, (255,255,255),5,cv2.LINE_AA)
      
        writer.write(frame)

        sys.stdout.write("\r%d / %d" % (frame_num, total))
        sys.stdout.flush()        

        frame_num += 1

    cap.release()
    cv2.destroyAllWindows()


    sys.exit()
    



if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('not enough parms')
        exit()

    video_path = sys.argv[1]
    csv_path = sys.argv[2]

    writer(video_path,csv_path)
