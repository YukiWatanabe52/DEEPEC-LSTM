import cv2
import csv
import sys
import os.path
import numpy as np
import dlib
import copy

eyes_detector = dlib.get_frontal_face_detector()
eyes_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

img_rows, img_cols = 36, 60 #画像の縦横サイズ

#GCNはdata_utilで行う

def preprocessing_img(img):
        resized = cv2.resize(img, (img_cols, img_rows))
        gray = cv2.cvtColor(resized,cv2.COLOR_BGR2GRAY)
        return gray

def clip_eye(img, parts):
    parts_x = []
    parts_y = []
    for part in parts:
        parts_x.append(part.x)
        parts_y.append(part.y)

    top = min(parts_y)
    bottom = max(parts_y)
    left = min(parts_x)
    right = max(parts_x)

    width = right - left
    height = bottom - top

    #目領域の識別誤差に対応するためのマージン
    margin_w = width * 0.4
    margin_h = height * 0.4

    x = np.random.uniform(-margin_w,margin_w)
    y = np.random.uniform(-margin_h,margin_h)

    top    = top    - margin_h + y * 0.1
    bottom = bottom + margin_h + y * 0.1
    left   = left   - margin_w + x * 0.1
    right  = right  + margin_w + x * 0.1

    #width = right - left
    #height = bottom - top

    #60:36くらいにする
    if height < width * 0.6:     #横長の場合
        top = (top + bottom) / 2 - width * 0.3
        bottm = (top + bottom) /2 + width * 0.3
    else:     #縦長の場合
        left = (left + right) / 2 - height * 0.3
        right = (left + right) /2 + height * 0.3


    return img[int(top + 0.5):int(bottom + 0.5),int(left + 0.5):int(right + 0.5)]

def detect_shape(img):
    dets = eyes_detector(img, 1)

    lefts = []
    rights = []

    #marked_img = copy.deepcopy(img)
    
    for k,d in enumerate(dets):

        shape = eyes_predictor(img, d)
        
        for shape_point_count in range(shape.num_parts):
            shape_point = shape.part(shape_point_count)

            #cv2.putText(marked_img, '.',(int(shape_point.x), int(shape_point.y)),cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1)

            if shape_point_count == 36: #左目尻
                left_position = shape_point
            if shape_point_count == 45: #右目尻
                right_position = shape_point

            if shape_point_count > 35 and shape_point_count < 42:
                lefts.append(shape_point)
            elif shape_point_count > 41 and shape_point_count < 48:
                rights.append(shape_point)

              

    if lefts: 
        return clip_eye(img,lefts), clip_eye(img,rights), left_position, right_position #, marked_img
    else:
        return None, None, None, None#, marked_img

    

def detect(video_path,width,height):
    #動画読み込み
    cap = cv2.VideoCapture(video_path)

    dir_path ,video = os.path.split(video_path)
    video_name, ext = os.path.splitext(video)


    #mkdir
    left_dir = 'dataset/images/{0}/left'.format(video_name)
    right_dir = 'dataset/images/{0}/right'.format(video_name)
    if not os.path.exists('dataset/images/{0}'.format(video_name)):
        os.mkdir('dataset/images/{0}'.format(video_name))
    if not os.path.exists('dataset/images/{0}/left'.format(video_name)):
        os.mkdir('dataset/images/{0}/left'.format(video_name))
    if not os.path.exists('dataset/images/{0}/right'.format(video_name)):
        os.mkdir('dataset/images/{0}/right'.format(video_name))

    face_positions = []

    #new_video_path = os.path.join('{0}-detect.avi'.format(video_name))
    #fourcc = cv2.VideoWriter_fourcc('M','P','4','V')
    #writer = cv2.VideoWriter(new_video_path, fourcc, 30.0, (width, height))

    frame_num = 0    
    prev_left_width = width

    while(cap.isOpened()):
        #フレームを取得
        ret, frame = cap.read()

        #if frame_num < 900:
        #    frame_num += 1
        #    face_positions.append([0.5, 0.5, 0.5, 0.5])
        #    continue
        #if frame_num > 1000:
        #    break

        if frame is None:
            break

        sys.stdout.write("\r%d" % frame_num)
        sys.stdout.flush()

        left_eye_img, right_eye_img, left_position, right_position = detect_shape(frame)
        #left_eye_img, right_eye_img, marked_frame = detect_shape(frame)

        #writer.write(marked_frame)

        #cv2.imwrite(os.path.join(original_dir,'{0}.png'.format(frame_num)),marked_frame) 

        if left_position == None:
            face_positions.append([0.5, 0.5, 0.5, 0.5])
        else:
            face_positions.append([left_position.x / width, left_position.y / height, right_position.x / width, right_position.y / height])

        if left_eye_img is None or right_eye_img is None:
            prev_left_width = width #いったん初期化
            frame_num += 1
            continue
        else:
            left_height, left_width, left_channels = left_eye_img.shape  
            right_height, right_width, right_channels = right_eye_img.shape
            if left_height > 0 and right_height > 0 and left_width > 0 and right_width > 0 and left_width < prev_left_width * 3:
                processed_left_eye_img = preprocessing_img(left_eye_img)
                processed_right_eye_img = preprocessing_img(right_eye_img)                           
                cv2.imwrite(os.path.join(left_dir,'{0}.png'.format(frame_num)), processed_left_eye_img)
                cv2.imwrite(os.path.join(right_dir,'{0}.png'.format(frame_num)), processed_right_eye_img)

                prev_left_width = left_width

        frame_num += 1
        

    np.savetxt('dataset/face_positions/{0}.csv'.format(video_name), face_positions,delimiter=',')

    cap.release()


if __name__ == '__main__':
    if len(sys.argv) < 4:
        print('not enough parms')
        exit()

    video_path = sys.argv[1]
    width = int(sys.argv[2])
    height = int(sys.argv[3])

    detect(video_path,width,height)
