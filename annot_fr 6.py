###############################################################
## 해당 파일은 로컬 cam을 이용하여 본인의 이미지를 촬영하고 저장합니다.
###############################################################

import os
import cv2
import numpy as np
import time
from tqdm import tqdm
import argparse
import glob
import shutil



#############################################################################
########################### Update Released##################################
#############################################################################

################################ ver.4 ######################################
# -   args.name 추가, default=yun
#     python annot_fr.py --name CUSTOM_NAME 으로 실행 시, 
#     CUSTOM_NAME_NUMBER.jpg, .txt 생성됨

# -   args.num 추가, default=1000

################################ ver.5 ######################################
# -   args.mode 추가, default='both'
#     python annot_fr.py --mode del 하면 삭제부분만 실행


################################ ver.6 ######################################
# -   args.mode 기능 추가, other
#     python annot_fr.py --mode other 하면 다른 사람 데이터 파일명 수정 및 annot 수정

#############################################################################
########################### 새로 추가한 부분##################################
#############################################################################


### 이미지 사이즈 입력받기
parser = argparse.ArgumentParser(description="Image size to save")

# 입력받을 인자값 등록
# parser.add_argument('--img_size', default=512, type=int, help='image size to save')
parser.add_argument('--fp', default='./Datasets/custom', help='file path')
parser.add_argument('--name', default='yun', help='Input your name')
parser.add_argument('--num', default=1000, type=int, help='maximum number of images to extract')
parser.add_argument('--mode', default='both', choices=['both', 'del', 'other'], help='both: create & delete, del: delete only, other: preprocess')
# 입력받은 인자값을 args에 저장 (type: namespace)
args = parser.parse_args()

#############################################################################
#############################################################################
#############################################################################


## 이미지 저장 폴더가 없다면 폴더 생성
ori_path = f'{args.fp}/ori'
bbox_path = f'{args.fp}/bboxed'
annot_path = f'{args.fp}/annot'
os.makedirs(ori_path, exist_ok=True)
os.makedirs(annot_path, exist_ok=True)
os.makedirs(bbox_path, exist_ok=True)


def del_noTxt():
    img_fp_list = os.listdir(ori_path)
    bboxImg_fp_list = os.listdir(bbox_path)
    annot_fp_list = os.listdir(annot_path)
    
    for file_name in tqdm(img_fp_list):
        if file_name not in bboxImg_fp_list:
            if os.path.exists(f'{ori_path}/{file_name}'): os.remove(f'{ori_path}/{file_name}')
            if os.path.exists(f'{annot_path}/{file_name.split('.jp')[0]}.txt'): os.remove(f'{annot_path}/{file_name.split('.jp')[0]}.txt')
        

## 웹캠으로 얼굴 사진을 찍어 저장하는 함수
def capture_owner_images(num_images=args.num) : ## num_images에 숫자를 입력한만큼 이미지 저장
    ## 0은 기본 웹캠
    cap = cv2.VideoCapture(0)
    ## haarcascade 알고리즘으로 얼굴 탐지
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    ## 위에서 지정한 수만큼 촬영 및 저장하는 반복문
    color = (0, 255, 0)
    
    ## 기존에 존재하는 이미지 넘버링 인계받기
    if len(os.listdir(f'{args.fp}/ori')) == 0:
        last_num = -1
    else:
        nums = [int(num.split('_')[-1].split('.jp')[0]) for num in os.listdir(f'{args.fp}/ori')]
        last_num = np.max(nums)
        
    img_count = 0

    for count in tqdm(range(num_images)):
        _, frame = cap.read()
        
        frame = cv2.flip(frame, 1)
        frame_h, frame_w = frame.shape[0], frame.shape[1]
        ## haarcascade 알고리즘은 흑백 이미지의 명암 차이로 탐지를 하는 것이기에 흑백으로 변환
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ## 변환된 프레임에서 얼굴 탐지 시도
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        idx = 0
        x_ = []
        y_ = []
        w_ = []
        h_ = []
        
        original_file = f'{args.fp}/ori/{args.name}_{count + last_num + 1}.jpg'
        cv2.imwrite(original_file, frame)
        
        ## 탐지된 얼굴에서 좌표를 가져오는 반복문
        for (x, y, w, h) in faces :
            x_.append( f'{( (x+w/2)/frame_w ):.8f}' )
            y_.append( f'{( (y+h/2)/frame_h ):.8f}' )
            w_.append( f'{( w/frame_w ):.8f}')
            h_.append( f'{( h/frame_h ):.8f}')
            ## 프레임에서 얼굴 영역만 가져온다
            
            ## 변환된 얼굴 이미지 출력하여 확인

            cv2.rectangle(frame, (x,y), (x+w, y+h), color, 2)
            cv2.imshow('Captured Face', frame)

        
        str = []
        for i in range(len(x_)):
            if (w_[i] == sorted(w_, reverse=True)[0]) and (h_[i] == sorted(h_, reverse=True)[0]):
                str.append(f'{idx} {x_[i]} {y_[i]} {w_[i]} {h_[i]}')
        
        if len(x_) == 1:
            captured_file = f"{args.fp}/bboxed/{args.name}_{count + last_num + 1}.jpg"
            cv2.imwrite(captured_file, frame)
            with open(f'{args.fp}/annot/{args.name}_{count + last_num + 1}.txt', 'w+') as f:
                f.write('\n'.join(str))
            
            img_count += 1
            
        count += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        time.sleep(0.25)

    cap.release()
    cv2.destroyAllWindows()
    
    return img_count

def modify_others(args):
    path = args.fp + '/other'
    data_list = os.listdir(path)

    our_members = ['onion', 'seojin']
    diff_members = ['yeonghyun', 'byeongjin', 'gwangha']

    for folder_name in tqdm(data_list, position=0):
        name = folder_name.split('_')[0]
        
        if name in our_members:
            ori_path = path + f'/{folder_name}/ori'

            if os.listdir(ori_path)[0].split('_')[0] == 'yun':
                filenames = glob.glob(path + f'/{folder_name}/*/*')

                for file in tqdm(filenames, position=1, leave=False):
                    ftype = file.split('\\')[-2]
                    basename_num = os.path.basename(file).split('_')[1]
                    basename = f'{name}_{basename_num}'
                    os.rename(src=file, dst=f'{path}/{folder_name}/{ftype}/{basename}')

        elif name in diff_members:
            # annot과 ori가 있으나 파일명 중복이 의심되는 경우
            ori_path = path + f'/{folder_name}/ori'

            if os.listdir(ori_path)[0].split('_')[0] != name:
                filenames = glob.glob(path + f'/{folder_name}/*/*')

                for file in tqdm(filenames, position=1, leave=False):
                    ftype = file.split('\\')[-2]
                    basename= os.path.basename(file)
                    basename = f'{name}_{basename}'
                    os.rename(src=file, dst=f'{path}/{folder_name}/{ftype}/{basename}')

def modify_annot(args):
    path = args.fp + '/other'
    db_list = os.listdir(path)

    for db in tqdm(db_list, position=0):
        print(f'Modifying labels in {db}')
        labels_fp_list = glob.glob(f'{path}/{db}/annot/*.txt')

        for labels_fp in tqdm(labels_fp_list, position=1, leave=False):
            new_lines = []
            with open(labels_fp, 'r') as f:
                lines = f.readlines()
            for i, line in enumerate(lines):
                elements = line.split(' ')
                elements[0] = '1'
                new_lines.append( ' '.join(elements) )

            filename = os.path.basename(labels_fp)
            os.makedirs(f'{path}/{db}/annot_edit', exist_ok=True)
            with open(f'{path}/{db}/annot_edit/{filename}', 'w') as f:
                f.write(''.join(new_lines))

        shutil.rmtree(f'{path}/{db}/annot')
        os.rename(src=f'{path}/{db}/annot_edit', dst=f'{path}/{db}/annot')


if args.mode == 'both':
    print('데이터 취득 중...')
    img_count = capture_owner_images()
    print('완료!')
    print(f'총 {img_count}장 생성되었습니다!')

if args.mode in ['both', 'del']:
    print('어노테이션 없는 데이터 제거 중...')
    del_noTxt()
    print('완료!')

if args.mode == 'other':
    print('다른 분들 데이터 파일명 수정 중...')
    modify_others(args)
    print('완료!')
    print('다른 분들 데이터 annotation 수정 중...')
    modify_annot(args)
    print('완료!')