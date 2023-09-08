from queue import Empty
import pandas as pd
import numpy as np
import cv2
import re, glob, os, json, shutil, toml
from shutil import copyfile
from skimage import data, filters
import random
import matplotlib.pyplot as plt
from scipy.stats import skew
from numpy.linalg import norm




def read_textfile(filepath):
    read_file = pd.read_csv (filepath,header=None)
    read_file.columns= ['video_id', 'frame', 'track_id', 'bb_left', 'bb_top', 'bb_width', 'bb_height', 'class']
    return read_file
"""
def vis_data(video_folder, anno_csv):
    vid_path = glob.glob(video_folder + '/*')
    nu = 0
    for vidpath in vid_path:
        vidnam = int(vidpath.split('\\')[-1].split('.')[0])
        anno_df = anno_csv[anno_csv['video_id'] == vidnam]

        # Load video
        cap = cv2.VideoCapture(vidpath)
        colors = [(255, 0, 0), (255, 255, 0), (0, 255, 255), (0, 255, 0), (0, 0, 255), (10, 150, 130), (102, 25, 85),
                  (74, 36, 200)]

        while cap.isOpened():
            res, img = cap.read()
            if img is None:
                break
            h, w, c = img.shape
            frame_num = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            sub_df = anno_df[anno_df['frame'] == frame_num]
            x = sub_df['bb_left'].values
            y = sub_df['bb_top'].values
            width = sub_df['bb_width'].values
            height = sub_df['bb_height'].values
            cls = sub_df['class'].values

            img = cv2.putText(img, 'cls 1: motorbike', (350, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1,
                              cv2.LINE_AA)
            img = cv2.putText(img, 'cls 2: DHelmet', (350, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1,
                              cv2.LINE_AA)
            img = cv2.putText(img, 'cls 3: DNoHelmet', (350, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1,
                              cv2.LINE_AA)
            img = cv2.putText(img, 'cls 4: P1Helmet', (350, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1,
                              cv2.LINE_AA)
            img = cv2.putText(img, 'cls 5: P1NoHelmet', (350, 350), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1,
                              cv2.LINE_AA)
            img = cv2.putText(img, 'cls 6: P2Helmet', (350, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1,
                              cv2.LINE_AA)
            img = cv2.putText(img, 'cls 7: P2NoHelmet', (350, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1,
                              cv2.LINE_AA)

            for i, j, w, h, m in zip(x, y, width, height, cls):
                pt1 = (int(i), int(j))
                pt2 = (int(i + w), int(j + h))
                img = cv2.rectangle(img, pt1, pt2, colors[int(m) - 1], 4)
                img = cv2.putText(img, 'cls:' + str(int(m)), pt1, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1,
                                  cv2.LINE_AA)

            cv2.imwrite('vis_new_1.7_1.22/{}_{}_{}.jpg'.format(vidnam, frame_num, nu), img)
            cv2.imshow('frame', img)
            nu += 1

            if cv2.waitKey(100) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

"""
def vis_data(video_folder, anno_csv):
    vid_path=glob.glob(video_folder +'/*')
    nu=0
    for vidpath in vid_path:
        vidnam=int(vidpath.split('\\')[-1].split('.')[0])
        anno_df=anno_csv[anno_csv['video_id']==vidnam]
        #load video
        cap= cv2.VideoCapture(vidpath)
        colors=[(255,0,0),(255,255,0),(0,255,255),(0,255,0),(0,0,255),(10,150,130),(102,25,85),(74,36,200),]
        while (cap.isOpened()):
            res, img= cap.read()
            if img is None:
                break
            h,w,c=img.shape
            # print(h,w,c)
            frame_num=int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            sub_df=anno_df[anno_df['frame']==frame_num]
            x=sub_df['bb_left'].values
            y=sub_df['bb_top'].values
            w=sub_df['bb_width'].values
            h=sub_df['bb_height'].values
            cls=sub_df['class'].values


            img=cv2.putText(img, 'cls 1: motorbike', (350,150), 2, 1, (255,0,0), 1, cv2.LINE_AA)
            img=cv2.putText(img, 'cls 2: DHelmet', (350,200), 2, 1, (255,0,0), 1, cv2.LINE_AA)
            img=cv2.putText(img, 'cls 3: DNoHelmet', (350,250), 2, 1, (255,0,0), 1, cv2.LINE_AA)
            img=cv2.putText(img, 'cls 4: P1Helmet', (350,300), 2, 1, (255,0,0), 1, cv2.LINE_AA)
            img=cv2.putText(img, 'cls 5: P1NoHelmet', (350,350), 2, 1, (255,0,0), 1, cv2.LINE_AA)
            img=cv2.putText(img, 'cls 6: P2Helmet', (350,400), 2, 1, (255,0,0), 1, cv2.LINE_AA)
            img=cv2.putText(img, 'cls 7: P2NoHelmet', (350,450), 2, 1, (255,0,0), 1, cv2.LINE_AA)

            for i,j,k,l,m in zip(x,y,w,h,cls):

                img=cv2.rectangle(img,(i,j),(i+k,l+j),colors[m],4)
                img=cv2.putText(img, 'cls:'+str(m), (i+k//2,j), 1, 1, (0,0,0), 1, cv2.LINE_AA)

            #cv2.imwrite('vis/{}.png'.format(nu),img)
           # cv2.imwrite('vis_new_1.7_1.22/{}_{}_{}.jpg'.format(vidnam, frame_num, nu), img)
            cv2.imshow('frame',img)
            nu+=1
            if cv2.waitKey(100) and 0xFF == ord('q'):
                break

        # print(vidnam)
def create_dir(pth):
    try:
        os.makedirs(pth)
    except FileExistsError:
   # directory already exists
        pass

def text2yolo(filepath,width,height):
    folder='yolo_anno'
    create_dir(folder)
    anno_csv=read_textfile(filepath)
    uniq_vid=anno_csv['video_id'].unique()
    for vid_id in uniq_vid:
        vid_path=os.path.join(folder,str(vid_id))
        create_dir(vid_path)
        sub_dfs=anno_csv[anno_csv['video_id']==vid_id]
        uniq_frame_id = sub_dfs['frame'].unique()
        for i in uniq_frame_id:
            sub_df=sub_dfs[sub_dfs['frame']==i]
            clss=sub_df['class'].values
            x1s=sub_df['bb_left'].values
            x2s=sub_df['bb_width'].values
            y1s=sub_df['bb_top'].values
            y2s=sub_df['bb_height'].values

            for cls,x1,x2n,y1,y2n in zip(clss,x1s,x2s,y1s,y2s):
                dw = 1. / width
                dh = 1. / height
                x2=x1+x2n
                y2=y1+y2n
                x = (x1 + x2) / 2.0
                y = (y1 + y2) / 2.0
                w = x2 - x1
                h = y2 - y1
                x = x * dw
                w = w * dw
                y = y * dh
                h = h * dh  

                # print(h)          

                with open(os.path.join(vid_path,str(i)+'.txt'), 'a+') as f:
                                f.write(' '.join([str(cls), str(float(x)), str(float(y)), str(float(w)), str(float(h))])+'\n')

    return


def get_images(vidpath):
    folder='vid2img'
    create_dir(folder)
    vid_path=glob.glob(vidpath)
    for vidpath in vid_path:
        vidnam=int(vidpath.split('\\')[1].split('.')[0])
        vid_path=os.path.join(folder,str(vidnam))
        create_dir(vid_path)
        cap= cv2.VideoCapture(vidpath)
        while (cap.isOpened()):
            res, img= cap.read()
            frame_num=int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            if img is None:
                break
            h,w,c=img.shape

            cv2.imwrite(vid_path+'/'+str(frame_num)+'.jpg',img)
    return


def get_images_per_time_of_day(vidpath,tme,sample):
    folder=os.path.join('vid2img',tme)
    create_dir(folder)
    vid_path=glob.glob(vidpath)
    n=0
    for vidpath in vid_path:
        vidnam=int(vidpath.split('\\')[1].split('.')[0])
        vid_path=os.path.join(folder,str(vidnam))
        create_dir(vid_path)
        cap= cv2.VideoCapture(vidpath)
        while (cap.isOpened()):
            res, img= cap.read()
            frame_num=int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            if img is None:
                break
            h,w,c=img.shape

            if n%sample==0:
                cv2.imwrite(vid_path+'/'+str(frame_num)+'.jpg',img)
            n+=1
    return



def combine_folders(img_folder,anno_folder):
    folder='automl_path'
    create_dir(folder)
    img_fld_lst=glob.glob(img_folder)
    anno_fld_lst=glob.glob(anno_folder)

    # print(img_fld_lst)
    # print(anno_fld_lst)
    rge=[*range(1,101)]
    for im,ann,nme in zip(img_fld_lst,anno_fld_lst,rge):
        names=folder+'/'+str(nme)
        create_dir(names)
        shutil.move(im,names+'/images')
        shutil.move(ann,names+'/annotations')




def toml2cvac(automl_path,dst, cls):

    if not os.path.isdir(dst):
        os.makedirs(os.path.join(dst,'images'))
    else:
        shutil.rmtree(dst)
        cdst = os.path.join(dst,'images')
        os.makedirs(cdst)
    full_path = os.path.join(automl_path, 'annotations')
    files = os.listdir(full_path)
    annotations = []
    images = []
    segmentation = []
    image_id = 0; det_id = 0; iscrowd = 0

    categories = []
    for cl in cls:
        cls_id = cls.index(cl)+1
        categories.append({"supercategory": "", "id":cls_id, "name":cl})

    img_added = []
    img_added_ = []
    for curfile in files:
        img_added_.append(curfile)
        # img_added.append(curfile) # delete
        cfilepath = os.path.join(full_path, curfile)
        data_configs = toml.load(cfilepath)
        for data_config in data_configs['objects']:
            w = (data_config['xmax'] - data_config['xmin'])
            h = (data_config['ymax'] - data_config['ymin'])
            area = w*h

            bbox = [data_config['xmin'], data_config['ymin'],
                    w, h]

            category_id = cls.index(data_config['class']) + 1
            cobj = {'segmentation': segmentation, 'category_id': category_id, 'id': det_id, 'area': area,
             'iscrowd': iscrowd, 'bbox': bbox, 'image_id': image_id}
            annotations.append(cobj)
            if not curfile in img_added:
                cimgs = {"flickr_url": "", "id": image_id, "date_captured": 0, "width": data_configs['width'],
                "license": 0, "file_name": curfile.replace('.toml',''),
                "coco_url": "", "height": data_configs['height']}
                images.append(cimgs)
                img_added.append(curfile)

            det_id += 1
        image_id+=1

    df_json = {}
    df_json['info'] = {"contributor": "", "year": "", "description": "", "version": "", "url": "", "date_created": ""}
    df_json['annotations'] = annotations
    df_json['images'] = images
    df_json['categories'] = categories
    df_json['licenses'] = [{"url": "", "id": 0, "name": ""}]
    with open(dst + '/demo.json', 'w') as fp:
        json.dump(df_json, fp)

    # ## copy images
    full_path = os.path.join(automl_path, 'images')
    img_files = os.listdir(full_path)
    # ccdst = os.path.abspath(dst)
    for img in img_files:
        src_file = os.path.join(full_path,img)
        dst_file = os.path.join(dst, 'images',img)
        
        if os.path.basename(dst_file).replace('.jpg','.toml') in img_added_:
            if not(os.path.isfile(dst_file)):
                shutil.copy(src_file,dst_file)


def cvat2yolo(json_file,img_path,i):
    with open(json_file) as f:
        json_data = json.load(f)
    
    img_data = json_data['images']
    all_annts = json_data['annotations']
    all_cls = []
    cls_names = {1:'motorbike',2:'DHelmet',3:'DNoHelmet',4:'P1Helmet',5:'P1NoHelmet',6:'P2Helmet',7:'P2NoHelmet'}

    cls_indx = [0,1,2,3,4,5,6]
    cur_cls = ['motorbike','DHelmet','DNoHelmet','P1Helmet','P1NoHelmet','P2Helmet','P2NoHelmet']
    k=i
    for img in img_data:
        cur_img, cur_id, width, height = img['file_name'],img['id'], img['width'], img['height']
        annt_match = []
        for cur_annt in all_annts:
            if(cur_annt['image_id']) == cur_id:
                annt_match.append(cur_annt)
        """ 
        full_img_path = os.path.join(img_path,"{}.jpg".format(cur_img))
        dst_dir = 'yolov5'
        inside_dir='images'
        os.makedirs(dst_dir, exist_ok=True)
        os.makedirs(inside_dir, exist_ok=True)
        shutil.copy(full_img_path,os.path.join('yolov5','images',"{}.jpg".format(cur_img)))
        """
    
        #frame = cv2.imread(full_img_path)
        for cur_match in annt_match:
            bbox = cur_match['bbox']
            cls = cls_names[cur_match['category_id']]
            if not cls in all_cls:all_cls.append(cls)
            x1,y1 = int(bbox[0]),int(bbox[1])
            x2, y2 = int(bbox[0])+int(bbox[2]),int(bbox[1]+int(bbox[3]))
    
            dw = 1. / width
            dh = 1. / height
            x = (x1 + x2) / 2.0
            y = (y1 + y2) / 2.0
            w = x2 - x1
            h = y2 - y1
            x = x * dw
            w = w * dw
            y = y * dh
            h = h * dh
            full_img_path = os.path.join(img_path, "{}.jpg".format(cur_img))
            frame = cv2.imread(full_img_path)
            #cv2.imshow("Image", frame)

            """dst_dir = 'yolov5'
            images_dir = 'images'
            labels_dir ='labels'
            out_dir ='k'

            os.makedirs(dst_dir, exist_ok=True)
            os.makedirs(images_dir, exist_ok=True)
            os.makedirs(labels_dir, exist_ok=True)
            os.makedirs(out_dir, exist_ok=True)"""
            dst_dir = 'yolov51'
            images_dir = os.path.join(dst_dir, 'images')
            labels_dir = os.path.join(dst_dir, 'labels')
            os.makedirs(images_dir, exist_ok=True)
            os.makedirs(labels_dir, exist_ok=True)
            shutil.copy(full_img_path, os.path.join(images_dir ,"{}_{}.jpg".format(str(k),cur_img)))
            txt_file_name = "{}_{}.txt".format(str(k), cur_img.replace('.jpg', ''))
            with open(os.path.join(labels_dir, txt_file_name), 'a+') as f:

            #with open(os.path.join(labels_dir,str(k)_cur_img.replace('.jpg','.txt')), 'a+') as f:
                f.write(' '.join([str(int(cls_indx[cur_cls.index(cls)])), str(float(x)), str(float(y)), str(float(w)), str(float(h))])+'\n')
                print(cur_img)
                print(str(int(cls_indx[cur_cls.index(cls)])), str(float(x)), str(float(y)), str(float(w)), str(float(h)))
                f.write('\n')


    return


def text2toml(filepath,width,height):
    folder='yolo_anno_1'
    create_dir(folder)
    anno_csv=read_textfile(filepath)
    uniq_vid=anno_csv['video_id'].unique()
    anno=['motorbike','DHelmet','DNoHelmet','P1Helmet','P1NoHelmet','P2Helmet','P2NoHelmet']
    for vid_id in uniq_vid:
        vid_path=os.path.join(folder,str(vid_id))
        create_dir(vid_path)
        sub_dfs=anno_csv[anno_csv['video_id']==vid_id]
        uniq_frame_id = sub_dfs['frame'].unique()
        for i in uniq_frame_id:
            sub_df=sub_dfs[sub_dfs['frame']==i]
            clss=sub_df['class'].values
            x1s=sub_df['bb_left'].values
            x2s=sub_df['bb_width'].values
            y1s=sub_df['bb_top'].values
            y2s=sub_df['bb_height'].values

            for cls,x1,x2n,y1,y2n in zip(clss,x1s,x2s,y1s,y2s):
                dw = 1. / width
                dh = 1. / height
                x2=x1+x2n
                y2=y1+y2n
                x = (x1 + x2) / 2.0
                y = (y1 + y2) / 2.0
                w = x2 - x1
                h = y2 - y1
                x = x * dw
                w = w * dw
                y = y * dh
                h = h * dh  

                # print(h)          

                with open(os.path.join(vid_path,str(i)+'.toml'), 'a+') as f:
                    f.write(' \n'.join(['width = '+str(width),'height = '+str(height), '[[objects]]','xmin = ' +str(int(x1)),'ymin = ' +str(int(y1)),'ymax = ' +str(int(y2)), 'xmax = ' +str(int(x2)),'class = ' +f'"{anno[cls-1]}"'+' \n\n']))          

    return

def toml2cvact(imgs,anno,dsts, cls):
    for i in range(1,101):
        dst=os.path.join(dsts,str(i))
        if not os.path.isdir(dst):
            os.makedirs(os.path.join(dst,'images'))
        else:
            shutil.rmtree(dst)
            cdst = os.path.join(dst,'images')
            os.makedirs(cdst)
        full_path = os.path.join(anno, str(i))
        files = os.listdir(full_path)
        annotations = []
        images = []
        segmentation = []
        image_id = 0; det_id = 0; iscrowd = 0

        categories = []
        for cl in cls:
            cls_id = cls.index(cl)+1
            categories.append({"supercategory": "", "id":cls_id, "name":cl})

        img_added = []
        img_added_ = []
        for curfile in files:
            img_added_.append(curfile)
            # img_added.append(curfile) # delete
            cfilepath = os.path.join(full_path, curfile)
            data_configs = toml.load(cfilepath)
            for data_config in data_configs['objects']:
                w = (data_config['xmax'] - data_config['xmin'])
                h = (data_config['ymax'] - data_config['ymin'])
                area = w*h

                bbox = [data_config['xmin'], data_config['ymin'],
                        w, h]

                category_id = cls.index(data_config['class']) + 1
                cobj = {'segmentation': segmentation, 'category_id': category_id, 'id': det_id, 'area': area,
                'iscrowd': iscrowd, 'bbox': bbox, 'image_id': image_id}
                annotations.append(cobj)
                if not curfile in img_added:
                    cimgs = {"flickr_url": "", "id": image_id, "date_captured": 0, "width": data_configs['width'],
                    "license": 0, "file_name": curfile.replace('.toml',''),
                    "coco_url": "", "height": data_configs['height']}
                    images.append(cimgs)
                    img_added.append(curfile)

                det_id += 1
            image_id+=1

        df_json = {}
        df_json['info'] = {"contributor": "", "year": "", "description": "", "version": "", "url": "", "date_created": ""}
        df_json['annotations'] = annotations
        df_json['images'] = images
        df_json['categories'] = categories
        df_json['licenses'] = [{"url": "", "id": 0, "name": ""}]
        with open(dst + '/demo.json', 'w') as fp:
            json.dump(df_json, fp)

        # ## copy images
        full_path = os.path.join(imgs, str(i))
        img_files = os.listdir(full_path)
        # ccdst = os.path.abspath(dst)
        for img in img_files:
            src_file = os.path.join(full_path,img)
            dst_file = os.path.join(dst, 'images',img)
            
            if os.path.basename(dst_file).replace('.jpg','.toml') in img_added_:
                if not(os.path.isfile(dst_file)):
                    shutil.copy(src_file,dst_file)




def sort_videos(videofolder):
    folder_name='sorted_videos'
    create_dir(folder_name)
    all_vid_names=glob.glob(videofolder+'/*')
    for vidpath in all_vid_names:
        vidnam=int(vidpath.split('\\')[-1].split('.')[0])

        #nme=folder_name+'/sort_frame/{}'.format(str(vidnam))
        nme=folder_name+'/sort_frame'
        night=os.path.join(folder_name,'night')
        day=os.path.join(folder_name,'day')
        foggy=os.path.join(folder_name,'foggy')
        create_dir(nme)
        create_dir(night)
        create_dir(day)
        create_dir(foggy)


        cap = cv2.VideoCapture(vidpath)
        input = list(range(1,int(cap.get(cv2.CAP_PROP_FRAME_COUNT))))
        n = 10000
        img_groups = [input[i:i+n] for i in range(0, len(input), n)]
        for frame_gropu in img_groups:
            samp_freq = int(len(frame_gropu)*0.1)
            # print (samp_freq)
            frameIds = np.sort(random.sample(frame_gropu,samp_freq))
            frames = []
            for fid in frameIds:
                cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
                ret, frame = cap.read()
                #print(frame)
                frames.append(frame)
                #print(len(frames))
            #print("frames" ,len(frames))
           # print(frames)
            # Calculate the median along the time axis
            # print (frames)
            medianFrame1 = np.median(frames, axis=0).astype(dtype=np.uint8)
            #th, dframe = cv2.threshold(medianFrame1, 150, 200, cv2.THRESH_BINARY)
            medianFrame = cv2.cvtColor(medianFrame1, cv2.COLOR_BGR2GRAY)
            z = np.bincount(medianFrame.flatten())

            #Variance of laplacian
            var = cv2.Laplacian(medianFrame, cv2.CV_64F).var()


            # Display median frame
            #final_frame = cv2.hconcat((medianFrame, dframe))

          
            fig,(ax1, ax2) = plt.subplots(1, 2,figsize=(10, 5))


            ax1.imshow(medianFrame1)
            ax2.hist(medianFrame.ravel(),bins=256,range=[0,256])
            ax1.set_title('Background')
            ax2.set_title('Pixel histogram')
            #z= np.bincount(medianFrame.ravel())
            y = np.bincount(medianFrame1.ravel()).argmax()
            cut=np.percentile(medianFrame.ravel(),25)
            

            if y<146:
                shutil.copy(vidpath,night)
            else:
                skewness = skew(z)
                #if abs(skewness) < 0.1:
                if var < 90:
                    shutil.copy(vidpath,foggy)
                else:
                    shutil.copy(vidpath,day)



            #print(medianFrame)
            #print("Median Frame length " ,len(medianFrame))
           # print(medianFrame.ravel())
           # print("medianFrame.ravel() length " ,len(medianFrame.ravel()))
            #print(z)
            #print(y)
            print(skewness)
            print(var)

            plt.show()

            plt.savefig(os.path.join(nme, 'vid_'+ str(vidnam) +'_{}'.format(y)+ '.jpg'))
            

            cv2.imwrite(os.path.join(nme, 'vids_'+ '_' + str(vidnam) + '.jpg'), medianFrame)

            # cv2.imshow('frame', final_frame)
            # cv2.waitKey(1)
    return

def select_training_images():
    fog='sorted_videos/foggy/*'
    day='sorted_videos/day/*'
    night='sorted_videos/night/*'

    pth_lst=[fog,day,night]
    tmes=['fog','day','night']
    for pth,tme in zip(pth_lst,tmes):
        if tme=='fog':
            get_images_per_time_of_day(pth,tme,2)
        elif tme=='night':
            get_images_per_time_of_day(pth,tme,2)
        else:
            get_images_per_time_of_day(pth,tme,5)

    return



def move_images():
    imgpath='vid2img/*'
    imgs='images'
    # create_dir(imgs)
    allimgsPath=glob.glob(imgpath)
    for pth in allimgsPath:
        folders=glob.glob(pth+'/*')
        for fld in folders:
            name=fld.split('\\')[-1]
            print(fld)
            shutil.copytree(fld,imgs+'/{}'.format(name))




def select_anno_train_images():
    tomlpath='yolo_anno_1/'
    imgpath='vid2img/*'

    allimgsPath=glob.glob(imgpath)

    for pth in allimgsPath:
        folders=glob.glob(pth+'/*')
        for fld in folders:
            name=fld.split('\\')[-1]
            newpth='selected toml files/{}'.format(name)
            create_dir(newpth)
            tmlpth=tomlpath+str(name)
            imgpthss=glob.glob(fld+'/*')
            # tomlspths=


            for imgpths in imgpthss:
                img_num=imgpths.split('\\')[-1].split('.')[0]
                anno_tml=tmlpth+'/'+str(img_num)+'.toml'
                if os.path.isfile(anno_tml):
                    shutil.copy(anno_tml,newpth)
                # print(anno_tml)


    return