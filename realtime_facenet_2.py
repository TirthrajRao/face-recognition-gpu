from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from imutils import paths
import requests
import tensorflow as tf
from scipy import misc
import cv2
import numpy as np
import facenet
import detect_face
import os
import time
import pickle
from gtts import gTTS 
global str
import xlwt 
from xlwt import Workbook
import softhen_iamges

wb = Workbook() 

sheet1 = wb.add_sheet('Sheet 2')


def callApiFunction(previousPredictedFaceLabel):
    print("callApiFunction Called hurrah ")
    print("callApiFunction Called hurrah ", previousPredictedFaceLabel)
    API_ENDPOINT = "http://localhost:4000/attendance/fill-attendance"
    data = { 
        'api_from':'python',
        'api_of': "attendance_out",
        # 'api_of': "attendance_in",
        'userId': previousPredictedFaceLabel
    }
    r = requests.post(url = API_ENDPOINT, data = data)
    mytext = r.text
    language = 'en'
    myobj = gTTS(text=mytext, lang=language, slow=False) 
    myobj.save("welcome.mp3") 

    # Playing the converted file 
    os.system("mpg321 welcome.mp3") 
    # pastebin_url = r.text 
    print("The pastebin URL is:%s", r.text)  
    return r



print('Creating networks and loading parameters')
with tf.Graph().as_default():
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
    with sess.as_default():
        pnet, rnet, onet = detect_face.create_mtcnn(sess, './models/')

        minsize = 20  # minimum size of face
        threshold = [0.6, 0.7, 0.7]  # three steps's threshold
        factor = 0.709  # scale factor
        margin = 44
        frame_interval = 3
        batch_size = 1000
        image_size = 182
        input_image_size = 160

        # HumanNames = ['Andrew','Obama','ZiLin']    #train human name

        print('Loading feature extraction model')
        # modeldir = './All_trained_models/models-02/20170512-110547-02'  #pre-trained model
        modeldir = './models/20170512-110547'  #pre-trained model
        # modeldir = './models/facenet/20200415-171023'    #Self-trained model   
        facenet.load_model(modeldir)

        images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
        embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
        phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
        embedding_size = embeddings.get_shape()[1]

        # classifier_filename = './All_trained_models/myclassifier-02/my_classifier.pkl'
        classifier_filename = './myclassifier/my_classifier.pkl'
        classifier_filename_exp = os.path.expanduser(classifier_filename)
        with open(classifier_filename_exp, 'rb') as infile:
            (model, class_names) = pickle.load(infile)
            print('load classifier file-> %s' % classifier_filename_exp)
        # while True:
        #     imgResp= requests.get(url)
        #     imgNp=np.array(bytearray(imgResp.content),dtype=np.uint8)
        #     img=cv2.imdecode(imgNp,-1)

        #     # put the image on screen
        #     cv2.imshow('IPWebcam',img)
        #     if cv2.waitKey(1) == 27:
        #         break
        url='http://192.168.43.1:8080/video'
        # url='http://192.168.43.1:8080/video'
        # url='http://10.147.230.25:8080/video'
        # video_capture = cv2.VideoCapture(url)
        c = 0
        j = 1
        print('Start Recognition!')
        totalTimeSameFaceRecognised = 0
        previousPredictedFaceLabel = ''
        prevTime = 0
        # imagePaths = os.listdir("./279_dataset_test")
        imagePaths = list(paths.list_images("./279_dataset_test"))
        # imagePaths = list(paths.list_images("./freshTestingImages"))
        print("image path ==> ", imagePaths)
        for (i, imagePath) in enumerate(imagePaths):
            # imagePath = imagePath + '.jpg'
            print("image path ===>", imagePath);
            frame = imagePath
            frame = cv2.imread(frame)
            # frame = cv2.resize(frame, (0,0), fx=0.7, fy=0.7)    #resize frame (optional)
            frame = cv2.resize(frame, (700, 700))        
            # frame = softhen_iamges.soften_image(frame)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            bounding_boxes, _ = detect_face.detect_face(frame, minsize, pnet, rnet, onet, threshold, factor)
            nrof_faces = bounding_boxes.shape[0]
            print('Detected_FaceNum: %d' % nrof_faces)
            if nrof_faces > 0:
                det = bounding_boxes[:, 0:4]
                img_size = np.asarray(frame.shape)[0:2]

                # cropped = []
                # scaled = []
                # scaled_reshape = []
                bb = np.zeros((nrof_faces,4), dtype=np.int32)

                for i in range(nrof_faces):
                    emb_array = np.zeros((1, embedding_size))

                    bb[i][0] = det[i][0]
                    bb[i][1] = det[i][1]
                    bb[i][2] = det[i][2]
                    bb[i][3] = det[i][3]

                    if bb[i][0] <= 0 or bb[i][1] <= 0 or bb[i][2] >= len(frame[0]) or bb[i][3] >= len(frame):
                        print('face is inner of range!')
                        continue

                    cropped = (frame[bb[i][1]:bb[i][3], bb[i][0]:bb[i][2], :])
                    print("{0} {1} {2} {3}".format(bb[i][0], bb[i][1], bb[i][2], bb[i][3]))
                    cropped = facenet.flip(cropped, False)
                    scaled = (misc.imresize(cropped, (image_size, image_size), interp='bilinear'))
                    scaled = cv2.resize(scaled, (input_image_size,input_image_size),
                                        interpolation=cv2.INTER_CUBIC)
                    scaled = facenet.prewhiten(scaled)
                    scaled_reshape = (scaled.reshape(-1,input_image_size,input_image_size,3))
                    feed_dict = {images_placeholder: scaled_reshape, phase_train_placeholder: False}


                    emb_array[0, :] = sess.run(embeddings, feed_dict=feed_dict)

                    predictions = model.predict_proba(emb_array)
                    best_class_indices = np.argmax(predictions, axis=1)
                    best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]
                    cv2.rectangle(frame, (bb[i][0], bb[i][1]), (bb[i][2], bb[i][3]), (0, 255, 0), 2)
                    text_x = bb[i][0]
                    text_y = bb[i][3] + 20

                    # for H_i in HumanNames:
                    #     if HumanNames[best_class_indices[0]] == H_i:
                    result_names = class_names[best_class_indices[0]]
                    # strprob =  str(best_class_probabilities[0])
                    # result_names = result_names  + strprob
                    print("best_class_indices ==> ", best_class_indices, "class_names ===>", class_names)
                    print("best_class_probabilities ==> ", best_class_probabilities, "result name ===>", result_names)
                    probablity = float(best_class_probabilities[0])
                    print(" i ====>", i)
                    sheet1.write(j, 0, result_names) 
                    sheet1.write(j, 1, str(probablity)) 
                    j = j + 1
                    result_names = result_names + ' -- ' + str(probablity)
                    if (previousPredictedFaceLabel == ''):
                        previousPredictedFaceLabel = result_names
                    if (previousPredictedFaceLabel == result_names):
                        totalTimeSameFaceRecognised = totalTimeSameFaceRecognised+1
                    else:
                        totalTimeSameFaceRecognised = 0
                        previousPredictedFaceLabel = result_names
                    print("totalTimeSameFaceRecognised ========+++> ", totalTimeSameFaceRecognised) 
                    # if(totalTimeSameFaceRecognised == 20):
                    #     resposne = callApiFunction(previousPredictedFaceLabel)
                    #     print("resposnese ===========> ", resposne)
                    #print(result_names)
                    cv2.putText(frame, result_names, (text_x, text_y), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                1, (0, 0, 255), thickness=1, lineType=2)
            else:
                print('Unable to align')
            cv2.imshow(imagePath, frame)
            cv2.waitKey(0)
            # if cv2.waitKey(33) == ord('a'):
            #     print( "pressed a")    
        wb.save('xlwt example_without_soft_new.xls') 

        #Read image from folder. 
        # while True:
        #     # ret, frame = video_capture.read()
        #     # print("frame.ndim =================> ", frame.ndim)

        #     # print("frame =================> ", frame)
        #     frame = cv2.resize(frame, (0,0), fx=0.3, fy=0.3)    #resize frame (optional)

        #     # curTime = time.time()    # calc fps
        #     # timeF = frame_interval

        #     # if (c % timeF == 0):
        #     find_results = []

        #     if frame.ndim == 2:
        #         frame = facenet.to_rgb(frame)
        #     frame = frame[:, :, 0:3]
        #     #print(frame.shape[0])
        #     #print(frame.shape[1])

        #     ## Use MTCNN to get the bounding boxes
        #     bounding_boxes, _ = detect_face.detect_face(frame, minsize, pnet, rnet, onet, threshold, factor)
        #     nrof_faces = bounding_boxes.shape[0]
        #     #print('Detected_FaceNum: %d' % nrof_faces)

        #     if nrof_faces > 0:
        #         det = bounding_boxes[:, 0:4]
        #         img_size = np.asarray(frame.shape)[0:2]

        #         # cropped = []
        #         # scaled = []
        #         # scaled_reshape = []
        #         bb = np.zeros((nrof_faces,4), dtype=np.int32)

        #         for i in range(nrof_faces):
        #             emb_array = np.zeros((1, embedding_size))

        #             bb[i][0] = det[i][0]
        #             bb[i][1] = det[i][1]
        #             bb[i][2] = det[i][2]
        #             bb[i][3] = det[i][3]

        #             if bb[i][0] <= 0 or bb[i][1] <= 0 or bb[i][2] >= len(frame[0]) or bb[i][3] >= len(frame):
        #                 print('face is inner of range!')
        #                 continue

        #             # cropped.append(frame[bb[i][1]:bb[i][3], bb[i][0]:bb[i][2], :])
        #             # cropped[0] = facenet.flip(cropped[0], False)
        #             # scaled.append(misc.imresize(cropped[0], (image_size, image_size), interp='bilinear'))
        #             # scaled[0] = cv2.resize(scaled[0], (input_image_size,input_image_size),
        #             #                        interpolation=cv2.INTER_CUBIC)
        #             # scaled[0] = facenet.prewhiten(scaled[0])
        #             # scaled_reshape.append(scaled[0].reshape(-1,input_image_size,input_image_size,3))
        #             # feed_dict = {images_placeholder: scaled_reshape[0], phase_train_placeholder: False}

        #             cropped = (frame[bb[i][1]:bb[i][3], bb[i][0]:bb[i][2], :])
        #             print("{0} {1} {2} {3}".format(bb[i][0], bb[i][1], bb[i][2], bb[i][3]))
        #             cropped = facenet.flip(cropped, False)
        #             scaled = (misc.imresize(cropped, (image_size, image_size), interp='bilinear'))
        #             scaled = cv2.resize(scaled, (input_image_size,input_image_size),
        #                                 interpolation=cv2.INTER_CUBIC)
        #             scaled = facenet.prewhiten(scaled)
        #             # scaled_reshape = (scaled.reshape(-1,input_image_size,input_image_size,3))
        #             feed_dict = {images_placeholder: scaled_reshape, phase_train_placeholder: False}


        #             emb_array[0, :] = sess.run(embeddings, feed_dict=feed_dict)

        #             predictions = model.predict_proba(emb_array)
        #             best_class_indices = np.argmax(predictions, axis=1)
        #             best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]
        #             cv2.rectangle(frame, (bb[i][0], bb[i][1]), (bb[i][2], bb[i][3]), (0, 255, 0), 2)
        #             text_x = bb[i][0]
        #             text_y = bb[i][3] + 20

        #             # for H_i in HumanNames:
        #             #     if HumanNames[best_class_indices[0]] == H_i:
        #             result_names = class_names[best_class_indices[0]]
        #             # strprob =  str(best_class_probabilities[0])
        #             # result_names = result_names  + strprob
        #             print("best_class_probabilities ==> ", best_class_probabilities[0], "result name ===>", result_names)
        #             if (previousPredictedFaceLabel == ''):
        #                 previousPredictedFaceLabel = result_names
        #             if (previousPredictedFaceLabel == result_names):
        #                 totalTimeSameFaceRecognised = totalTimeSameFaceRecognised+1
        #             else:
        #                 totalTimeSameFaceRecognised = 0
        #                 previousPredictedFaceLabel = result_names
        #             print("totalTimeSameFaceRecognised ========+++> ", totalTimeSameFaceRecognised) 
        #             # if(totalTimeSameFaceRecognised == 20):
        #             #     resposne = callApiFunction(previousPredictedFaceLabel)
        #             #     print("resposnese ===========> ", resposne)
        #             #print(result_names)
        #             cv2.putText(frame, result_names, (text_x, text_y), cv2.FONT_HERSHEY_COMPLEX_SMALL,
        #                         1, (0, 0, 255), thickness=1, lineType=2)
        #     else:
        #         print('Unable to align')

        #     # sec = curTime - prevTime
        #     # prevTime = curTime
        #     # fps = 1 / (sec)
        #     # str = 'FPS: %2.3f' % fps
        #     # text_fps_x = len(frame[0]) - 150
        #     # text_fps_y = 20
        #     # cv2.putText(frame, str, (text_fps_x, text_fps_y),
        #     #             cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 0), thickness=1, lineType=2)
        #     # # c+=1

        #     cv2.imshow('Video', frame)

        #     if cv2.waitKey(1) & 0xFF == ord('q'):
        #         break

        # video_capture.release()
        # #video writer
        # out.release()
        cv2.destroyAllWindows()