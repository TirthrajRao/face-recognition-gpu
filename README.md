# face-recognition-gpu
This repo contains the face-yolo code only but of 2nd functionality mentioned in the code.

File name according to its functionalities 

1)Face alignment
	File name		Type 

	align_dataset_yolo_gpu.py	Folder
	unaligned_faces	Folder
	det1.npy`		file		
	det2.npy`		file		
	det3.npy`		file		
	detect_dace.py	file
	align-dataset-mtcnn.py	file ( Main file to execute to align faces)

2)Align Dataset
	File name 		Type

	train_tripleloss_2.py	File ( Main File to execute to train face)
	Download pre-trained model from here https://drive.google.com/file/d/0B5MzpY9kBtDVZ2RpVDYwWmxoSUk/edit                             
	Pretrained model location must be as /models/`pre-trained-model-name`

3)Classify dataset
	File name 		Type

	make-classifier.py      (Main file to execute)
	PS: Create folder name `myclassifier` (if don't exists). 
	    Inside that create file name my_classifier.pkl (If don't exits)

4)Face-recognition
	File name 		Type

	realtime_facenet_2.py   (Main file to execute)

Note: Download any required file that is missing and also for reference refer this link. : https://github.com/AzureWoods/faceRecognition-yolo-facenet
In this it is GPU configured therefore refer readme fiel of above mentioned link.
