import os

for count, filename in enumerate(os.listdir("./279_dataset_test")): 
	dst  = filename+".jpg"
	src ='./279_dataset_test/'+ filename 
	dst ='./279_dataset_test/'+ dst 
	  
	# rename() function will 
	# rename all the files 
	os.rename(src, dst) 