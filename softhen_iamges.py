# Soften images 

import numpy as np
import cv2 
from matplotlib import pyplot as plt
from PIL import Image, ImageFilter
import os

# for fileName in os.listdir('./unaligned_faces'):
# 	for fileName1 in os.listdir('./unaligned_faces/'+fileName):
# 		print(fileName1)	
# 		image = cv2.imread('./unaligned_faces/'+fileName+'/'+fileName1) # reads the image
# 		print(len(image))

# 		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# 		image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)



# 		def convolutionImage(topLeft, topMiddle, topRight, middleLeft, middle, middleRight, bottomLeft, bottomMiddle, bottomRight):
# 			arr = []
# 			# print(topLeft, topMiddle, topRight, '\n' ,middleLeft, middle, middleRight, '\n' ,bottomLeft, bottomMiddle, bottomRight)
# 			first = (int(topLeft[0]) * 0) + (int(topMiddle[0]) * 1) + (int(topRight[0]) * 0) + (int(middleLeft[0]) * 1) + (int(middle[0]) * 4) + (int(middleRight[0]) * 1) + (int(bottomLeft[0]) * 0) + (int(bottomMiddle[0]) * 1 ) + (int(bottomRight[0]) * 0)
# 			# print("first ==> ", first)
# 			first = round(first / 8)
# 			# print("first / 8==> ", first)
# 			arr.append(first)
# 			# arr.append((topLeft[0] + topMiddle[0] + topRight[0] + middleLeft[0] + middle[0] + middleRight[0] + bottomLeft[0] + bottomMiddle[0] + bottomRight[0])/ 9)
# 			second = (int(topLeft[1])  * 0 ) + (int(topMiddle[1])  * 1 ) + (int(topRight[1])  * 0 ) + (int(middleLeft[1])  * 1 ) + (int(middle[1])  * 4 ) + (int(middleRight[1])  * 1 ) + (int(bottomLeft[1])  * 0 ) + (int(bottomMiddle[1])  * 1 ) + (int(bottomRight[1])  * 0 )
# 			# print("second ==> ", second)
# 			second = round(second / 8)
# 			# print("second / 8==> ", second)

# 			arr.append(second)
# 			third = (int(topLeft[2])  * 0 ) + (int(topMiddle[2])  * 1 ) + (int(topRight[2])  * 0 ) + (int(middleLeft[2])  * 1 ) + (int(middle[2])  * 4 ) + (int(middleRight[2])  * 1 ) + (int(bottomLeft[2])  * 0 ) + (int(bottomMiddle[2])  * 1 ) + (int(bottomRight[2])  * 0 )
# 			# print("third ==> ", third)
# 			third = round(third / 8)
# 			# print("third / 8==> ", third )
# 			arr.append(third)
# 			# print(arr)		
# 			return arr


# 		image1 = image.copy();


# 		i = 0

# 		# i ==> row
# 		# j ==> coloumn
# 		i = 0
# 		count = 0
# 		for i in range(len(image1)):
# 			# print("row ===> ", i)	
# 			j = 0
# 			finalSt = ''
# 			for j in range(len(image1[i])): 
# 				if(i != 0) and (i != len(image1) - 1) and (j != 0) and (j != len(image1[i]) - 1) :
# 					image1[i][j] = convolutionImage(image1[i-1][j-1], image1[i-1][j], image1[i-1][j+1], 
# 						image1[i][j-1], image1[i][j], image1[i][j+1],
# 						image1[i+1][j-1], image1[i+1][j], image1[i+1][j+1])
# 					count = count + 1			

# 				j = j + 1
# 			# print("\n")


# 		# print(image1[0][699], len(image1[0]), len(image1))


# 		print("i ==> ", i, " j ===> ", j)



# 		image = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)
# 		# image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

# 		image1 = cv2.cvtColor(image1, cv2.COLOR_HSV2RGB)
# 		# image1 = cv2.cvtColor(image1, cv2.COLOR_RGB2BGR)

# 		cv2.imwrite('./unaligned_faces/'+fileName+'/'+fileName1, image1)





		# image = cv2.imread('./unaligned_faces/'+fileName+'/'+fileName1) # reads the image
		# print(len(image))

		# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		# image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)



def convolutionImage(topLeft, topMiddle, topRight, middleLeft, middle, middleRight, bottomLeft, bottomMiddle, bottomRight):
	arr = []
	# print(topLeft, topMiddle, topRight, '\n' ,middleLeft, middle, middleRight, '\n' ,bottomLeft, bottomMiddle, bottomRight)
	first = (int(topLeft[0]) * 0) + (int(topMiddle[0]) * 1) + (int(topRight[0]) * 0) + (int(middleLeft[0]) * 1) + (int(middle[0]) * 4) + (int(middleRight[0]) * 1) + (int(bottomLeft[0]) * 0) + (int(bottomMiddle[0]) * 1 ) + (int(bottomRight[0]) * 0)
	# print("first ==> ", first)
	first = round(first / 8)
	# print("first / 8==> ", first)
	arr.append(first)
	# arr.append((topLeft[0] + topMiddle[0] + topRight[0] + middleLeft[0] + middle[0] + middleRight[0] + bottomLeft[0] + bottomMiddle[0] + bottomRight[0])/ 9)
	second = (int(topLeft[1])  * 0 ) + (int(topMiddle[1])  * 1 ) + (int(topRight[1])  * 0 ) + (int(middleLeft[1])  * 1 ) + (int(middle[1])  * 4 ) + (int(middleRight[1])  * 1 ) + (int(bottomLeft[1])  * 0 ) + (int(bottomMiddle[1])  * 1 ) + (int(bottomRight[1])  * 0 )
	# print("second ==> ", second)
	second = round(second / 8)
	# print("second / 8==> ", second)

	arr.append(second)
	third = (int(topLeft[2])  * 0 ) + (int(topMiddle[2])  * 1 ) + (int(topRight[2])  * 0 ) + (int(middleLeft[2])  * 1 ) + (int(middle[2])  * 4 ) + (int(middleRight[2])  * 1 ) + (int(bottomLeft[2])  * 0 ) + (int(bottomMiddle[2])  * 1 ) + (int(bottomRight[2])  * 0 )
	# print("third ==> ", third)
	third = round(third / 8)
	# print("third / 8==> ", third )
	arr.append(third)
	# print(arr)		
	return arr


	image1 = image.copy();


	i = 0

def soften_image(image1):
	# i ==> row
	# j ==> coloumn
	i = 0
	count = 0
	for i in range(len(image1)):
		# print("row ===> ", i)	
		j = 0
		finalSt = ''
		for j in range(len(image1[i])): 
			if(i != 0) and (i != len(image1) - 1) and (j != 0) and (j != len(image1[i]) - 1) :
				image1[i][j] = convolutionImage(image1[i-1][j-1], image1[i-1][j], image1[i-1][j+1], 
					image1[i][j-1], image1[i][j], image1[i][j+1],
					image1[i+1][j-1], image1[i+1][j], image1[i+1][j+1])
				count = count + 1			
				
				j = j + 1
		# print("\n")


		# print(image1[0][699], len(image1[0]), len(image1))


	print("i ==> ", i, " j ===> ", j)
	return image1


		# image = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)
		# # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

		# image1 = cv2.cvtColor(image1, cv2.COLOR_HSV2RGB)
		# # image1 = cv2.cvtColor(image1, cv2.COLOR_RGB2BGR)

		# cv2.imwrite('./unaligned_faces/'+fileName+'/'+fileName1, image1)

