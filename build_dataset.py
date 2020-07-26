import os
import numpy as np
from requests import exceptions
import argparse
import requests
import cv2
# API_KEY = "36a8c3d3f27a4d80bf90434444dbc6bd"



existingName = [];
for f in os.listdir('./unaligned_faces'): 
	# print(f)
	existingName.append(f)
# list1 =  os.walk('./unaligned_faces')
# print(list1)

newEntries = [ "Dharmendra ",
 "Sunil Dutt",
 "Vinod Khanna",
 "Mithun Chakraborty",
 "Rishi Kapoor",
 "Shatrughan Sinha",
 "Hrithik Roshan",
 "Feroz Khan",
 "Randhir Kapoor",
 "Govinda",
 "Jackie Shroff",
 "Amjad Khan",
 "Irrfan Khan",
 "Kader Khan",
 "Manoj Bajpayee ",
 "Shakti Kapoor ",
 "John Abraham",
 "Imran Khan",
 "Arjun Rampal",
 "Sanjay Mishra",
 "Farhan Akhtar",
 "Annu Kapoor",
 "Tusshar Kapoor",
	"Boman Irani",
	"Jimmy Sheirgill",
	"Neil Nitin Mukesh",
	"Aditya Roy Kapoor",
	"Sooraj Pancholi",
	"Rahul Bose",
	"Sohail Khan",
	"Sharman Joshi",
	"Omi Vaidya",
	"Karan Singh Grover",
	"Ronit Roy",
 "Rajkummar Rao",
 "Vidya Balan",
 "Kangana Ranaut",
 "Sonam Kapoor",
 "Jacqueline Fernandez",
 "Amy Jackson",
 "Rani Mukerji",
 "Kareena Kapoor",
 "Amrita Rao",
 "Ayesha Takia ",
 "Shraddha Kapoor ",
 "Sonakshi Sinha ",
 "Asin Thottumkal",
 "Mallika Sherawat",
 "Parineeti Chopra",
 "Yami Gautam",
 "Disha Patani",
 "Kajal Aggarwal",
 "Ileana D'Cruz",
 "Taapsee Pannu",
 "Kriti Sanon",
 "Ananya Panday",
 "Vaani Kapoor",
 "Zareen Khan",
 "Neha Sharma",
 "Radhika Apte",
 "Shruti Haasan",
 "Tamannaah Bhatia",
 "Kalki Koechlin",
 "Urvashi Rautela",
  "Rakul Preet Singh",
  "Kiara Advani",
  "Pooja Hegde",
  "Hema Malini",
  "Madhuri Dixit",
   "Kajol",
   "Juhi Chawla",
   "Aishwarya Rai Bachchan",
   "Raveena Tandon",
   "Kareena Kapoor",
   "Preity Zinta",
   "Shilpa Shetty Kundra",
   "Twinkle Khanna",
   "Sushmita Sen",
   "Neetu Singh",
   "Farida Jalal",
   "Scarlett Johansson",
   "Don Cheadle",
   "Benedict Cumberbatch",
   "Paul Rudd",
   "Chadwick Boseman",
   "Brie Larson",
   	"Tom Holland",
   	"Karen Gillan",
   	"Zoe Saldana",
   	"Evangeline Lilly",
   "Tessa Thompson",
   "Tessa Thompson",
   "Elizabeth Olsen",
   "Anthony Mackie",
   "Sebastian Stan",
   "Tom Hiddleston",
   	"Dave Bautista",
   	"Letitia Wright",
   	"James D'Arcy",
   		"Bradley Cooper",
   		"Josh Brolin",
   		"Yvette Nicole Brown",
   		"Daniel Radcliffe",
   		"Rupert Grint",
   		"Emma Watson",
   		"Richard Harris",
   		"Michael Gambon",
   		"Robbie Coltrane",
   		"Bonnie Wright",
   		"James Phelps",
   		"Oliver Phelps",
   		"Julie Walters",
   		"Mark Williams",
   		"Chris Rankin",
   		"Domhnall Gleeson",
   		"Geraldine Somerville",
   		"Adrian Rawlins",
   		"Ralph Fiennes",
   		"Alan Rickman",
   		 "Maggie Smith",
   		 "Tom Felton",
   		 "Josh Herdman",
   		  "Jason Isaacs",
   		  "Helen McCrory",
   		  "Helena Bonham Carter",
   		  "Gary Oldman",
		"David Thewlis",
   		   "Timothy Spall",
   		   "Brendan Gleeson",
   		   "Richard Griffiths",
   		   "Matthew Lewis",
   		   "Alfred Enoch",
   		   "John Hurt",
   		   "Zoë Wanamaker",
   		   "David Bradley",
   		   "Evanna Lynch",
   		   "Katie Leung",
   		   "Robert Pattinson",
   		   "Stanislav Yanevski",
   		   "Shefali Chowdhury",
   		   "Robert Hardy",
   		   "Imelda Staunton",
   		    "Charlotte Skeoch",
   		    "Christian Coulson",
   		    "Georgina Leonidas",
   		    "Hansika Motwani",
   		    "Joseph Vijay",
   		    "Prabhas",
   		    "Suriya",
   		    "Vikram",
   		    "Dhanush",
   		    "Mahesh Babu",
			"Allu Arjun",
			"Ram Charan",
			"Pawan Kalyan",
			"N.T. Rama Rao Jr.",
			"Ravi Teja",
			"Rana Daggubati	",
			"Mohammad Rafi",
			"Sonu Nigam",
			"Udit Narayan",
			"Lata Mangeshkar",
			"Armaan Malik",
			"A. R. Rahman",
			"Arijit Singh",
			"Abhijeet Sawant",
			"Badshah",
			"Bappi Lahiri",
			"Darshan Raval",
			"Daler Mehndi",
			"Diljit Dosanjh",
			"Gajendra Verma",
			"Guru Randhawa",
			"Himesh Reshammiya",
			"Kailash Kher",
			"Mika Singh",
			"Mohit Chauhan",
			"Sidhu Moose Wala",
			"Yo Yo Honey Singh",
			"Taylor swift",
			"NancyMomoland" ]



singleArr = np.concatenate((existingName, newEntries))

print(len(singleArr))


def Remove(singleArr): 
    final_list = [] 
    for num in singleArr: 
        if num not in final_list: 
            final_list.append(num) 
    return final_list 
      



remainDataset = Remove(singleArr)

print(len(remainDataset))


# CREATE dataset according to array formation

# construct the argument parser and parse the arguments


# construct the argument parser and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-q", "--query", required=True,
# 	help="search query to search Bing Image API for")
# ap.add_argument("-o", "--output", required=True,
# 	help="path to output directory of images")
# args = vars(ap.parse_args())
# array = ['Abhishek Bachchan','Aditya Roy Kapur','Akkineni Nagarjuna','Akshaye Khanna','Anil Kapoor','Anupam Kher','Arbaaz Khan','Arjun Kapoor','Arshad Warsi','Ayushman Khurrana','Anu Kapoor','Amrish Puri','Alok Nath','Bobby Deol','Dharam Singh Deol','Dilip Joshi','Emraan Hashmi','Himesh Reshammiya','Honey Singh','Jackie Shroff','John Abraham','Karan Johar','Kartik Aaryan','Kay Kay Menon','Nana Patekar','Naseeruddin Shah','Nawazuddin Siddiqui','Om Puri','Prabhas','Prabhu Deva','Rishi Kapoor','Rajinikanth','Rajpal Yadav','Riteish Deshmukh','Sonu Sood','Sunny Deol','Sushant Singh Rajput','Varun Dhawan','Vivek Oberoi','Sidharth Malhotra']
# array = os.listdir('/var/www/html/face-detection/ai-recognition/faceNet-yolo/unaligned_faces')

# set your Microsoft Cognitive Services API key along with (1) the
# maximum number of results for a given search and (2) the group size
# for results (maximum of 50 per request)
API_KEY = "36a8c3d3f27a4d80bf90434444dbc6bd"
MAX_RESULTS = 2
GROUP_SIZE = 1
# set the endpoint API URL
URL = "https://api.cognitive.microsoft.com/bing/v7.0/images/search"

for name in remainDataset:
	# Directory 
	directory = name
	  
	# Parent Directory path 
	parent_dir = "./279_dataset"
	parent_dir_test = "./279_dataset_test"
	# Path 
	path = os.path.join(parent_dir, directory) 
	# path_test = os.path.join(parent_dir_test, directory) 
	  
	# Create the directory 
	# 'GeeksForGeeks' in 
	# '/home / User / Documents' 
	os.mkdir(path) 
	# os.mkdir(path_test) 
	print("Directory '% s' created" % directory) 	

	# when attempting to download images from the web both the Python
	# programming language and the requests library have a number of
	# exceptions that can be thrown so let's build a list of them now
	# so we can filter on them
	EXCEPTIONS = set([IOError, FileNotFoundError,
		exceptions.RequestException, exceptions.HTTPError,
		exceptions.ConnectionError, exceptions.Timeout])

	# store the search term in a convenience variable then set the
	# headers and search parameters
	# term = args["query"]
	term = name
	headers = {"Ocp-Apim-Subscription-Key" : API_KEY}
	params = {"q": term, "offset": 0, "count": GROUP_SIZE}
	# make the search
	print("[INFO] searching Bing API for '{}'".format(term))
	search = requests.get(URL, headers=headers, params=params)
	search.raise_for_status()
	# grab the results from the search, including the total number of
	# estimated results returned by the Bing API
	results = search.json()
	# print("results =========+++>", results);
	estNumResults = min(results["totalEstimatedMatches"], MAX_RESULTS)
	print("[INFO] {} total results for '{}'".format(estNumResults,
		term))
	# initialize the total number of images downloaded thus far
	total = 0


	# loop over the estimated number of results in `GROUP_SIZE` groups
	for offset in range(0, estNumResults, GROUP_SIZE):
		# update the search parameters using the current offset, then
		# make the request to fetch the results
		print("[INFO] making request for group {}-{} of {}...".format(
			offset, offset + GROUP_SIZE, estNumResults))
		params["offset"] = offset
		search = requests.get(URL, headers=headers, params=params)
		search.raise_for_status()
		results = search.json()
		print("[INFO] saving images for group {}-{} of {}...".format(
			offset, offset + GROUP_SIZE, estNumResults))

		print("results['alue'] ========> ", results["value"])

		# loop over the results
		for v in results["value"]:
			# try to download the image
			try:
				# make a request to download the image
				print("[INFO] fetching: {}".format(v["contentUrl"]))
				r = requests.get(v["contentUrl"], timeout=30)
				# build the path to the output image
				ext = v["contentUrl"][v["contentUrl"].rfind("."):]
				# path = './finalFolder'
				if(total == 0):
					p = os.path.sep.join([ path ,directory.format(
					str(total).zfill(8), ext)])
				
				else:
					p = os.path.sep.join([ parent_dir_test ,directory.format(
					str(total).zfill(8), ext)])
				
					# p = os.path.sep.join([ path ,directory.format(
					# 	str(total).zfill(8), ext)])
				print("p ====================>",p)
				# write the image to disk
				f = open(p, "wb")
				# if p is not ""
				f.write(r.content)
				f.close()
			# catch any errors that would not unable us to download the
			# image
			except Exception as e:
				# check to see if our exception is in our list of
				# exceptions to check for
				if type(e) in EXCEPTIONS:
					print("[INFO] skipping: {}".format(v["contentUrl"]))
					continue

			# try to load the image from disk
			image = cv2.imread(p)
			# if the image is `None` then we could not properly load the
			# image from disk (so it should be ignored)
			# if image is None:
			# 	print("[INFO] deleting: {}".format(p))
			# 	os.remove(p)
			# 	continue
			# update the counter
			total += 1