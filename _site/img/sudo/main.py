import cv2
import numpy as np
import sys
import time
import math
import random as rn
import copy
from picamera.array import PiRGBArray
from picamera import PiCamera
from sudoku_solve import solver
import os
import threading
import led

def speak(current):
   p = current.reshape(1,81)
   for i in p:
            for j in i:
               os.system("espeak  -w out.wav {0} && aplay out.wav".format(j))
class puzzleStatusClass:
    # this class defines the actual mathematical properties of the sudoku puzzle
    # and the associated methods used to solve the actual sudoku puzzle
    def __init__(self):
        # .current is a 9x9 grid of all solved values for the puzzle
        self.current = np.zeros((9, 9), np.uint8)
        self.currentBackup = np.zeros((9, 9), np.uint8)
        # .last is used to compare to .current to evaluate whether two consecutive OCR results match
        self.last = np.zeros((9, 9), np.uint8)
        # .orig is used to store the state of .current that is obtained from OCR,
        # but before solving for any new values
        self.orig = np.zeros((9, 9), np.uint8)
        # .solve starts off by containing 1-9 in a 9 by 9 grid,
        # by process of elimination .solve will produce the final solution
        self.solve = [[[1, 2, 3, 4, 5, 6, 7, 8, 9] for x in range(9)] for y in range(9)]
        self.solveBackup = []
        # .change is True when the solver algorithm has made a change to .solve
        self.change = True
        # .guess is True when the solver has given up on analytical techniques
        # and has begun randomly guessing at the solution
        self.guess = False



    def checkSolution(self):
        # check puzzle using three main rules
        err = 0  # error code
        # 1) no number shall appear more than once in a row
        for x in range(9):  # for each row
            # count how many of each number exists
            check = np.bincount(self.current[x, :])
            for i in range(len(check)):
                if i == 0:
                    if check[i] != 0:
                        err = 1  # incomplete, when the puzzle is complete no zeros should exist
                else:
                    if check[i] > 1:
                        err = -1  # incorrect, there can't be more than one of any number
                        print "ERROR in row ", x, " with ", i
                        return err
        # 2) no number shall appear more than once in a column
        for y in range(9):  # for each column
            check = np.bincount(self.current[:, y])
            for i in range(len(check)):
                if i == 0:
                    if check[i] != 0:
                        err = 1  # incomplete
                else:
                    if check[i] > 1:
                        err = -1  # incorrect
                        print "ERROR in col ", y, " with ", i
                        return err
        # 3) no number shall appear more than once in a 3x3 cell
        for x in range(3):
            for y in range(3):
                check = np.bincount(self.current[x * 3:x * 3 + 3, y * 3:y * 3 + 3].flatten())
                for i in range(len(check)):
                    if i == 0:
                        if check[i] != 0:
                            err = 1  # incomplete
                    else:
                        if check[i] > 1:
                            err = -1  # incorrect
                            print "ERROR in box ", x, y, " with ", i
                            return err
        return err



class imageClass:
    #this class defines all of the important image matrices, and information about the images.
    #also the methods associated with capturing input, displaying the output,
    #and warping and transforming any of the images to assist with OCR
    def __init__(self):
        #.captured is the initially captured image
        self.captured = []
        #.gray is the grayscale captured image
        self.gray = []
        #.thres is after adaptive thresholding is applied
        self.thresh = []
        #.contours contains information about the contours found in the image
        self.contours = []
        #.biggest contains a set of four coordinate points describing the
        #contours of the biggest rectangular contour found
        self.biggest = None;
        #.maxArea is the area of this biggest rectangular found
        self.maxArea = 0
        #.output is an image resulting from the warp() method
        self.output = []
        self.outputBackup = []
        self.outputGray = []
        #.mat is a matrix of 100 points found using a simple gridding algorithm
        #based on the four corner points from .biggest
        self.mat = np.zeros((100,2),np.float32)
        #.reshape is a reshaping of .mat
        self.reshape = np.zeros((100,2),np.float32)
        
    def captureImage(self,status):
        #captures the image and finds the biggest rectangle
	camera = PiCamera()
	camera.resolution = (480, 400)
	camera.framerate = 32
	rawCapture = PiRGBArray(camera, size=(480, 400))
 
# allow the camera to warmup
	time.sleep(0.1)
	#try:
	for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
            #rgb, _ = freenect.sync_get_video()
            #cv2.namedWindow("SUDOKU Solver")
            #vc = cv2.VideoCapture(1)
            #if vc.isOpened():  # try to get the first frame
             #   rval, rgb = vc.read()
            #else:
             #   rval = False
            #bgr = cv2.cvtColor(rgb, cv2.COLOR_BGR2R
		image = frame.array
		#cv2.imshow("Frame", image)
		#key = cv2.waitKey(1) & 0xFF
		self.captured = image
		rawCapture.truncate(0)
        #except TypeError:
            #print "No Kinect Detected!"
            #print "Loading sudoku.jpg..."
            # for testing purposes
            #img = cv2.imread("sudoku_test3.png")
            #self.captured = cv2.resize(img, (600, 600))

		# convert to grayscale
		self.gray = cv2.cvtColor(self.captured, cv2.COLOR_BGR2GRAY)
		print "gray"
		# noise removal with gaussian blur
		self.gray = cv2.GaussianBlur(self.gray, (5, 5), 0)
		# then do adaptive thresholding
		self.thresh = cv2.adaptiveThreshold(self.gray, 255, 1, 1, 11, 2)

		# find countours in threshold image
		self.contours, hierarchy = cv2.findContours(self.thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

		# evaluate all blobs to find blob with biggest area
		# biggest rectangle in the image must be sudoku square
		self.biggest = None
		self.maxArea = 0
		for i in self.contours:
		    area = cv2.contourArea(i)
		    if area > 50000:  # 50000 is an estimated value for the kind of blob we want to evaluate
		        peri = cv2.arcLength(i, True)   #return the perimeter of the contour
		        approx = cv2.approxPolyDP(i, 0.02 * peri, True)
		        if area > self.maxArea and len(approx) == 4:
		            self.biggest = approx
		            self.maxArea = area
		            best_cont = i
	        if status.detect == 20:
                   led.turn_on()
		if status.detect == 25:
		    status.puzzleFound = True
		    print "Sudoku puzzle detected!"
		if self.maxArea > 0:
		    status.noDetect = 0  # reset
		    status.detect += 1
		    # draw self.biggest approx contour
		    if status.completed:
		        cv2.polylines(self.captured, [self.biggest], True, (0, 255, 0), 3)
		    elif status.puzzleFound:
		        cv2.polylines(self.captured, [self.biggest], True, (0, 255, 255), 3)
		    else:
		        cv2.polylines(self.captured, [self.biggest], True, (0, 0, 255), 3)
		    self.reorder()  # reorder self.biggest
		else:
		    status.noDetect += 1
		    if status.noDetect == 20:
                        led.turn_off()
		        print "No sudoku puzzle detected!"
		    if status.noDetect > 50:
		        #status.restart = True
			pass
		"""if status.detect == 25:
		    status.puzzleFound = True
		    print "Sudoku puzzle detected!"""
		if status.beginSolver == False or self.maxArea == 0:
		    cv2.imshow('sudoku', self.captured)
		    key = cv2.waitKey(10)
		    if key == 27:
		        sys.exit()
		if status.puzzleFound == True:
			camera.close()
			break
		

            
    def reorder(self):
        #reorders the points obtained from finding the biggest rectangle
        #[top-left, top-right, bottom-right, bottom-left]
        a = self.biggest.reshape((4,2))
        b = np.zeros((4,2),dtype = np.float32)
     
        add = a.sum(1)
        b[0] = a[np.argmin(add)] #smallest sum
        b[2] = a[np.argmax(add)] #largest sum
             
        diff = np.diff(a,axis = 1) #y-x
        b[1] = a[np.argmin(diff)] #min diff
        b[3] = a[np.argmax(diff)] #max diff
        self.biggest = b

    def perspective(self):
        #create 100 points using "biggest" and simple gridding algorithm,
        #these 100 points define the grid of the sudoku puzzle
        #topLeft-topRight-bottomRight-bottomLeft = "biggest"
        b = np.zeros((100,2),dtype = np.float32)
        c_sqrt=10
        if self.biggest.any == None:
            self.biggest = [[0,0],[640,0],[640,480],[0,480]]
        tl,tr,br,bl = self.biggest[0],self.biggest[1],self.biggest[2],self.biggest[3]
        for k in range (0,100):
            i = k%c_sqrt
            j = k/c_sqrt
            ml = [tl[0]+(bl[0]-tl[0])/9*j,tl[1]+(bl[1]-tl[1])/9*j]
            mr = [tr[0]+(br[0]-tr[0])/9*j,tr[1]+(br[1]-tr[1])/9*j]
##            self.mat[k,0] = ml[0]+(mr[0]-ml[0])/9*i
##            self.mat[k,1] = ml[1]+(mr[1]-ml[1])/9*i
            self.mat.itemset((k,0),ml[0]+(mr[0]-ml[0])/9*i)
            self.mat.itemset((k,1),ml[1]+(mr[1]-ml[1])/9*i)
        self.reshape = self.mat.reshape((c_sqrt,c_sqrt,2))

    def warp(self):
        #take distorted image and warp to flat square for clear OCR reading
        mask = np.zeros((self.gray.shape),np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(11,11))
        close = cv2.morphologyEx(self.gray,cv2.MORPH_CLOSE,kernel)
        division = np.float32(self.gray)/(close)
        result = np.uint8(cv2.normalize(division,division,0,255,cv2.NORM_MINMAX))
        result = cv2.cvtColor(result,cv2.COLOR_GRAY2BGR)
        output = np.zeros((450,450,3),np.uint8)
        c_sqrt=10
        for i,j in enumerate(self.mat):
            ri = i/c_sqrt
            ci = i%c_sqrt
            if ci != c_sqrt-1 and ri != c_sqrt-1:
                source = self.reshape[ri:ri+2, ci:ci+2 , :].reshape((4,2))
                dest = np.array( [ [ci*450/(c_sqrt-1),ri*450/(c_sqrt-1)],[(ci+1)*450/(c_sqrt-1),
                            ri*450/(c_sqrt-1)],[ci*450/(c_sqrt-1),(ri+1)*450/(c_sqrt-1)],
                            [(ci+1)*450/(c_sqrt-1),(ri+1)*450/(c_sqrt-1)] ], np.float32)
                trans = cv2.getPerspectiveTransform(source,dest)
                warp = cv2.warpPerspective(result,trans,(450,450))
                output[ri*450/(c_sqrt-1):(ri+1)*450/(c_sqrt-1) , ci*450/(c_sqrt-1):(ci+1)*450/
                       (c_sqrt-1)] = warp[ri*450/(c_sqrt-1):(ri+1)*450/(c_sqrt-1) ,
                        ci*450/(c_sqrt-1):(ci+1)*450/(c_sqrt-1)].copy()
        output_backup = np.copy(output)
        cv2.imshow('output',output)
        key = cv2.waitKey(1)
        self.output = output
        self.outputBackup = output_backup

    
    def virtualImage(self, puzzle,status):
        # output known sudoku values to the real image
        j = 0
        tsize = (math.sqrt(self.maxArea)) / 400
        w = int(20 * tsize)
        h = int(25 * tsize)
        for i in range(100):
            ##            x = int(self.mat[i][0]+8*tsize)
            ##            y = int(self.mat[i][1]+8*tsize)
            x = int(self.mat.item(i, 0) + 8 * tsize)
            y = int(self.mat.item(i, 1) + 8 * tsize)
            if i % 10 != 9 and i / 10 != 9:
                yc = j % 9
                xc = j / 9
                j += 1
                if puzzle.original[xc, yc] == 0 and puzzle.current[xc, yc] != 0:
                    string = str(puzzle.current[xc, yc])
                    cv2.putText(self.captured, string, (x + w / 4, y + h), 0, tsize, (0, 0, 0), 2)
        if status.completed:
            #pl=self.path.split('.')
            #name=pl[0]+"_solved."+pl[1]
	    name= "solve"+str(rn.randint(0,5000))+".jpg"
	    print name
            cv2.imwrite(name, self.captured)
            led.turn_off()
            #os.system("espeak -g 5 -w out.wav 'I have solved the puzzle' && aplay out.wav")
            os.system("mplayer solved.mp3")
	    #t1=threading.Thread(target=speak, args=(puzzle.current,))
            #t1.start()
        cv2.imshow('sudoku', self.captured)
        """p = puzzle.current.reshape(1,81)
        for i in p:
            for j in i:
               os.system("espeak  -w out.wav {0} && aplay out.wav".format(j))"""
        key = cv2.waitKey(1000000000)
        if key == 27:
           #sys.exit()
	   cv2.destroyAllWindows()
        #t1.join()
class OCRmodelClass:
    #this class defines the data used for OCR,
    #and the associated methods for performing OCR
    def __init__(self):
        samples = np.loadtxt('generalsamples.data',np.float32)
        responses = np.loadtxt('generalresponses.data',np.float32)
        responses = responses.reshape((responses.size,1))
        #.model uses kNearest to perform OCR
        self.model = cv2.KNearest()
        self.model.train(samples,responses)
        #.iterations contains information on what type of morphology to use
        self.iterations = [-1,0,1,2]
        self.lvl = 0 #index of .iterations
        
    def OCR(self,status,image,puzzle):
        #preprocessing for OCR
        #convert image to grayscale
        gray = cv2.cvtColor(image.output, cv2.COLOR_BGR2GRAY)
        #noise removal with gaussian blur
        gray = cv2.GaussianBlur(gray,(5,5),0)
        image.outputGray = gray
        
        #attempt to read the image with 4 different morphology values and find the best result
        self.success = [0,0,0,0]
        self.errors = [0,0,0,0]
        for self.lvl in self.iterations:
            image.output = np.copy(image.outputBackup)
            self.OCR_read(status,image,puzzle)
            if self.errors[self.lvl+1]==0:
                self.errors[self.lvl+1] = puzzle.checkSolution()
        best = 8
        for i in range(4):
            if self.success[i] > best and self.errors[i]>=0:
                best = self.success[i]
                ibest = i
        print "success:",self.success
        print "errors:",self.errors
        
        if best==8:
            print "ERROR - OCR FAILURE"
            status.restart = True
        else:
            print "final morph erode iterations:",self.iterations[ibest]
            image.output = np.copy(image.outputBackup)
            self.lvl = self.iterations[ibest]
            self.OCR_read(status,image,puzzle)
            cv2.imshow('output',image.output)
            key = cv2.waitKey(1)

    def OCR_read(self,status,image,puzzle):
        #perform actual OCR using kNearest model
        thresh = cv2.adaptiveThreshold(image.outputGray,255,1,1,7,2)
        if self.lvl >= 0:
            morph = cv2.morphologyEx(thresh,cv2.MORPH_ERODE,None,iterations = self.lvl)
        elif self.lvl == -1:
            morph = cv2.morphologyEx(thresh,cv2.MORPH_DILATE,None,iterations = 1)

        thresh_copy = morph.copy()
        #thresh2 changes after findContours
        contours,hierarchy = cv2.findContours(morph,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
        thresh = thresh_copy

        puzzle.current = np.zeros((9,9),np.uint8)

        # testing section
        for cnt in contours:
            if cv2.contourArea(cnt)>20:
                [x,y,w,h] = cv2.boundingRect(cnt)
                if  h>20 and h<40 and w>8 and w<40:
                    if w<20:
                        diff = 20-w
                        x -= diff/2
                        w += diff
                    sudox = x/50
                    sudoy = y/50
                    cv2.rectangle(image.output,(x,y),(x+w,y+h),(0,0,255),2)
                    #prepare region of interest for OCR kNearest model
                    roi = thresh[y:y+h,x:x+w]
                    roismall = cv2.resize(roi,(25,35))
                    roismall = roismall.reshape((1,875))
                    roismall = np.float32(roismall)
                    #find result
                    retval, results, neigh_resp, dists = self.model.find_nearest(roismall, k = 1)
                    #check for read errors
                    if results[0][0]!=0:
                        string = str(int((results[0][0])))
                        if puzzle.current[sudoy,sudox]==0:
                            puzzle.current[sudoy,sudox] = int(string)
                        else:
                            self.errors[self.lvl+1]=-2 #double read error
                        self.success[self.lvl+1]+=1
                        cv2.putText(image.output,string,(x,y+h),0,1.4,(255,0,0),3)
                    else:
                        self.errors[self.lvl+1]=-3 #read zero error
                    

class solverStatusClass:
    #this class defines the status of the main loop
    def __init__(self):
        #.beginSolver becomes true when the puzzle is completely captured and ready to solve
        self.beginSolver = False
        #.puzzleFound becomes true when the puzzle is thought to be found but not yet read with OCR
        self.puzzleFound = False
        #.puzzleRead becomes true when OCR has confirmed the puzzle
        self.puzzleRead = False
        #.restart becomes true when the main loop needs to restart
        self.restart = False
        #.completed becomes true when the puzzle has been solved
        self.completed = False
        #.number of times imageClass.captureImage() detects no puzzle
        self.noDetect = 0
        #.number of times imageClass.captureImage() detects a puzzle
        self.detect = 0

def main():
  try:
    reader = OCRmodelClass()
    while True:
        speak = 0
      #try:
        status = solverStatusClass()
        while status.beginSolver == False:
            #status = solverStatusClass()
            puzzle = puzzleStatusClass()
            image = imageClass()
            print "Waiting for puzzle..."
            while status.puzzleFound == False:
                image.captureImage(status)
                if status.restart == True:
                    break
            while status.puzzleRead == False and status.puzzleFound == True :
                if speak != 1:
                   #os.system("espeak -g 5 -w out.wav 'I found a puzzle' && aplay out.wav")
                	os.system("mplayer detected.mp3")   
		speak = 1
                time.sleep(2)
		print "Reading"
                #image.captureImage(status)
                image.perspective()
                image.warp()
                reader.OCR(status, image, puzzle)
                if status.restart == True:
                    print "Restarting..."
                    break
                elif np.array_equal(puzzle.current, puzzle.last):
                    status.beginSolver = True
                    status.puzzleRead = True
                    print puzzle.current
                    puzzle.original = np.copy(puzzle.current)
                    #image.virtualImage(puzzle)
                else:
                    print "Rechecking for Puzzle Match..."
                    puzzle.last = np.copy(puzzle.current)


        if status.beginSolver == True:
                #os.system("espeak -g 5 -w out.wav 'I have recognised the puzzle and now I am trying to solve it' && aplay out.wav")
		os.system("mplayer recognised.mp3")
		print "solving start"
	        s=puzzle.original.reshape(1,81)
	        s=s.astype("S")
	        l = s.tolist()
	        p = "".join(l[0])
	
	        sudo_solver = solver(p)
	        sudo_solver.display(sudo_solver.solve())
	        status.completed = True
	
	        solved_seq = np.fromstring(sudo_solver.seq, np.int8) - 48
	        solved_seq=solved_seq.reshape(9,9)
	        puzzle.current = np.copy(solved_seq)
	        print puzzle.current
	        """p = puzzle.current.reshape(1,81)
	        for i in p:
                   for j in i:
                      os.system("espeak  -w out.wav {0} && aplay out.wav".format(j))"""
	        puzzle.solve = np.copy(solved_seq)
	        #image.virtualImage(puzzle)
		print "puzzle solved"
		"""if status.completed:
			cv2.polylines(image.captured, [self.biggest], True, (0, 255, 0), 3)
		elif status.puzzleFound:
			cv2.polylines(image.captured, [self.biggest], True, (0, 255, 255), 3)"""
	
		image.perspective()
		image.virtualImage(puzzle,status)
	
	        #image.captureImage(status)
	        #if image.maxArea > 0:
	      	"""except:
	        print "Error occured, restarting....."
		continue"""
  except AttributeError:
      #os.system("espeak -g 5 -w out.wav 'I am unable to recognise it ,please show me again' && aplay out.wav")
        os.system("mplayer unable.mp3")
	main()

  except KeyboardInterrupt:
     led.turn_off()

if __name__ == '__main__': main()
