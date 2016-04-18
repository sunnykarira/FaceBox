import cv2, os, argparse

DEFAULT_OUTPUT_PATH = 'FaceCaptureImages/'
DEFAULT_INPUT_PATH = 'haarcascade_frontalface_alt.xml'

class VideoCapture:

	def __init__(self):
		self.count = 0
		self.argsObj = Parse() # when script is running it can take console arguments
		self.faceCascade = cv2.CascadeClassifier(self.argsObj.input_path)
		self.videoSource = cv2.VideoCapture(0)

	def CaptureFrames(self):
		while True:

			#Create a unique number for each frame
			frameNumber = '%08d' % (self.count)

			#Capture frame by frame
			ret, frame = self.videoSource.read()

			# Set screen color to gray, so that haar cascade can easily detect edges and faces
			screenColor = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

			#Customize how the cascade detects your face
			# This is stored in the form of array
			faces = self.faceCascade.detectMultiScale(
				screenColor,
				scaleFactor = 1.1,
				minNeighbors = 5,    # Applying the frame on face Min neighbours faces cascade can detect
				minSize = (30,30),
				flags = cv2.CASCADE_SCALE_IMAGE)

			#Display the resulting frame
			cv2.imshow('Video', screenColor)

			#Checking for amount of faces detected
			#if length of faces is zero (no faces detcted)
			if len(faces) == 0:
				pass	
			
			elif len(faces) > 0:
				print('Face Detected')

				#Graph face and draw rectangle arond the face
				# picture, (x,y) , (x+width, y+height), color, thickness of rect
				for (x,y,w,h) in faces:
					cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)

				cv2.imwrite(DEFAULT_OUTPUT_PATH + frameNumber + '.png', frame)

			#increment to get next face number
			self.count = self.count + 1

			# If 'esc' is hot ,the video is closed. We are going 
			# to wait a fraction of a second per loop
			# 1 = one frame one millisecond wait for esc key
			if cv2.waitKey(1) == 27:
				break

		# when everything is done, release (close webcam) the capture and close windows
		self.videoSource.release()
		cv2.waitKey(500)  # wait half second
		cv2.destroyAllWindows()
		cv2.waitKey(500)


def Parse():
	parser = argparse.ArgumentParser(description='Path for face Detection')
	parser.add_argument('-i','--input_path', type=str, default= DEFAULT_INPUT_PATH, help = 'cascade input path')
	parser.add_argument('-o','--output_path', type=str, default= DEFAULT_OUTPUT_PATH, help='output path for pictures via webcam')
	args = parser.parse_args()
	return args

def clearImageFolder():
	if not (os.path.exists(DEFAULT_OUTPUT_PATH)):
		os.makedirs(DEFAULT_OUTPUT_PATH)
	else:
		for files in os.listdir(DEFAULT_OUTPUT_PATH):
			filePath = os.path.join(DEFAULT_OUTPUT_PATH, files)
			if os.path.isfile(filePath):
				os.unlink(filePath)
			else:
				continue

def main():
	clearImageFolder()

	#intialize class object
	faceDetect = VideoCapture()

	# call capture frames from class to face detection
	faceDetect.CaptureFrames()

if __name__ == '__main__':
	main()