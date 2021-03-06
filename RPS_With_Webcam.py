import cv2
from keras.models import load_model
import numpy as np
import time

# Runs the Rock, Paper, Scissors game, input is human's answer, output is score change + win/lose/draw msg
def playRPS(answer):
    states = ["rock","paper","scissors"]
    imgStates = ["rock.jpg","paper.jpg","scissors.jpg"]
    comp = np.random.randint(3)
    if answer > 2:
        return 3, cv2.imread(imgStates[comp])
    elif answer-comp == 1 or answer-comp == -2:
        return 0, cv2.imread(imgStates[comp])
    elif answer == comp:
        return 2, cv2.imread(imgStates[comp])
    else:
        return 1, cv2.imread(imgStates[comp])

# Displays the camera and runs the TensorFLow model
def readModel(model,cap,data,comp_image,score,centre_text):
    ret, frame = cap.read()
    resized_frame = cv2.resize(frame, (224, 224), interpolation = cv2.INTER_AREA)
    image_np = np.array(resized_frame)
    normalized_image = (image_np.astype(np.float32) / 127.0) - 1 # Normalize the image
    data[0] = normalized_image
    prediction = model.predict(data)
    frame1 = cv2.resize(frame, (640, 480), interpolation = cv2.INTER_AREA)
    comp_image1 = cv2.resize(comp_image, (640, 480), interpolation = cv2.INTER_AREA)
    full_img = np.hstack((frame1, comp_image1))
    cv2.putText(full_img, 'Score:' + str(score[0]), (20,40), cv2.FONT_HERSHEY_PLAIN, 3, (0,0,0), 2)
    cv2.putText(full_img, 'Score:' + str(score[1]), (1070,40), cv2.FONT_HERSHEY_PLAIN, 3, (0,0,0), 2)
    cv2.putText(full_img, 'Press P to play!', (20,570), cv2.FONT_HERSHEY_PLAIN, 3, (0,0,0), 2)
    cv2.putText(full_img, 'Hold Q to quit!', (900,570), cv2.FONT_HERSHEY_PLAIN, 3, (0,0,0), 2)
    cv2.imshow('frame', full_img)

    # If model has >80% confidence in an answer, returns argument of that answer
    if np.max(prediction) > 0.8:
        return np.argmax(prediction)
    # Else returns argument of null answer
    else: return 3

# Displays the camera without running TF model to save processing power
def readCam(cap,comp_image,score,centre_text):
    border = np.zeros((50,1280,3), np.uint8) + 255
    ret, frame = cap.read()
    frame1 = cv2.resize(frame, (640, 480), interpolation = cv2.INTER_AREA)
    comp_image1 = cv2.resize(comp_image, (640, 480), interpolation = cv2.INTER_AREA)
    combined_img = np.hstack((frame1, comp_image1))
    full_img = np.vstack((border,combined_img,border))
    textfont = cv2.FONT_HERSHEY_PLAIN
    cv2.putText(full_img, 'Score:' + str(score[0]), (20,40), textfont, 3, (0,0,0), 2)
    cv2.putText(full_img, 'Score:' + str(score[1]), (1070,40), textfont, 3, (0,0,0), 2)
    cv2.putText(full_img, 'Press P to play!', (20,570), textfont, 3, (0,0,0), 2)
    cv2.putText(full_img, 'Hold Q to quit!', (900,570), textfont, 3, (0,0,0), 2)
    ct_size = cv2.getTextSize(centre_text,textfont,3, 2)[0]
    ct_X = int(np.floor((full_img.shape[1] - ct_size[0]) / 2))
    cv2.putText(full_img, centre_text, (ct_X,40), textfont, 3, (0,0,0), 2)
    cv2.imshow('frame', full_img)

# Counts for 4 seconds, saying "Rock, Paper, Scissors, Shoot" 
# Returns false if quit button is pressed in this time so round does not go ahead
def countdown(cap,score):
    pictures = ["rock.jpg","paper.jpg","scissors.jpg","shoot.png"]
    words = ['Rock...','Paper...','Scissors...','Shoot!']
    for i in range(4):
        comp_image = cv2.imread(pictures[i])
        centre_text = words[i]
        t0 = time.clock_gettime(time.CLOCK_BOOTTIME)
        while time.clock_gettime(time.CLOCK_BOOTTIME)-t0 < 1: 
            readCam(cap,comp_image,score,centre_text)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                return False
    return True

model = load_model('keras_model.h5')
cap = cv2.VideoCapture(0)
comp_image = cv2.imread('wait_screen.png')
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
score = [0,0]
our_pronouns = ["You ","I "]
centre_text = ""
while np.max(score) < 3:
    readCam(cap,comp_image,score,centre_text)
    if cv2.waitKey(1) & 0xFF == ord('p'):
        if countdown(cap,score) == True:
            answer = readModel(model,cap,data,comp_image,score,centre_text)
            result, comp_image = playRPS(answer)
            if result == 0:
                score[0] += 1
                centre_text = "You win!"
            elif result == 1:
                score[1] += 1
                centre_text = "You lose!"
            elif result == 2:
                centre_text = "It's a draw!"
            else: centre_text = "Try again!"
            t0 = time.clock_gettime(time.CLOCK_BOOTTIME)
            while time.clock_gettime(time.CLOCK_BOOTTIME)-t0 < 1: 
                readCam(cap,comp_image,score,centre_text)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            centre_text = ""
            comp_image = cv2.imread('wait_screen.png')
        else: break
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
if np.max(score) == 3:
    centre_text = our_pronouns[np.argmax(score)] + "win the match!"
else: centre_text = "Game cancelled."
t0 = time.clock_gettime(time.CLOCK_BOOTTIME)
while time.clock_gettime(time.CLOCK_BOOTTIME)-t0 < 1: 
    readCam(cap,comp_image,score,centre_text)
    cv2.waitKey(1)
centre_text = "Goodbye!"
t0 = time.clock_gettime(time.CLOCK_BOOTTIME)
while time.clock_gettime(time.CLOCK_BOOTTIME)-t0 < 1: 
    readCam(cap,comp_image,score,centre_text)
    cv2.waitKey(1)