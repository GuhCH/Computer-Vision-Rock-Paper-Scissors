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
        return [0,0], "Invalid answer, try again!", cv2.imread(imgStates[comp])
    elif answer-comp == 1 or answer-comp == -2:
        return [1,0], "Your " + states[answer] + " beats my " + states[comp] + ". You win this round!", cv2.imread(imgStates[comp])
    elif answer == comp:
        return [0,0], "We both play " + states[answer] + ". It's a draw!", cv2.imread(imgStates[comp])
    else:
        return [0,1], "My " + states[comp] + " beats your " + states[answer] + ". You lose this round!", cv2.imread(imgStates[comp])

# Displays the camera and runs the TensorFLow model
def readModel(model,cap,data,comp_image,score):
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
    cv2.imshow('frame', full_img)

    # If model has >80% confidence in an answer, returns argument of that answer
    if np.max(prediction) > 0.8:
        return np.argmax(prediction)
    # Else returns argument of null answer
    else: return 3

# Displays the camera without running TF model to save processing power
def readCam(cap,comp_image,score):
    border = np.zeros((50,1280,3), np.uint8) + 255
    ret, frame = cap.read()
    frame1 = cv2.resize(frame, (640, 480), interpolation = cv2.INTER_AREA)
    comp_image1 = cv2.resize(comp_image, (640, 480), interpolation = cv2.INTER_AREA)
    combined_img = np.hstack((frame1, comp_image1))
    full_img = np.vstack((border,combined_img,border))
    cv2.putText(full_img, 'Score:' + str(score[0]), (20,40), cv2.FONT_HERSHEY_PLAIN, 3, (0,0,0), 2)
    cv2.putText(full_img, 'Score:' + str(score[1]), (1070,40), cv2.FONT_HERSHEY_PLAIN, 3, (0,0,0), 2)
    cv2.putText(full_img, 'Press P to play!', (20,570), cv2.FONT_HERSHEY_PLAIN, 3, (0,0,0), 2)
    cv2.putText(full_img, 'Hold Q to quit!', (900,570), cv2.FONT_HERSHEY_PLAIN, 3, (0,0,0), 2)
    cv2.imshow('frame', full_img)

# Counts for 4 seconds, saying "Rock, Paper, Scissors, Shoot" 
# Returns false if quit button is pressed in this time so round does not go ahead
def countdown(cap,score):
    words = ["rock.jpg","paper.jpg","scissors.jpg","shoot.png"]
    for i in range(4):
        comp_image = cv2.imread(words[i])
        t0 = time.clock_gettime(time.CLOCK_BOOTTIME)
        while time.clock_gettime(time.CLOCK_BOOTTIME)-t0 < 1: 
            readCam(cap,comp_image,score)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                return False
    return True

model = load_model('keras_model.h5')
cap = cv2.VideoCapture(0)
comp_image = cv2.imread('wait_screen.png')
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
score = [0,0]
our_pronouns = ["You ","I "]
while np.max(score) < 3:
    readCam(cap,comp_image,score)
    if cv2.waitKey(1) & 0xFF == ord('p'):
        if countdown(cap,score) == True:
            answer = readModel(model,cap,data,comp_image,score)
            scoreChange, out_msg, comp_image = playRPS(answer)
            print(out_msg)
            if out_msg != "Invalid answer, try again!":
                score[0] += scoreChange[0]
                score[1] += scoreChange[1]
                print("The score is now:")
                print("You: " + str(score[0]))
                print("Me: " + str(score[1]))
                t0 = time.clock_gettime(time.CLOCK_BOOTTIME)
                while time.clock_gettime(time.CLOCK_BOOTTIME)-t0 < 2: 
                    readCam(cap,comp_image,score)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
            comp_image = cv2.imread('wait_screen.png')
        else: break
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
if np.max(score) == 3:
    print(our_pronouns[np.argmax(score)] + "win!")
else: print("Game cancelled.")
print("Goodbye!")