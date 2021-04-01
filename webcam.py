import cv2
from cv2 import face as face
import numpy as np
import math
import PIL.Image as Image
import pyvirtualcam

def distance(p1, p2):
    return math.sqrt( ((p1[0] - p2[0]) ** 2) + ((p1[1] - p2[1]) ** 2) )

print(distance( (5, 4), (3, 2) ))

BUFFER_ROWS = 480 * 2
BUFFER_COLS = 640 * 2

MIN_EYE_OPENING = 0.2
MAX_EYE_OPENING = 0.22

MIN_MOUTH_OPENING = 0.42
MAX_MOUTH_OPENING = 0.5


def blit(surface, target, position, centered=False):
    if not centered:
            bg_image = Image.fromarray(surface.astype('uint8'), 'RGBA')
            fg_image = Image.fromarray(target.astype("uint8"), "RGBA")

            bg_image.paste(fg_image, position, fg_image)

            #print(f"Pasting image\nBackground shape: {surface.shape}\nForeground shape: {target.shape}")

            return np.array(bg_image)
    else:
            bg_image = Image.fromarray(surface.astype('uint8'), 'RGBA')
            fg_image = Image.fromarray(target.astype("uint8"), "RGBA")

            bg_image.paste(fg_image, (position[0] - (target.shape[0] // 2), position[1] - (target.shape[1] // 2)), fg_image)

            return np.array(bg_image)

def overlay_image_alpha(img, img_overlay, x, y, alpha_mask):
    """Overlay `img_overlay` onto `img` at (x, y) and blend using `alpha_mask`.

    `alpha_mask` must have same HxW as `img_overlay` and values in range [0, 1].
    """
    # Image ranges
    y1, y2 = max(0, y), min(img.shape[0], y + img_overlay.shape[0])
    x1, x2 = max(0, x), min(img.shape[1], x + img_overlay.shape[1])

    # Overlay ranges
    y1o, y2o = max(0, -y), min(img_overlay.shape[0], img.shape[0] - y)
    x1o, x2o = max(0, -x), min(img_overlay.shape[1], img.shape[1] - x)

    # Exit if nothing to do
    if y1 >= y2 or x1 >= x2 or y1o >= y2o or x1o >= x2o:
        return

    # Blend overlay within the determined ranges
    img_crop = img[y1:y2, x1:x2]
    img_overlay_crop = img_overlay[y1o:y2o, x1o:x2o]
    alpha = alpha_mask[y1o:y2o, x1o:x2o, np.newaxis]
    alpha_inv = 1.0 - alpha

    img_crop[:] = alpha * img_overlay_crop + alpha_inv * img_crop

def rotate(img, angle):
    pil_image = Image.fromarray(img)
    pil_image = pil_image.rotate(angle, Image.NEAREST, expand=1)

    return np.array(pil_image)

def scale(img, amt):
    return cv2.resize(img, ( int(img.shape[0] * amt) , int(img.shape[1] * amt)), interpolation=cv2.INTER_NEAREST_EXACT)

class Smolbot:
    def __init__(self):
        self.sprite_eye_1 =cv2.imread("sprites/eye-1.png", flags=cv2.IMREAD_UNCHANGED)
        self.sprite_eye_2 =cv2.imread("sprites/eye-2.png", flags=cv2.IMREAD_UNCHANGED)
        self.sprite_eye_3 =cv2.imread("sprites/eye-3.png", flags=cv2.IMREAD_UNCHANGED)

        self.sprite_body = cv2.imread("sprites/body.png", flags=cv2.IMREAD_UNCHANGED)

        self.sprite_mouth_1 = cv2.imread("sprites/mouth-1.png", flags=cv2.IMREAD_UNCHANGED)
        self.sprite_mouth_2 = cv2.imread("sprites/mouth-2.png", flags=cv2.IMREAD_UNCHANGED)
        self.sprite_mouth_3 = cv2.imread("sprites/mouth-3.png", flags=cv2.IMREAD_UNCHANGED)

        self.eye_l_x = 750
        self.eye_r_x = 350
        self.eye_l_y = 450
        self.eye_r_y = 450
        self.body_x = 0
        self.body_y = 0
        self.rotation = 0
        self.mouth_x = (36 * 30) // 2
        self.mouth_y = 650

        self.scale = 0


    def render(self, traits, face_w, face_h, face_x, face_y, capture_w, capture_h):
        left_eye_y = traits[45][1]
        right_eye_y = traits[36][1]
        left_eye_x = traits[45][0]
        right_eye_x = traits[36, 0]
 

        tan = (left_eye_y - right_eye_y) / (left_eye_x - right_eye_x)
        deg = math.degrees( math.atan(tan) )

        buffer = np.zeros((BUFFER_ROWS, BUFFER_COLS, 4))
        buffer[:, :, 1] = 255
        body =  scale(self.sprite_body, 30)

        left_eye_opening = distance(traits[40], traits[20]) / face_h
        right_eye_opening = distance(traits[47], traits[24]) / face_h

        mouth_opening = distance( traits[8], traits[50] ) / face_h 

        print("--------------------------------------------------")
        print(f"Left eye opening:; {left_eye_opening}\nRight eye opening: {right_eye_opening}\nMouth Opening: {mouth_opening}")


        self.scale = ((face_w / capture_w + face_h / capture_h) / 2) * 2


        self.body_x = int(face_x * buffer.shape[0])
        self.body_y = int(face_y * buffer.shape[1])

        eye_level_l = abs(MAX_EYE_OPENING - MIN_EYE_OPENING) / left_eye_opening
        eye_level_r = abs(MAX_EYE_OPENING - MIN_EYE_OPENING) / right_eye_opening

        mouth_level = abs(MAX_MOUTH_OPENING - MIN_MOUTH_OPENING) / mouth_opening

        print(mouth_level)

        #print(eye_level_l)
        if eye_level_l < 0.2:
            body = blit(body, scale(self.sprite_eye_1, 30), ( self.eye_l_x, self.eye_l_y), centered=True)
        elif eye_level_l < 0.7:
            body = blit(body, scale(self.sprite_eye_2, 30), ( self.eye_l_x, self.eye_l_y), centered=True)
        else:
            body = blit(body, scale(self.sprite_eye_2, 30), ( self.eye_l_x, self.eye_l_y), centered=True)
        body = blit(body, scale(self.sprite_eye_2, 30), ( self.eye_r_x, self.eye_r_y), centered=True)

        if mouth_level < 0.3:
            body = blit(body, scale(self.sprite_mouth_1, 30), (self.mouth_x, self.mouth_y), centered=True)
        elif mouth_level < 0.7:
            body = blit(body, scale(self.sprite_mouth_2, 30), (self.mouth_x, self.mouth_y), centered=True)
        else:
            body = blit(body, scale(self.sprite_mouth_3, 30), (self.mouth_x, self.mouth_y), centered=True)


        buffer = blit(buffer, rotate(scale(body, self.scale), -deg), (self.body_x, self.body_y), centered=True)

        cv2.imshow("cazzoburro", buffer)

        return buffer



smolbot = Smolbot()

def main():
    face_width = 0
    face_height = 0
    traits = []
    # Load the cascade
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    # To capture video from webcam. 
    cap = cv2.VideoCapture(2)
    # To use a video file as input 
    # cap = cv2.VideoCapture('filename.mp4')
    facemark = face.createFacemarkLBF()
    facemark.loadModel("model.yaml")

    _, img = cap.read()
    CAPTURE_WIDTH = img.shape[0]
    CAPTURE_HEIGHT = img.shape[1]
    font = cv2.FONT_HERSHEY_SIMPLEX

    with pyvirtualcam.Camera(width=BUFFER_COLS, height=BUFFER_ROWS, fps=30) as cam:
        print(f'Using virtual camera: {cam.device}')
        frame = np.zeros((cam.height, cam.width, 3), np.uint8)  # RGB
        while True:
            # Read the frame
            _, img = cap.read()
            img = cv2.flip(img, 1)
            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Detect the faces
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            # Draw the rectangle around each face
            for i in range(len(faces)):
                landmarks = facemark.fit(gray, faces)
                arr = np.asarray(landmarks[1])
                arr = arr[i][0]
                traits = arr
                for i in range(arr.shape[0]):
                    point = arr[i]
                    cv2.circle(img, (point[0], point[1]), radius=3, color=(255,255,255),thickness=-1)
                    cv2.circle(img, (arr[52][0], arr[52][1]), 5, (100, 100, 100), -1 )
                    cv2.putText(img, str(i), (point[0], point[1]), font, 0.4, (255, 255, 255), 2)
            for (x, y, w, h) in faces:
                face_width = w
                face_height = h
                cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

                frame = cv2.cvtColor(smolbot.render(traits, face_width, face_height, (x + w // 2) / CAPTURE_WIDTH, (y + h // 2) / CAPTURE_HEIGHT, CAPTURE_WIDTH, CAPTURE_HEIGHT), cv2.COLOR_BGRA2RGB)
                cam.send(frame)
                cam.sleep_until_next_frame()
            # Display
            cv2.imshow('ti piscio in testa', img)
            # Stop if escape key is pressed
            k = cv2.waitKey(30) & 0xff
            if k==27:
                break
        # Release the VideoCapture object
        cap.release()

main()