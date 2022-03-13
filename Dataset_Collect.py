import cv2
import mediapipe as mp
import numpy as np
import time, os

action =['come','away','spin']
seq_length =20
secs_for_action=30

mp_hands = mp.solutions.hands 
mp_drawing=mp.solutions.drawing_utils
hands= mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

cap=cv2.VideoCapture(0)
created_time=int(time.time())
os.makedirs('dataset',exist_ok=True)

while cap.isOpened():
    for idx, action in enumerate(action):
        data =[]
        ret,img=cap.read()
        img=cv2.flip(img,1)
        cv2.putText(img, f'Waiting for collection {action.upper()} action...', org=(10, 30), fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=1,color=(255,255,255),thickness=2)
        cv2.imshow('img',img)
        cv2.waitKey(3000)
        
        start_time =time.time()
        while time.time() - start_time < secs_for_action:
            ret, img =cap.read()
            img =cv2.flip(img,1)
            img=cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
            result = hands.process(img)
            img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            
            if result.multi_hand_landmarks is not None:
                for res in result.multi_hand_landmarks:
                    joint = np.zeros((21,4))
                    for j, lm in enumerate(res.landmarks):
                        joint[j]=[lm.x,lm.y,lm.z,lm.visibility]
                    v1=joint[[]]
                    v2=joint[[]]
                    v= v2-v1
                    
                    
                    v=v/np.linalg.norm(v, axis=1)[:,np.newaxis]
                    angle = np.arccos(np.einsum('nt,nt->n',
                    v[[],:],
                    v[[],:]
                                                ))
                    angle=np.degrees(angle)
                    angle_label=np.array([angle], dtype=np.float32)
                    angle_label = np.append(angle_label,idx)
                    
                    d=np.concatenate([joint.flatten(), angle_label])
                    data.append(d)
                    mp_drawing.draw_landmarks(img,res,mp_hands.HAND_CONNECTIONS)
            cv2.imshow('img',img)
            if cv2.waitkey(1) ==ord('q'):
                break
        data = np.array(data)
        print(action,data.shape)
        np.save(os.path.join('dataset', f'raw_{action}_{created_time}'),data)
        
        full_seq_data =[]
        for seq in range(len(data) - seq_length):
            full_seq_data.append(data[seq:seq + seq_length])
        
        full_seq_data = np.array(full_seq_data)
        print(action, full_seq_data.shape)
        np.save(os.path.join('dataset', f'seq_{action}_{created_time}'), full_seq_data)
    break
            