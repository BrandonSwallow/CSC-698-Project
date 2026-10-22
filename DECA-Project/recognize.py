import cv2
import os
import numpy as np
import sys
from insightface.app import FaceAnalysis

TRAIN_DIR = "training_data"
# This line tells it to look at the terminal command for the picture name
TEST_IMAGE = sys.argv[1] if len(sys.argv) > 1 else "test.jpg"
THRESHOLD = 0.45

app = FaceAnalysis(name="buffalo_l")
app.prepare(ctx_id=0, det_size=(640, 640))

known_embeddings = []
known_names = []

for person in os.listdir(TRAIN_DIR):
    person_folder = os.path.join(TRAIN_DIR, person)
    if not os.path.isdir(person_folder):
        continue
    for file in os.listdir(person_folder):
        path = os.path.join(person_folder, file)
        img = cv2.imread(path)
        if img is None:
            continue
        faces = app.get(img)
        if len(faces) == 0:
            continue
        emb = faces[0].embedding
        emb = emb / np.linalg.norm(emb)
        known_embeddings.append(emb)
        known_names.append(person)

def similarity_to_percent(score):
    percent = ((score + 1) / 2) * 100
    return max(0, min(100, percent))

img = cv2.imread(TEST_IMAGE)
faces = app.get(img)

# Fallback: If no face is found
if len(faces) == 0:
    print("\n>> ERROR: InsightFace could not detect any faces in this specific photo! <<\n")
else:
    # If faces ARE found, run the math and print the score FOR EACH face
    for face in faces:
        emb = face.embedding
        emb = emb / np.linalg.norm(emb)
        best_score = -1
        best_name = "Unknown"

        for i, known_emb in enumerate(known_embeddings):
            score = float(np.dot(emb, known_emb))
            if score > best_score:
                best_score = score
                best_name = known_names[i]

        confidence = similarity_to_percent(best_score)
        if best_score < THRESHOLD:
            best_name = "Unknown"

        x1, y1, x2, y2 = face.bbox.astype(int)
        cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2)
        cv2.putText(img, f"{best_name} ({confidence:.1f}%)",
                    (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.75, (0,255,0), 2)
        
        # Print the score right here inside the loop!
        print(f"\n>> MATCH RESULT: {best_name.upper()} at {confidence:.1f}% Confidence <<\n")

# Always save the final image, whether it drew a box or not
cv2.imwrite("final_score.jpg", img)