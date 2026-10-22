import os
import shutil
import glob

DECA_RESULTS_DIR = "DECA/TestSamples/examples/results"
TRAIN_DIR = "training_data"

os.makedirs(TRAIN_DIR, exist_ok=True)

clean_faces = glob.glob("/root/deca-project/DECA/TestSamples/examples/results/*_vis.jpg")
print(f"Found {len(clean_faces)} conditioned faces from DECA. Routing to training folders...")

for face_path in clean_faces:
    filename = os.path.basename(face_path)
    # Splits "matt_01_crop.jpg" to create a folder named "matt"
    person_name = filename.split('_')[0].lower()
    
    person_folder = os.path.join(TRAIN_DIR, person_name)
    os.makedirs(person_folder, exist_ok=True)
    
    destination = os.path.join(person_folder, filename)
    shutil.copy(face_path, destination)
    print(f" Routed: {filename} -> {person_folder}/")

print("Handoff complete! InsightFace is ready.")