import mediapipe as mp
print(dir(mp))
try:
    from mediapipe.python.solutions import face_mesh
    print('face_mesh via python.solutions: OK')
except Exception as e:
    print('python.solutions failed:', e)
