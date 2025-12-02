🚗 Driver Drowsiness Detection (Eye + Head Movement AI)

A real-time driver drowsiness detection system built using OpenCV and MediaPipe Face Mesh.
The system monitors eye closure, blink rate, and head movements (pitch, yaw, roll) to detect early signs of drowsiness and alert the driver with an audible alarm.

⭐ Features

Eye Aspect Ratio (EAR) based blink & eye-closure detection

Head pose estimation using MediaPipe:

Pitch (looking down)

Yaw (looking sideways)

Roll (head tilting left/right)

Real-time webcam feed with alerts overlayed

Built-in alarm sound (no extra packages needed)

Lightweight, fast, and runs on any laptop webcam

🧠 Detection Logic

The system classifies drowsiness based on:

EAR below threshold for extended duration → eyes closing / microsleep

High pitch angle → head drooping

Large roll angle → sideways tilting

Automatic alarm when any threshold is crossed

All thresholds are tunable inside the code.

🛠️ Tech Stack

Python 3.10+

OpenCV

MediaPipe

NumPy

Winsound (for alarm)