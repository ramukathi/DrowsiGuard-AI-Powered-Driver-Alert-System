# DrowsiGuard-AI-Powered-Driver-Alert-System
## 🚗 Driver Drowsiness Detection System using AI 🧠 + Computer Vision 👁️

## 🔍 Problem Statement
Driver fatigue is one of the major causes of road accidents worldwide. Many drivers unknowingly fall asleep or become inattentive while driving, especially on long journeys or during night hours. A real-time alert system can significantly help reduce such accidents by warning drivers when drowsiness is detected.

---

## 🎯 Project Objective
To build a real-time **Driver Drowsiness Detection System** using **Computer Vision**, **Deep Learning**, and **Facial Landmark Detection** that:
- Detects closed eyes for more than 5 seconds
- Identifies drowsy facial expressions
- Gives audio alerts and captures incident images
- Rings a siren and stops the vehicle after 3 alerts
- Sends an SMS alert to an emergency contact using **Twilio**

---

## 💡 Features

### ✅ 1. Eye Closure Detection (EAR based)
- Uses Eye Aspect Ratio (EAR) to detect if eyes are closed
- Triggers an alert if closed > 5 seconds
- Captures frame with timestamp

### ✅ 2. Drowsy Face Detection (Deep Learning + CV)
- Identifies drowsy facial expressions (e.g., yawning, heavy eyelids)
- Captures image and displays “Driver is Drowsy” message

### ✅ 3. Multiple Alerts & Vehicle Shutdown
- After 3 alerts: rings a siren
- Displays “🚨 Vehicle Stopped Due to Driver Drowsiness” 
- Saves final captured image

### ✅ 4. SMS Notification
- Uses Twilio API to send SMS to emergency number after 3 alerts

---

## 🧠 Tech Stack

| Component        | Technology            |
|------------------|------------------------|
| Programming      | Python 3               |
| Computer Vision  | OpenCV                 |
| Landmark Detection | MediaPipe             |
| Eye & Mouth Detection | Eye Aspect Ratio (EAR), Mouth Aspect Ratio (MAR) |
| Sound Alerts     | `pygame`               |
| SMS Integration  | Twilio API             |
| IDE              | VS Code                |

---

## ⚙️ How It Works

1. **Face Detection**: Mediapipe identifies 468 facial landmarks in real-time.
2. **EAR Calculation**: Monitors eye openness. If closed > 5 seconds → ALERT.
3. **Drowsiness Check**: If face shows drowsy traits (closed eyes, yawning) → Capture frame.
4. **Alert System**:
   - 1st, 2nd, 3rd time: Short alert sound
   - After 3rd time: Long siren, vehicle stops
5. **SMS Notification**: An emergency SMS is sent after 3 alerts.

---

## 📸 Output Examples

- Captured image saved to `captured_faces/` folder with timestamp.
- Console displays:
  - `[INFO] Driver is Drowsy`
  - `[INFO] Alert frame saved: captured_faces/alert_<timestamp>.jpg`
  - `[ALERT] Vehicle stopped due to driver drowsiness`
  - `[SMS] Emergency alert sent to +91XXXXXXXXXX`

---

## 🛠 Setup Instructions

```bash
# Clone the repo
git clone https://github.com/yourusername/driver-drowsiness-detector.git
cd driver-drowsiness-detector

# Install dependencies
pip install -r requirements.txt

# To run the application 
python drowsiness_detector.py
