Raspberry Pi 5 Deployment (Python 3.11.9)
=========================================

1. Clone repository
- git clone <repo_url> /home/pi/PBL
- cd /home/pi/PBL

2. Configure environment
- cp backend/.env.example backend/.env
- Edit backend/.env for camera/servo/LCD tuning.

3. Run setup script
- cd backend
- chmod +x deploy/pi5_setup.sh deploy/run_pi.sh
- ./deploy/pi5_setup.sh

4. Manual run (validation)
- cd /home/pi/PBL/backend
- source .venv/bin/activate
- python run_pi.py

5. Install systemd service
- sudo cp /home/pi/PBL/backend/deploy/doorlock.service /etc/systemd/system/
- sudo systemctl daemon-reload
- sudo systemctl enable doorlock.service
- sudo systemctl start doorlock.service
- sudo systemctl status doorlock.service

6. Verify hardware and API
- curl http://127.0.0.1:5000/health
- curl http://127.0.0.1:5000/hardware/status
- i2cdetect -y 1
- v4l2-ctl --list-devices

Notes
- Use a dedicated 5V supply for SG90 servo; do not power servo from Pi 5V pin under load.
- For USB camera instability, keep CAMERA_BACKEND=auto and CAMERA_FORCE_MJPEG=1.
