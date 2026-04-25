import React, { useState } from 'react';
import Alert from '../components/Alert';
import Button from '../components/Button';
import Card from '../components/Card';
import { API_BASE_URL, ENDPOINTS } from '../api/config';
import { enrollmentService } from '../api/services';
import './Enrollment.css';

const Enrollment = () => {
  const [name, setName] = useState('');
  const [session, setSession] = useState(null);
  const [captured, setCaptured] = useState(0);
  const [targetFrames, setTargetFrames] = useState(15);
  const [busy, setBusy] = useState(false);
  const [alert, setAlert] = useState(null);

  const cameraUrl = `${API_BASE_URL}${ENDPOINTS.CAMERA_FRAME}?captured=${captured}`;

  const showAlert = (type, message) => {
    setAlert({ type, message });
    setTimeout(() => setAlert(null), 5000);
  };

  const startEnrollment = async () => {
    if (!name.trim()) return;
    setBusy(true);
    try {
      const response = await enrollmentService.start(name.trim());
      setSession(response.sessionId);
      setCaptured(response.captured || 0);
      setTargetFrames(response.targetFrames || 5);
      showAlert('success', response.message || 'Enrollment started');
    } catch (err) {
      showAlert('error', err.response?.data?.error || 'Failed to start enrollment');
    } finally {
      setBusy(false);
    }
  };

  const captureImage = async () => {
    setBusy(true);
    try {
      const response = await enrollmentService.capture(session);
      setCaptured(response.captured);
      setTargetFrames(response.targetFrames);
      showAlert('info', response.message || 'Captured image');
    } catch (err) {
      showAlert('error', err.response?.data?.error || 'Capture failed');
    } finally {
      setBusy(false);
    }
  };

  const completeEnrollment = async () => {
    setBusy(true);
    try {
      const response = await enrollmentService.complete(session);
      showAlert('success', response.message || 'Enrollment complete');
      setSession(null);
      setName('');
      setCaptured(0);
    } catch (err) {
      showAlert('error', err.response?.data?.error || 'Enrollment failed');
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="enrollment page-shell">
      <div className="page-header">
        <div>
          <h1>Enroll User</h1>
          <p>Create face embeddings from the live Raspberry Pi camera.</p>
        </div>
      </div>

      {alert && <Alert type={alert.type} message={alert.message} onClose={() => setAlert(null)} />}

      <div className="enroll-grid">
        <Card title="Live Camera">
          <div className="camera-panel">
            {session ? (
              <img src={cameraUrl} alt="Live camera frame" className="camera-feed" />
            ) : (
              <div className="camera-placeholder">Start enrollment to preview the camera.</div>
            )}
          </div>
        </Card>

        <Card title="Capture Flow">
          <div className="enroll-form">
            <label htmlFor="enroll-name">Name</label>
            <input
              id="enroll-name"
              value={name}
              onChange={(event) => setName(event.target.value)}
              placeholder="Full name"
              disabled={Boolean(session) || busy}
            />
            <div className="capture-progress">
              <span>Capturing image {captured}/{targetFrames}</span>
              <div className="progress-track">
                <div className="progress-fill" style={{ width: `${Math.min(100, (captured / targetFrames) * 100)}%` }} />
              </div>
            </div>
            {!session ? (
              <Button variant="primary" onClick={startEnrollment} disabled={busy || !name.trim()} fullWidth>
                Start Enrollment
              </Button>
            ) : (
              <>
                <Button variant="success" onClick={captureImage} disabled={busy || captured >= targetFrames} fullWidth>
                  Capture Image
                </Button>
                <Button variant="primary" onClick={completeEnrollment} disabled={busy || captured < 3} fullWidth>
                  Complete Enrollment
                </Button>
              </>
            )}
          </div>
        </Card>
      </div>
    </div>
  );
};

export default Enrollment;
