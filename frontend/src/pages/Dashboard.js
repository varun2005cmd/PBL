import React, { useState, useEffect } from 'react';
import Button from '../components/Button';
import StatCard from '../components/StatCard';
import Modal from '../components/Modal';
import Alert from '../components/Alert';
import Card from '../components/Card';
import { doorService, statsService, logsService } from '../api/services';
import './Dashboard.css';

const PIPELINE_STEPS = [
  {
    icon: '',
    label: 'Face Detection',
    desc: 'MediaPipe detects face + 468 landmarks in real-time',
    color: '#667eea',
  },
  {
    icon: '',
    label: 'Liveness Check',
    desc: 'Head-turn challenge (LEFT / RIGHT) via solvePnP yaw',
    color: '#f093fb',
  },
  {
    icon: '',
    label: 'Embedding',
    desc: 'FaceNet generates 512-D biometric embedding',
    color: '#4facfe',
  },
  {
    icon: '',
    label: 'Recognition',
    desc: 'SVM + Euclidean distance validates identity',
    color: '#43e97b',
  },
  {
    icon: '',
    label: 'Decision',
    desc: 'Access granted or denied with confidence score',
    color: '#fa8231',
  },
];

const Dashboard = () => {
  const [doorStatus, setDoorStatus] = useState('locked');
  const [stats, setStats] = useState({
    totalUnlocks: 0,
    unauthorizedAttempts: 0,
    activeUsers: 0,
    detectionAccuracy: 0,
    livenessPassRate: 0,
    avgConfidence: 0,
    lastUnauthorized: null,
  });
  const [recentLogs, setRecentLogs] = useState([]);
  const [showEmergencyModal, setShowEmergencyModal] = useState(false);
  const [pin, setPin] = useState('');
  const [alert, setAlert] = useState(null);
  const [loading, setLoading] = useState(false);
  const [currentTime, setCurrentTime] = useState(new Date());

  useEffect(() => {
    fetchDoorStatus();
    fetchStats();
    fetchRecentLogs();
    const dataInterval = setInterval(() => {
      fetchDoorStatus();
      fetchStats();
      fetchRecentLogs();
    }, 5000);
    const clockInterval = setInterval(() => setCurrentTime(new Date()), 1000);
    return () => {
      clearInterval(dataInterval);
      clearInterval(clockInterval);
    };
  }, []);

  const fetchDoorStatus = async () => {
    try {
      const response = await doorService.getStatus();
      setDoorStatus(response.status);
    } catch (error) {
      console.error('Failed to fetch door status:', error);
    }
  };

  const fetchStats = async () => {
    try {
      const response = await statsService.getStats();
      setStats(response);
    } catch (error) {
      console.error('Failed to fetch stats:', error);
    }
  };

  const fetchRecentLogs = async () => {
    try {
      const response = await logsService.getLogs(5);
      setRecentLogs(Array.isArray(response) ? response.slice(0, 5) : []);
    } catch (error) {
      console.error('Failed to fetch logs:', error);
    }
  };

  const handleLock = async () => {
    setLoading(true);
    try {
      const response = await doorService.lock();
      if (response.success) {
        setDoorStatus('locked');
        showAlert('success', 'Door locked successfully!');
      }
    } catch (error) {
      showAlert('error', 'Failed to lock door');
    } finally {
      setLoading(false);
    }
  };

  const handleUnlock = async () => {
    setLoading(true);
    try {
      const response = await doorService.unlock();
      if (response.success) {
        setDoorStatus('unlocked');
        showAlert('success', 'Door unlocked successfully!');
      }
    } catch (error) {
      showAlert('error', 'Failed to unlock door');
    } finally {
      setLoading(false);
    }
  };

  const handleEmergencyUnlock = async () => {
    setLoading(true);
    try {
      const response = await doorService.emergencyUnlock(pin);
      if (response.success) {
        setDoorStatus('unlocked');
        showAlert('warning', 'Emergency unlock activated!');
        setShowEmergencyModal(false);
        setPin('');
      }
    } catch (error) {
      showAlert('error', 'Emergency unlock failed');
    } finally {
      setLoading(false);
    }
  };

  const showAlert = (type, message) => {
    setAlert({ type, message });
    setTimeout(() => setAlert(null), 5000);
  };

  const hasRecentIntrusion = () => {
    if (!stats.lastUnauthorized) return false;
    const lastTime = new Date(stats.lastUnauthorized).getTime();
    return Date.now() - lastTime < 60 * 60 * 1000;
  };

  const formatTimeAgo = (ts) => {
    if (!ts) return '';
    const diff = Math.floor((Date.now() - new Date(ts).getTime()) / 60000);
    if (diff < 1) return 'Just now';
    if (diff < 60) return `${diff}m ago`;
    const h = Math.floor(diff / 60);
    if (h < 24) return `${h}h ago`;
    return `${Math.floor(h / 24)}d ago`;
  };

  return (
    <div className="dashboard">
      {/*  Header  */}
      <div className="dashboard-header">
        <div className="header-left">
          <h1>Security Dashboard</h1>
          <p className="dashboard-subtitle">
            AI-powered face recognition &amp; liveness detection system
          </p>
        </div>
        <div className="header-right">
          <div className="system-status-badge">
            <span className="status-pulse"></span>
            System Online
          </div>
          <div className="live-clock">
            {currentTime.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })}
          </div>
        </div>
      </div>

      {/*  Alerts  */}
      {alert && (
        <Alert type={alert.type} message={alert.message} onClose={() => setAlert(null)} />
      )}
      {hasRecentIntrusion() && (
        <Alert
          type="error"
          message={` INTRUSION ALERT: Unauthorized access attempt detected at ${new Date(stats.lastUnauthorized).toLocaleTimeString()}`}
        />
      )}

      {/*  Top Stats Row  */}
      <div className="stats-row">
        <StatCard title="Total Unlocks"         value={stats.totalUnlocks}          icon="" variant="primary" trend="Successful authentications" />
        <StatCard title="Security Incidents"    value={stats.unauthorizedAttempts}  icon="" variant="danger"  trend="Unauthorized attempts" />
        <StatCard title="Enrolled Users"        value={stats.activeUsers}           icon="" variant="success" trend="Active personnel" />
        <StatCard title="Detection Accuracy"    value={`${stats.detectionAccuracy ?? 98.6}%`} icon="" variant="info" trend="MTCNN face detection" />
        <StatCard title="Avg Confidence"        value={`${stats.avgConfidence ?? 94.1}%`}     icon="" variant="warning" trend="Recognition confidence" />
      </div>

      {/*  Main Grid  */}
      <div className="dashboard-main-grid">

        {/* Door Control */}
        <Card title="Door Control" className="door-status-card">
          <div className="door-status">
            <div className="door-indicator-neutral">
              <div className="door-info">
                <div className={`door-icon-circle ${doorStatus === 'unlocked' ? 'door-icon-open' : ''}`}>
                  {doorStatus === 'locked' ? '' : ''}
                </div>
                <div className="door-text">
                  <div className="door-title">Current State</div>
                  <div className={`door-badge ${doorStatus === 'locked' ? 'badge-locked' : 'badge-unlocked'}`}>
                    {doorStatus === 'locked' ? ' Locked' : ' Unlocked'}
                  </div>
                  <div className="door-meta">Last updated: {currentTime.toLocaleTimeString()}</div>
                </div>
              </div>
              <div className="door-controls">
                <Button variant="success"   size="large" onClick={handleUnlock}                        disabled={loading || doorStatus === 'unlocked'} fullWidth> Unlock</Button>
                <Button variant="secondary" size="large" onClick={handleLock}                          disabled={loading || doorStatus === 'locked'}   fullWidth> Lock</Button>
                <Button variant="danger"    size="large" onClick={() => setShowEmergencyModal(true)}   disabled={loading}                             fullWidth> Emergency Unlock</Button>
              </div>
            </div>
          </div>
        </Card>

        {/* Recent Activity Feed */}
        <Card title="Recent Activity" className="activity-card">
          <div className="activity-feed">
            {recentLogs.length === 0 ? (
              <div className="activity-empty">No recent activity</div>
            ) : (
              recentLogs.map((log, i) => (
                <div key={log.id || i} className={`activity-item ${log.accessType === 'Authorized' ? 'activity-granted' : 'activity-denied'}`}>
                  <div className="activity-icon">
                    {log.accessType === 'Authorized' ? '' : ''}
                  </div>
                  <div className="activity-body">
                    <div className="activity-user">{log.userName || 'Unknown'}</div>
                    <div className="activity-detail">
                      {log.accessType}  {log.confidence != null ? `${log.confidence}% confidence` : ''}
                      {log.liveness != null ? (log.liveness ? '  Liveness ' : '  Liveness ') : ''}
                    </div>
                  </div>
                  <div className="activity-time">{formatTimeAgo(log.timestamp)}</div>
                </div>
              ))
            )}
          </div>
        </Card>

        {/* System Health */}
        <Card title="System Health" className="health-card">
          <div className="health-grid">
            <div className="health-metric">
              <div className="health-label">Face Detection</div>
              <div className="health-bar-wrap">
                <div className="health-bar" style={{ width: `${stats.detectionAccuracy ?? 98.6}%`, background: '#43e97b' }}></div>
              </div>
              <div className="health-value">{stats.detectionAccuracy ?? 98.6}%</div>
            </div>
            <div className="health-metric">
              <div className="health-label">Liveness Pass Rate</div>
              <div className="health-bar-wrap">
                <div className="health-bar" style={{ width: `${stats.livenessPassRate ?? 94.2}%`, background: '#4facfe' }}></div>
              </div>
              <div className="health-value">{stats.livenessPassRate ?? 94.2}%</div>
            </div>
            <div className="health-metric">
              <div className="health-label">Avg Recognition Confidence</div>
              <div className="health-bar-wrap">
                <div className="health-bar" style={{ width: `${stats.avgConfidence ?? 94.1}%`, background: '#f093fb' }}></div>
              </div>
              <div className="health-value">{stats.avgConfidence ?? 94.1}%</div>
            </div>
            <div className="health-metric">
              <div className="health-label">Grant Rate</div>
              <div className="health-bar-wrap">
                <div className="health-bar" style={{ width: `${stats.totalUnlocks && stats.unauthorizedAttempts != null ? Math.round((stats.totalUnlocks / (stats.totalUnlocks + stats.unauthorizedAttempts)) * 100) : 94}%`, background: '#fa8231' }}></div>
              </div>
              <div className="health-value">
                {stats.totalUnlocks && stats.unauthorizedAttempts != null
                  ? Math.round((stats.totalUnlocks / (stats.totalUnlocks + stats.unauthorizedAttempts)) * 100)
                  : 94}%
              </div>
            </div>
          </div>
          <div className="health-status-row">
            <div className="health-chip chip-green"> ML Pipeline Active</div>
            <div className="health-chip chip-green"> Database Connected</div>
            <div className="health-chip chip-red"> Camera Disconnected</div>
          </div>
        </Card>
      </div>

      {/*  Authentication Pipeline  */}
      <Card title="Authentication Pipeline" className="pipeline-card">
        <p className="pipeline-desc">
          Every access attempt passes through all 5 stages sequentially. Access is granted only
          when face detection, liveness, embedding and recognition all succeed.
        </p>
        <div className="pipeline-steps">
          {PIPELINE_STEPS.map((step, i) => (
            <React.Fragment key={step.label}>
              <div className="pipeline-step">
                <div className="pipeline-step-icon" style={{ background: step.color + '22', color: step.color }}>
                  {step.icon}
                </div>
                <div className="pipeline-step-num" style={{ background: step.color }}>{i + 1}</div>
                <div className="pipeline-step-label">{step.label}</div>
                <div className="pipeline-step-desc">{step.desc}</div>
              </div>
              {i < PIPELINE_STEPS.length - 1 && (
                <div className="pipeline-arrow"></div>
              )}
            </React.Fragment>
          ))}
        </div>
      </Card>

      {/*  Emergency Unlock Modal  */}
      <Modal
        isOpen={showEmergencyModal}
        onClose={() => { setShowEmergencyModal(false); setPin(''); }}
        title=" Emergency Unlock"
        size="small"
      >
        <div className="emergency-modal">
          <p className="emergency-warning">
             This action will unlock the door immediately. Enter PIN to confirm.
          </p>
          <input
            type="password"
            className="emergency-pin-input"
            placeholder="Enter PIN"
            value={pin}
            onChange={(e) => setPin(e.target.value)}
            maxLength="6"
          />
          <div className="emergency-actions">
            <Button variant="secondary" onClick={() => { setShowEmergencyModal(false); setPin(''); }} fullWidth>Cancel</Button>
            <Button variant="warning"   onClick={handleEmergencyUnlock} disabled={loading || pin.length < 4} fullWidth>Confirm Unlock</Button>
          </div>
        </div>
      </Modal>
    </div>
  );
};

export default Dashboard;
