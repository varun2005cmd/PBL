// API Configuration
// Set REACT_APP_API_BASE in a .env file to point to a specific backend
// (e.g. when developing off the Pi). Otherwise auto-detects from hostname.
const PI_PORT = 5000;
export const API_BASE_URL =
  process.env.REACT_APP_API_BASE ||
  `http://${window.location.hostname}:${PI_PORT}`;

// API Endpoints
export const ENDPOINTS = {
  DOOR_STATUS: '/hardware/status',
  DOOR_LOCK: '/hardware/lock',
  DOOR_UNLOCK: '/hardware/unlock',
  ACCESS_LOGS: '/logs',
  USERS: '/users',
  USER_TOGGLE: '/users/:id/toggle',
  USER_DELETE: '/users/:id',
  STATS: '/stats',
  CAMERA_FRAME: '/camera/frame',
  HEALTH: '/health',
  VIOLATIONS: '/violations',
  ENROLL_START: '/enroll/start',
  ENROLL_CAPTURE: '/enroll/capture',
  ENROLL_COMPLETE: '/enroll/complete',
};
