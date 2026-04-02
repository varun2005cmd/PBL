// API Configuration
// When served from the Pi, the browser uses the Pi's IP automatically.
// window.location.hostname resolves to the Pi's IP when accessed from
// any device on the same network (phone, laptop, etc.)
const PI_PORT = 5000;
export const API_BASE_URL = `http://${window.location.hostname}:${PI_PORT}`;

// API Endpoints
export const ENDPOINTS = {
  DOOR_STATUS: '/hardware/status',
  DOOR_LOCK: '/hardware/lock',
  DOOR_UNLOCK: '/hardware/unlock',
  DOOR_EMERGENCY_UNLOCK: '/hardware/unlock',
  ACCESS_LOGS: '/logs',
  USERS: '/users',
  USER_TOGGLE: '/users/:id/toggle',
  STATS: '/stats',
  CAMERA_FRAME: '/camera/frame',
};
