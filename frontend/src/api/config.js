// API Configuration
export const API_BASE_URL = 'http://localhost:5000/api';

// API Endpoints
export const ENDPOINTS = {
  DOOR_STATUS: '/door/status',
  DOOR_LOCK: '/door/lock',
  DOOR_UNLOCK: '/door/unlock',
  DOOR_EMERGENCY_UNLOCK: '/door/emergency-unlock',
  ACCESS_LOGS: '/logs',
  USERS: '/users',
  USER_TOGGLE: '/users/:id/toggle',
  STATS: '/stats'
};
