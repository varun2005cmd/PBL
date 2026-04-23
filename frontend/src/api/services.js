import apiClient from './apiClient';
import { ENDPOINTS } from './config';

// Door Service
export const doorService = {
  getStatus: async () => {
    const response = await apiClient.get(ENDPOINTS.DOOR_STATUS);
    return response.data;
  },

  lock: async () => {
    const response = await apiClient.post(ENDPOINTS.DOOR_LOCK);
    return response.data;
  },

  unlock: async () => {
    const response = await apiClient.post(ENDPOINTS.DOOR_UNLOCK);
    return response.data;
  },
};

// Hardware Service
export const hardwareService = {
  getStatus: async () => {
    const response = await apiClient.get(ENDPOINTS.DOOR_STATUS);
    return response.data; // { status, camera, servo, lcd, timestamp }
  },
};

// Access Logs Service
export const logsService = {
  getLogs: async (limit = 50) => {
    const response = await apiClient.get(ENDPOINTS.ACCESS_LOGS, {
      params: { limit }
    });
    return response.data;
  }
};

// User Service
export const userService = {
  getUsers: async () => {
    const response = await apiClient.get(ENDPOINTS.USERS);
    return response.data;
  },

  addUser: async (name) => {
    const response = await apiClient.post(ENDPOINTS.USERS, { name });
    return response.data;
  },

  deleteUser: async (id) => {
    const endpoint = ENDPOINTS.USER_DELETE.replace(':id', id);
    const response = await apiClient.delete(endpoint);
    return response.data;
  },

  toggleUser: async (userId, currentStatus) => {
    const endpoint = ENDPOINTS.USER_TOGGLE.replace(':id', userId);
    const response = await apiClient.put(endpoint, {
      status: currentStatus === 'enabled' ? 'disabled' : 'enabled'
    });
    return response.data;
  }
};

// Stats Service
export const statsService = {
  getStats: async () => {
    const response = await apiClient.get(ENDPOINTS.STATS);
    return response.data;
  }
};

export const enrollmentService = {
  start: async (name) => {
    const response = await apiClient.post(ENDPOINTS.ENROLL_START, { name });
    return response.data;
  },

  capture: async (sessionId) => {
    const response = await apiClient.post(ENDPOINTS.ENROLL_CAPTURE, { sessionId });
    return response.data;
  },

  complete: async (sessionId) => {
    const response = await apiClient.post(ENDPOINTS.ENROLL_COMPLETE, { sessionId });
    return response.data;
  }
};

export const violationsService = {
  getViolations: async () => {
    const response = await apiClient.get(ENDPOINTS.VIOLATIONS);
    return response.data;
  }
};
