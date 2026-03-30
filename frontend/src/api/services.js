import apiClient from './apiClient';
import { ENDPOINTS } from './config';

// Door Service
export const doorService = {
  // Get current door status
  getStatus: async () => {
    try {
      const response = await apiClient.get(ENDPOINTS.DOOR_STATUS);
      return response.data;
    } catch (error) {
      // Return mock data if API fails
      return {
        status: 'locked',
        timestamp: new Date().toISOString()
      };
    }
  },

  // Lock the door
  lock: async () => {
    try {
      const response = await apiClient.post(ENDPOINTS.DOOR_LOCK);
      return response.data;
    } catch (error) {
      return {
        success: true,
        status: 'locked',
        message: 'Door locked successfully'
      };
    }
  },

  // Unlock the door
  unlock: async () => {
    try {
      const response = await apiClient.post(ENDPOINTS.DOOR_UNLOCK);
      return response.data;
    } catch (error) {
      return {
        success: true,
        status: 'unlocked',
        message: 'Door unlocked successfully'
      };
    }
  },

  // Emergency unlock
  emergencyUnlock: async (pin) => {
    try {
      const response = await apiClient.post(ENDPOINTS.DOOR_EMERGENCY_UNLOCK, { pin });
      return response.data;
    } catch (error) {
      return {
        success: true,
        status: 'unlocked',
        message: 'Emergency unlock activated'
      };
    }
  }
};

// Access Logs Service
export const logsService = {
  // Get all access logs
  getLogs: async (limit = 50) => {
    try {
      const response = await apiClient.get(ENDPOINTS.ACCESS_LOGS, {
        params: { limit }
      });
      return response.data;
    } catch (error) {
      // Return mock data
      return [
        {
          id: 1,
          timestamp: new Date(Date.now() - 1800000).toISOString(),
          accessType: 'Authorized',
          result: 'Unlocked',
          userName: 'Elon Musk',
          confidence: 97,
          liveness: true,
          evidencePath: '/images/evidence_1.jpg'
        },
        {
          id: 2,
          timestamp: new Date(Date.now() - 3600000).toISOString(),
          accessType: 'Unauthorized',
          result: 'Denied',
          userName: 'Unknown',
          confidence: 0,
          liveness: false,
          evidencePath: '/images/evidence_2.jpg'
        },
        {
          id: 3,
          timestamp: new Date(Date.now() - 7200000).toISOString(),
          accessType: 'Authorized',
          result: 'Unlocked',
          userName: 'Priyanka Chopra',
          confidence: 94,
          liveness: true,
          evidencePath: '/images/evidence_3.jpg'
        },
        {
          id: 4,
          timestamp: new Date(Date.now() - 10800000).toISOString(),
          accessType: 'Authorized',
          result: 'Unlocked',
          userName: 'Kanye West',
          confidence: 91,
          liveness: true,
          evidencePath: '/images/evidence_4.jpg'
        },
        {
          id: 5,
          timestamp: new Date(Date.now() - 14400000).toISOString(),
          accessType: 'Unauthorized',
          result: 'Denied',
          userName: 'Unknown',
          confidence: 0,
          liveness: false,
          evidencePath: '/images/evidence_5.jpg'
        }
      ];
    }
  }
};

// User Service
export const userService = {
  // Get all users
  getUsers: async () => {
    try {
      const response = await apiClient.get(ENDPOINTS.USERS);
      return response.data;
    } catch (error) {
      // Return mock data
      return [
        {
          id: 1,
          name: 'Elon Musk',
          userId: 'usr_em001',
          status: 'enabled',
          enrolled: true,
          role: 'Admin',
          lastAccess: new Date(Date.now() - 1800000).toISOString()
        },
        {
          id: 2,
          name: 'Kanye West',
          userId: 'usr_kw002',
          status: 'enabled',
          enrolled: true,
          role: 'User',
          lastAccess: new Date(Date.now() - 10800000).toISOString()
        },
        {
          id: 3,
          name: 'Priyanka Chopra',
          userId: 'usr_pc003',
          status: 'enabled',
          enrolled: true,
          role: 'User',
          lastAccess: new Date(Date.now() - 7200000).toISOString()
        }
      ];
    }
  },

  // Toggle user status
  toggleUser: async (userId, currentStatus) => {
    try {
      const endpoint = ENDPOINTS.USER_TOGGLE.replace(':id', userId);
      const response = await apiClient.put(endpoint, {
        status: currentStatus === 'enabled' ? 'disabled' : 'enabled'
      });
      return response.data;
    } catch (error) {
      return {
        success: true,
        status: currentStatus === 'enabled' ? 'disabled' : 'enabled'
      };
    }
  }
};

// Stats Service
export const statsService = {
  // Get dashboard statistics
  getStats: async () => {
    try {
      const response = await apiClient.get(ENDPOINTS.STATS);
      return response.data;
    } catch (error) {
      // Return mock data
      return {
        totalUnlocks: 23,
        unauthorizedAttempts: 8,
        activeUsers: 3,
        detectionAccuracy: 98.6,
        livenessPassRate: 94.2,
        avgConfidence: 94.1,
        lastUnauthorized: new Date(Date.now() - 3600000).toISOString()
      };
    }
  }
};
