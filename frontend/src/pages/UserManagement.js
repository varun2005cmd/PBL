import React, { useState, useEffect } from 'react';
import Card from '../components/Card';
import Alert from '../components/Alert';
import { userService } from '../api/services';
import './UserManagement.css';

const UserManagement = () => {
  const [users, setUsers] = useState([]);
  const [loading, setLoading] = useState(true);
  const [alert, setAlert] = useState(null);

  useEffect(() => {
    fetchUsers();
  }, []);

  const fetchUsers = async () => {
    try {
      const response = await userService.getUsers();
      setUsers(Array.isArray(response) ? response : []);
    } catch (error) {
      console.error('Failed to fetch users:', error);
      showAlert('error', 'Failed to load users');
    } finally {
      setLoading(false);
    }
  };

  const handleToggleUser = async (userId, currentStatus) => {
    try {
      const response = await userService.toggleUser(userId, currentStatus);
      if (response.success) {
        // Update user status in state
        setUsers(users.map(user => 
          user.id === userId 
            ? { ...user, status: response.status } 
            : user
        ));
        const newStatus = response.status === 'enabled' ? 'enabled' : 'disabled';
        showAlert('success', `User ${newStatus} successfully`);
      }
    } catch (error) {
      showAlert('error', 'Failed to update user status');
    }
  };

  const showAlert = (type, message) => {
    setAlert({ type, message });
    setTimeout(() => setAlert(null), 5000);
  };

  const formatLastAccess = (timestamp) => {
    const date = new Date(timestamp);
    const now = new Date();
    const diffMs = now - date;
    const diffMins = Math.floor(diffMs / 60000);
    const diffHours = Math.floor(diffMs / 3600000);
    const diffDays = Math.floor(diffMs / 86400000);

    if (diffMins < 1) return 'Just now';
    if (diffMins < 60) return `${diffMins} minutes ago`;
    if (diffHours < 24) return `${diffHours} hours ago`;
    if (diffDays === 1) return 'Yesterday';
    return `${diffDays} days ago`;
  };

  const getStatusCounts = () => {
    const enabled = users.filter(u => u.status === 'enabled').length;
    const disabled = users.filter(u => u.status === 'disabled').length;
    return { enabled, disabled };
  };

  const statusCounts = getStatusCounts();

  return (
    <div className="user-management">
      <div className="users-header">
        <div>
          <h1>User Management</h1>
          <p className="users-subtitle">Manage authorized users and access permissions</p>
        </div>
        <div className="users-summary">
          <div className="summary-item summary-enabled">
            <span className="summary-icon">✓</span>
            <span className="summary-count">{statusCounts.enabled}</span>
            <span className="summary-label">Enabled</span>
          </div>
          <div className="summary-item summary-disabled">
            <span className="summary-icon">✕</span>
            <span className="summary-count">{statusCounts.disabled}</span>
            <span className="summary-label">Disabled</span>
          </div>
        </div>
      </div>

      {alert && (
        <Alert 
          type={alert.type} 
          message={alert.message} 
          onClose={() => setAlert(null)} 
        />
      )}

      <Card>
        {loading ? (
          <div className="users-loading">
            <div className="spinner"></div>
            <p>Loading users...</p>
          </div>
        ) : users.length === 0 ? (
          <div className="users-empty">
            <span className="empty-icon">👥</span>
            <p>No users found</p>
          </div>
        ) : (
          <div className="users-grid">
            {users.map((user) => (
              <div key={user.id} className={`user-card user-${user.status}`}>
                <div className="user-avatar">
                  <span className="avatar-icon">👤</span>
                </div>
                <div className="user-info">
                  <h3 className="user-name">{user.name}</h3>
                  <div className="user-id">ID: {user.userId}</div>
                  {user.role && (
                    <div className="user-role-badge">{user.role}</div>
                  )}
                  <div className="user-last-access">
                    <span className="access-icon">🕐</span>
                    Last access: {formatLastAccess(user.lastAccess)}
                  </div>
                  <div className="user-enrolled-status">
                    {user.enrolled
                      ? <span className="enrolled-yes">✅ Face Enrolled</span>
                      : <span className="enrolled-no">⚠️ Not Enrolled</span>}
                  </div>
                </div>
                <div className="user-controls">
                  <div className="status-indicator">
                    <span className={`status-badge status-${user.status}`}>
                      {user.status === 'enabled' ? '✓ Enabled' : '✕ Disabled'}
                    </span>
                  </div>
                  <label className="toggle-switch">
                    <input
                      type="checkbox"
                      checked={user.status === 'enabled'}
                      onChange={() => handleToggleUser(user.id, user.status)}
                    />
                    <span className="toggle-slider"></span>
                  </label>
                </div>
              </div>
            ))}
          </div>
        )}
      </Card>
    </div>
  );
};

export default UserManagement;
