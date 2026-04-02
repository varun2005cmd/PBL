import React, { useState, useEffect } from 'react';
import Card from '../components/Card';
import Alert from '../components/Alert';
import Modal from '../components/Modal';
import Button from '../components/Button';
import { userService } from '../api/services';
import './UserManagement.css';

const UserManagement = () => {
  const [users, setUsers] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [alert, setAlert] = useState(null);

  // Add User modal
  const [showAddModal, setShowAddModal] = useState(false);
  const [newUserName, setNewUserName] = useState('');
  const [adding, setAdding] = useState(false);

  // Per-row deleting state
  const [deletingIds, setDeletingIds] = useState(new Set());

  useEffect(() => {
    fetchUsers();
    // Poll for updates every 10 seconds so new enrollments appear without manual refresh
    const interval = setInterval(fetchUsers, 10000);
    return () => clearInterval(interval);
  }, []);

  const fetchUsers = async () => {
    try {
      setError(null);
      const response = await userService.getUsers();
      setUsers(Array.isArray(response) ? response : []);
    } catch (err) {
      console.error('Failed to fetch users:', err);
      setError('Cannot reach backend. Is the server running?');
    } finally {
      setLoading(false);
    }
  };

  const handleToggleUser = async (userId, currentStatus) => {
    try {
      const response = await userService.toggleUser(userId, currentStatus);
      if (response.success) {
        setUsers(users.map(user =>
          user.id === userId
            ? { ...user, status: response.status }
            : user
        ));
        const newStatus = response.status === 'enabled' ? 'enabled' : 'disabled';
        showAlert('success', `User ${newStatus} successfully`);
      }
    } catch (err) {
      showAlert('error', 'Failed to update user status');
    }
  };

  const handleAddUser = async () => {
    const name = newUserName.trim();
    if (!name) return;
    setAdding(true);
    try {
      await userService.addUser(name);
      setShowAddModal(false);
      setNewUserName('');
      showAlert('success', `User "${name}" added successfully`);
      await fetchUsers();
    } catch (err) {
      const msg = err.response?.data?.error || 'Failed to add user';
      showAlert('error', msg);
    } finally {
      setAdding(false);
    }
  };

  const handleDeleteUser = async (user) => {
    if (!window.confirm(`Delete user "${user.name}"? This cannot be undone.`)) return;
    setDeletingIds(prev => new Set(prev).add(user.id));
    try {
      await userService.deleteUser(user.id);
      setUsers(users.filter(u => u.id !== user.id));
      showAlert('success', `User "${user.name}" deleted`);
    } catch (err) {
      showAlert('error', 'Failed to delete user');
    } finally {
      setDeletingIds(prev => {
        const next = new Set(prev);
        next.delete(user.id);
        return next;
      });
    }
  };

  const showAlert = (type, message) => {
    setAlert({ type, message });
    setTimeout(() => setAlert(null), 5000);
  };

  const formatLastAccess = (timestamp) => {
    if (!timestamp) return 'Never';
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
        <div className="users-header-right">
          <div className="users-summary">
            <div className="summary-item summary-enabled">
              <span className="summary-icon"></span>
              <span className="summary-count">{statusCounts.enabled}</span>
              <span className="summary-label">Enabled</span>
            </div>
            <div className="summary-item summary-disabled">
              <span className="summary-icon"></span>
              <span className="summary-count">{statusCounts.disabled}</span>
              <span className="summary-label">Disabled</span>
            </div>
          </div>
          <Button variant="primary" onClick={() => setShowAddModal(true)}>
            + Add User
          </Button>
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
        ) : error ? (
          <div className="users-empty">
            <span className="empty-icon">⚠️</span>
            <p>{error}</p>
            <Button variant="secondary" onClick={fetchUsers}>Retry</Button>
          </div>
        ) : users.length === 0 ? (
          <div className="users-empty">
            <span className="empty-icon"></span>
            <p>No users found. Add one to get started.</p>
          </div>
        ) : (
          <div className="users-grid">
            {users.map((user) => (
              <div key={user.id} className={`user-card user-${user.status}`}>
                <div className="user-avatar">
                  <span className="avatar-icon"></span>
                </div>
                <div className="user-info">
                  <h3 className="user-name">{user.name}</h3>
                  <div className="user-id">ID: {user.userId}</div>
                  {user.role && (
                    <div className="user-role-badge">{user.role}</div>
                  )}
                  <div className="user-last-access">
                    <span className="access-icon"></span>
                    Last access: {formatLastAccess(user.lastAccess)}
                  </div>
                  <div className="user-enrolled-status">
                    {user.enrolled
                      ? <span className="enrolled-yes"> Face Enrolled ({user.embeddingsCount} embeddings)</span>
                      : <span className="enrolled-no"> Not Enrolled</span>}
                  </div>
                </div>
                <div className="user-controls">
                  <div className="status-indicator">
                    <span className={`status-badge status-${user.status}`}>
                      {user.status === 'enabled' ? ' Enabled' : ' Disabled'}
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
                  <Button
                    variant="danger"
                    size="small"
                    onClick={() => handleDeleteUser(user)}
                    disabled={deletingIds.has(user.id)}
                  >
                    {deletingIds.has(user.id) ? 'Deleting…' : 'Delete'}
                  </Button>
                </div>
              </div>
            ))}
          </div>
        )}
      </Card>

      {/* Add User Modal */}
      <Modal
        isOpen={showAddModal}
        onClose={() => { setShowAddModal(false); setNewUserName(''); }}
        title="Add New User"
        size="small"
      >
        <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
          <p style={{ margin: 0, color: 'var(--text-secondary, #aaa)', fontSize: '0.9rem' }}>
            Creates a user record. Use <code>enroll_user.py</code> on the Pi to capture face embeddings.
          </p>
          <input
            type="text"
            placeholder="Full name"
            value={newUserName}
            onChange={(e) => setNewUserName(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && handleAddUser()}
            maxLength={100}
            style={{
              padding: '0.6rem 0.8rem',
              borderRadius: '6px',
              border: '1px solid var(--border, #444)',
              background: 'var(--input-bg, #1e1e2e)',
              color: 'var(--text, #fff)',
              fontSize: '1rem',
            }}
            autoFocus
          />
          <div style={{ display: 'flex', gap: '0.75rem' }}>
            <Button
              variant="secondary"
              onClick={() => { setShowAddModal(false); setNewUserName(''); }}
              fullWidth
            >
              Cancel
            </Button>
            <Button
              variant="primary"
              onClick={handleAddUser}
              disabled={adding || !newUserName.trim()}
              fullWidth
            >
              {adding ? 'Adding…' : 'Create User'}
            </Button>
          </div>
        </div>
      </Modal>
    </div>
  );
};

export default UserManagement;
