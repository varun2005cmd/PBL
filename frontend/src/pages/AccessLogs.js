import React, { useState, useEffect } from 'react';
import Card from '../components/Card';
import { logsService } from '../api/services';
import './AccessLogs.css';

const AccessLogs = () => {
  const [logs, setLogs] = useState([]);
  const [loading, setLoading] = useState(true);
  const [filter, setFilter] = useState('all'); // all, authorized, unauthorized

  useEffect(() => {
    fetchLogs();
    // Poll for updates every 10 seconds
    const interval = setInterval(fetchLogs, 10000);
    return () => clearInterval(interval);
  }, []);

  const fetchLogs = async () => {
    try {
      const response = await logsService.getLogs(100);
      setLogs(Array.isArray(response) ? response : []);
    } catch (error) {
      console.error('Failed to fetch logs:', error);
    } finally {
      setLoading(false);
    }
  };

  const formatDateTime = (timestamp) => {
    const date = new Date(timestamp);
    return {
      date: date.toLocaleDateString(),
      time: date.toLocaleTimeString()
    };
  };

  const filteredLogs = logs.filter(log => {
    if (filter === 'all') return true;
    return log.accessType.toLowerCase() === filter;
  });

  const getAccessTypeClass = (accessType) => {
    return accessType.toLowerCase() === 'authorized' ? 'authorized' : 'unauthorized';
  };

  return (
    <div className="access-logs">
      <div className="logs-header">
        <div>
          <h1>Access Logs</h1>
          <p className="logs-subtitle">View all door access attempts and events</p>
        </div>
        <div className="logs-filters">
          <button
            className={`filter-btn ${filter === 'all' ? 'active' : ''}`}
            onClick={() => setFilter('all')}
          >
            All ({logs.length})
          </button>
          <button
            className={`filter-btn ${filter === 'authorized' ? 'active' : ''}`}
            onClick={() => setFilter('authorized')}
          >
            Authorized ({logs.filter(l => l.accessType.toLowerCase() === 'authorized').length})
          </button>
          <button
            className={`filter-btn ${filter === 'unauthorized' ? 'active' : ''}`}
            onClick={() => setFilter('unauthorized')}
          >
            Unauthorized ({logs.filter(l => l.accessType.toLowerCase() === 'unauthorized').length})
          </button>
        </div>
      </div>

      <Card>
        {loading ? (
          <div className="logs-loading">
            <div className="spinner"></div>
            <p>Loading access logs...</p>
          </div>
        ) : filteredLogs.length === 0 ? (
          <div className="logs-empty">
            <span className="empty-icon"></span>
            <p>No access logs found</p>
          </div>
        ) : (
          <div className="logs-table-container">
            <table className="logs-table">
              <thead>
                <tr>
                  <th>Date & Time</th>
                  <th>Access Type</th>
                  <th>User</th>
                  <th>Result</th>
                  <th>Evidence</th>
                </tr>
              </thead>
              <tbody>
                {filteredLogs.map((log) => {
                  const datetime = formatDateTime(log.timestamp);
                  const accessClass = getAccessTypeClass(log.accessType);
                  return (
                    <tr key={log.id} className={`log-row log-${accessClass}`}>
                      <td className="log-datetime">
                        <div className="datetime-container">
                          <span className="log-date">{datetime.date}</span>
                          <span className="log-time">{datetime.time}</span>
                        </div>
                      </td>
                      <td>
                        <span className={`access-badge access-${accessClass}`}>
                          {log.accessType === 'Authorized' ? '' : ''} {log.accessType}
                        </span>
                      </td>
                      <td className="log-user">
                        <span className="user-icon">
                          {log.accessType === 'Authorized' ? '' : ''}
                        </span>
                        {log.userName}
                      </td>
                      <td>
                        <span className={`result-badge result-${log.result.toLowerCase()}`}>
                          {log.result}
                        </span>
                      </td>
                      <td>
                        <button className="evidence-btn" title="View Evidence">
                           View
                        </button>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </Card>

      <div className="logs-stats">
        <div className="stat-item">
          <span className="stat-label">Total Entries:</span>
          <span className="stat-value">{logs.length}</span>
        </div>
        <div className="stat-item">
          <span className="stat-label">Last Updated:</span>
          <span className="stat-value">{new Date().toLocaleTimeString()}</span>
        </div>
      </div>
    </div>
  );
};

export default AccessLogs;
