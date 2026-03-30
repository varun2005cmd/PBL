import React from 'react';
import './StatCard.css';

const StatCard = ({ title, value, icon, trend, variant = 'primary' }) => {
  return (
    <div className={`stat-card stat-card-${variant}`}>
      <div className="stat-card-content">
        <div className="stat-card-header">
          <span className="stat-card-icon">{icon}</span>
          <h3 className="stat-card-title">{title}</h3>
        </div>
        <div className="stat-card-value">{value}</div>
        {trend && <div className="stat-card-trend">{trend}</div>}
      </div>
    </div>
  );
};

export default StatCard;
