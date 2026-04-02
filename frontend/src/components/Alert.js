import React from 'react';
import './Alert.css';

const Alert = ({ type = 'info', message, onClose }) => {
  const icons = {
    success: '',
    error: '',
    warning: '',
    info: ''
  };

  return (
    <div className={`alert alert-${type}`}>
      <span className="alert-icon">{icons[type]}</span>
      <span className="alert-message">{message}</span>
      {onClose && (
        <button className="alert-close" onClick={onClose}>
          
        </button>
      )}
    </div>
  );
};

export default Alert;
