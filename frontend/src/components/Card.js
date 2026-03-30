import React from 'react';
import './Card.css';

const Card = ({ title, children, variant = 'default', className = '' }) => {
  return (
    <div className={`card card-${variant} ${className}`}>
      {title && <div className="card-header">{title}</div>}
      <div className="card-body">{children}</div>
    </div>
  );
};

export default Card;
