import React from 'react';
import { Link, useLocation } from 'react-router-dom';
import './Navbar.css';

const Navbar = () => {
  const location = useLocation();

  const navItems = [
    { path: '/', label: 'Dashboard', icon: '🏠' },
    { path: '/logs', label: 'Access Logs', icon: '📋' },
    { path: '/users', label: 'Users', icon: '👥' }
  ];

  return (
    <nav className="navbar">
      <div className="navbar-brand">
        <span className="navbar-icon">🔐</span>
        <span className="navbar-title">Smart Door Security</span>
      </div>
      <div className="navbar-menu">
        {navItems.map((item) => (
          <Link
            key={item.path}
            to={item.path}
            className={`navbar-item ${location.pathname === item.path ? 'active' : ''}`}
          >
            <span className="navbar-item-icon">{item.icon}</span>
            <span className="navbar-item-label">{item.label}</span>
          </Link>
        ))}
      </div>
    </nav>
  );
};

export default Navbar;
