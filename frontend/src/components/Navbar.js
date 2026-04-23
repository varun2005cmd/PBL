import React from 'react';
import { Link, useLocation } from 'react-router-dom';
import './Navbar.css';

const Navbar = () => {
  const location = useLocation();

  const navItems = [
    { path: '/', label: 'Dashboard', icon: 'D' },
    { path: '/logs', label: 'Logs', icon: 'L' },
    { path: '/users', label: 'Users', icon: 'U' },
    { path: '/enroll', label: 'Enroll', icon: 'E' },
    { path: '/violations', label: 'Violations', icon: 'V' }
  ];

  return (
    <nav className="navbar">
      <div className="navbar-brand">
        <span className="navbar-icon">SL</span>
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
