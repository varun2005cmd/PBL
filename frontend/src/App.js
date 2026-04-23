import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Navbar from './components/Navbar';
import Dashboard from './pages/Dashboard';
import AccessLogs from './pages/AccessLogs';
import UserManagement from './pages/UserManagement';
import Enrollment from './pages/Enrollment';
import Violations from './pages/Violations';
import './App.css';

function App() {
  return (
    <Router>
      <div className="app">
        <Navbar />
        <main className="app-content">
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/logs" element={<AccessLogs />} />
            <Route path="/users" element={<UserManagement />} />
            <Route path="/enroll" element={<Enrollment />} />
            <Route path="/violations" element={<Violations />} />
          </Routes>
        </main>
        <footer className="app-footer">
          <p>2026 Smart Door Security System</p>
        </footer>
      </div>
    </Router>
  );
}

export default App;
