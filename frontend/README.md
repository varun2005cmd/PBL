# Smart Door Security System - Frontend

IoT-Based Smart Door Security System with Face Recognition - Admin Panel

## Features

-  Real-time door status monitoring
-  Manual lock/unlock controls
-  Access logs and analytics
-  User management
-  Intrusion detection alerts

## Tech Stack

- React.js
- React Router
- Axios
- CSS3

## Getting Started

### Prerequisites

- Node.js (v14 or higher)
- npm or yarn

### Installation

```bash
# Install dependencies
npm install

# Start development server
npm start
```

The application will open at [http://localhost:3000](http://localhost:3000)

### Backend Configuration

Update the API base URL in `src/api/config.js` to point to your Flask backend.

Default: `http://localhost:5000/api`

## Project Structure

```
src/
   api/              # API service layer
   components/       # Reusable components
   pages/           # Main pages
   styles/          # CSS styles
   App.js           # Main app component
   index.js         # Entry point
```

## Available Scripts

- `npm start` - Run development server
- `npm build` - Build for production
- `npm test` - Run tests
