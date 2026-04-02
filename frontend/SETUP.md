# Setup Guide - Smart Door Security System Frontend

## Prerequisites

Before you begin, ensure you have the following installed:
- **Node.js** (v14.0.0 or higher)
- **npm** (v6.0.0 or higher) or **yarn**

## Installation Steps

### 1. Navigate to the frontend directory

```bash
cd frontend
```

### 2. Install dependencies

```bash
npm install
```

Or if you use yarn:

```bash
yarn install
```

### 3. Configure Backend API

The frontend is configured to connect to your Flask backend at `http://localhost:5000/api` by default.

If your backend runs on a different URL, update the API base URL in:

**File:** `src/api/config.js`

```javascript
export const API_BASE_URL = 'http://localhost:5000/api'; // Change this if needed
```

### 4. Start the development server

```bash
npm start
```

The application will automatically open in your browser at `http://localhost:3000`

## Project Structure

```
frontend/
 public/
    index.html           # HTML template
 src/
    api/                 # API service layer
       config.js       # API configuration
       apiClient.js    # Axios instance
       services.js     # API service methods
    components/          # Reusable components
       Alert.js
       Button.js
       Card.js
       Modal.js
       Navbar.js
       StatCard.js
    pages/              # Main pages
       Dashboard.js
       AccessLogs.js
       UserManagement.js
    App.js              # Main app component
    App.css
    index.js            # Entry point
    index.css           # Global styles
 package.json
 README.md
```

## Features Overview

### 1. Dashboard Page (`/`)
- **Door Status Indicator** - Shows if door is locked/unlocked
- **Control Buttons**
  - Lock Door
  - Unlock Door
  - Emergency Unlock (with PIN modal)
- **Statistics Cards**
  - Total successful unlocks
  - Unauthorized attempts
  - Active users
- **Intrusion Alert** - Displays when recent unauthorized attempt detected

### 2. Access Logs Page (`/logs`)
- **Filter Options**
  - All logs
  - Authorized only
  - Unauthorized only
- **Log Details Table**
  - Date & Time
  - Access Type (Authorized/Unauthorized)
  - User Name
  - Result (Unlocked/Denied)
  - Evidence placeholder
- **Visual Highlighting** - Unauthorized entries are highlighted in red

### 3. User Management Page (`/users`)
- **User List** - Grid view of all users
- **User Information**
  - Name
  - User ID
  - Last access time
  - Status (Enabled/Disabled)
- **Toggle Control** - Enable/Disable users
- **Summary Cards** - Count of enabled/disabled users

## API Integration

The application uses Axios to communicate with the Flask backend. All API calls are defined in `src/api/services.js`.

### Mock Data Fallback

If the backend is not running or returns errors, the application automatically falls back to mock data. This allows you to:
- Test the UI without the backend
- Develop frontend features independently
- Demonstrate the application

### Connecting to Real Backend

When your Flask backend is ready:

1. Ensure the backend is running on `http://localhost:5000`
2. The frontend will automatically connect to these endpoints:
   - `GET /api/door/status` - Get door status
   - `POST /api/door/lock` - Lock door
   - `POST /api/door/unlock` - Unlock door
   - `POST /api/door/emergency-unlock` - Emergency unlock
   - `GET /api/logs` - Get access logs
   - `GET /api/users` - Get users
   - `PUT /api/users/:id/toggle` - Toggle user status
   - `GET /api/stats` - Get dashboard statistics

## Customization

### Changing Colors

Main colors are defined using CSS gradients. Update these in the respective CSS files:

**Primary Gradient (Purple):**
```css
background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
```

**Success Gradient (Green):**
```css
background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
```

**Danger Gradient (Red):**
```css
background: linear-gradient(135deg, #eb3349 0%, #f45c43 100%);
```

### Adding New Pages

1. Create page component in `src/pages/`
2. Add route in `src/App.js`
3. Add navigation item in `src/components/Navbar.js`

Example:
```javascript
// In App.js
import NewPage from './pages/NewPage';

<Route path="/new-page" element={<NewPage />} />

// In Navbar.js
{ path: '/new-page', label: 'New Page', icon: '' }
```

## Building for Production

To create an optimized production build:

```bash
npm run build
```

The build folder will contain the production-ready files that can be deployed to any static hosting service.

## Troubleshooting

### Port 3000 is already in use
```bash
# Windows
set PORT=3001 && npm start

# Mac/Linux
PORT=3001 npm start
```

### Dependencies not installing
```bash
# Clear npm cache
npm cache clean --force

# Delete node_modules and reinstall
rm -rf node_modules
npm install
```

### Backend connection issues
1. Check if backend is running
2. Verify API_BASE_URL in `src/api/config.js`
3. Check browser console for CORS errors
4. Ensure Flask backend has CORS enabled

## Next Steps - Backend Integration

When connecting to the real Flask + ML backend:

1. **Update API endpoints** to match your Flask routes
2. **Handle authentication** - Add token management in `src/api/apiClient.js`
3. **Add real image display** - Update evidence path handling in AccessLogs
4. **Implement real-time updates** - Use WebSockets or Server-Sent Events
5. **Add face recognition UI** - Create component to display live camera feed
6. **Raspberry Pi integration** - Add hardware status indicators

## Support

For issues or questions:
- Check the browser console for errors
- Review the API calls in the Network tab
- Ensure all dependencies are correctly installed
- Verify the backend is running and accessible

---

**Happy Coding! **
