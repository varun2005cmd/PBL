# Smart Door Security System - Frontend

Complete React.js admin application for IoT-Based Smart Door Security System with Face Recognition.

## Quick Start

```bash
# Install dependencies
npm install

# Start development server
npm start
```

Application will open at: **http://localhost:3000**

## Features

✅ **Dashboard** - Real-time door status, manual controls, emergency unlock  
✅ **Access Logs** - Comprehensive access history with filtering  
✅ **User Management** - Enable/disable authorized users  
✅ **Mock Data Support** - Works standalone without backend  
✅ **Responsive Design** - Mobile and desktop friendly  
✅ **Intrusion Alerts** - Real-time security notifications  

## Tech Stack

- React.js 18
- React Router 6
- Axios
- CSS3 with gradients and animations

## Documentation

📖 See [SETUP.md](SETUP.md) for detailed installation and configuration guide.

## Backend Integration

The app connects to Flask backend at `http://localhost:5000/api` by default.

Update API URL in: `src/api/config.js`

## File Structure

```
src/
├── api/              # API layer (Axios)
├── components/       # Reusable UI components
├── pages/           # Main application pages
├── App.js           # Router & layout
└── index.js         # Entry point
```

## Future Enhancements

- WebSocket real-time updates
- Camera feed integration
- Facial recognition display
- Advanced analytics
- Hardware status monitoring

---

Built with ❤️ for secure access control
