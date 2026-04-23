# Frontend Setup

## Environment

Create `frontend/.env` only when the backend is not on the same host:

```bash
REACT_APP_API_BASE=http://<backend-host>:5000
REACT_APP_API_KEY=<optional-api-key>
```

## Commands

```bash
npm install
npm start
```

## Backend Routes Used

- `GET /health`
- `GET /hardware/status`
- `POST /hardware/lock`
- `POST /hardware/unlock`
- `GET /logs?limit=...`
- `GET /users`
- `POST /users`
- `DELETE /users/:id`
- `PUT /users/:id/toggle`
- `GET /stats`
- `GET /camera/frame`
- `POST /enroll/start`
- `POST /enroll/capture`
- `POST /enroll/complete`
- `GET /violations`
