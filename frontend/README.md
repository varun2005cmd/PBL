# Smart Door Security Frontend

React dashboard for the Flask smart door backend.

## Backend

By default the frontend calls:

```text
http://<current-browser-hostname>:5000
```

Override with `REACT_APP_API_BASE` in `frontend/.env` when needed.

## Pages

- `/` Dashboard: stats, hardware health, recent logs, manual lock/unlock
- `/logs` Access logs
- `/users` User management
- `/enroll` Live camera enrollment
- `/violations` Repeat-person violation evidence

## Run

```bash
npm install
npm start
```

## Production Build

```bash
npm run build
```
