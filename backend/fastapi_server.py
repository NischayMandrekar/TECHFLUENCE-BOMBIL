from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

app = FastAPI()

# Allow CORS for WebSocket connections
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Allow requests from the Next.js frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Store active WebSocket connections
active_connections = {}

# WebSocket endpoint
@app.websocket("/ws/{user_id}")
async def websocket_endpoint(websocket: WebSocket, user_id: str):
    await websocket.accept()
    active_connections[user_id] = websocket
    print(f"User {user_id} connected")

    try:
        while True:
            # Receive data from the client
            data = await websocket.receive_text()
            print(f"Received from user {user_id}: {data}")

            # Handle start/stop commands
            if data == "start":
                print(f"Starting webcam processing for user {user_id}")
                # Start webcam processing (you can call a function here)
            elif data == "stop":
                print(f"Stopping webcam processing for user {user_id}")
                # Stop webcam processing (you can call a function here)
            else:
                # Broadcast the data to all other users
                for uid, connection in active_connections.items():
                    if uid != user_id:
                        await connection.send_text(data)
    except WebSocketDisconnect:
        print(f"User {user_id} disconnected")
        del active_connections[user_id]
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        print(f"User {user_id} disconnected")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)