from flask import Flask, request, jsonify, session
from flask_socketio import SocketIO, emit, join_room, leave_room, disconnect
from flask_cors import CORS
import cv2
import mediapipe as mp
import numpy as np
import pickle
import logging
import time
import os
from pathlib import Path
from typing import Dict, Set, Any, Optional
from collections import deque
import base64

# Enhanced logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Terminal output
        logging.FileHandler("sign_language_server.log"),  # Also log to file for debugging
    ]
)
logger = logging.getLogger("sign-language-server")

# Force immediate flush for logs
for handler in logger.handlers:
    handler.flush()

app = Flask(__name__)
app.config['SECRET_KEY'] = 'sign_language_secret_key'
CORS(app, resources={r"/*": {"origins": "*"}})
# Fix: Remove eventlet async_mode or try alternative
socketio = SocketIO(app, cors_allowed_origins="*")  # Removed async_mode parameter

# Track active connections and states
active_connections: Dict[str, str] = {}  # Maps user_id to session_id
active_cameras: Set[str] = set()
users_in_meeting: Set[str] = set()

# Cache for storing recent predictions to smooth out results
prediction_cache: Dict[str, deque] = {}
# Number of frames to consider for smoothing predictions
CACHE_SIZE = 10
# Confidence threshold to report a sign
CONFIDENCE_THRESHOLD = 0.6

# Load model with better error handling
try:
    model_path = './model.p'
    logger.info(f"Attempting to load model from: {os.path.abspath(model_path)}")
    
    if os.path.exists(model_path):
        model_dict = pickle.load(open(model_path, 'rb'))
        model = model_dict['model']
        logger.info(f"Model loaded successfully from {model_path}")
        
        # Print model details for debugging
        logger.info(f"Model type: {type(model)}")
    else:
        # Creating dummy model for testing if real model doesn't exist
        logger.warning(f"Model file not found at {model_path}, creating dummy model for testing")
        
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=10)
        # Train with dummy data
        dummy_X = np.random.random((10, 84))
        dummy_y = np.random.randint(0, 3, 10)
        model.fit(dummy_X, dummy_y)
        logger.info("Dummy test model created and ready")
except Exception as e:
    logger.error(f"Error loading model: {str(e)}", exc_info=True)
    model = None

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.3, min_tracking_confidence=0.3, max_num_hands=1)

# Define labels - expand as needed
labels_dict = {0: 'yes', 1: 'no', 2: 'stop'}

# Cache to avoid sending the same caption repeatedly
caption_cache: Dict[str, str] = {}

# Rate limiting configuration
MAX_FRAMES_PER_SECOND = 10  # Maximum frames to process per second per user
rate_limit_counters: Dict[str, int] = {}  # Track frames processed per user

def process_frame(frame_data: bytes, user_id: str) -> Optional[str]:
    """
    Process a video frame to detect sign language gestures.
    
    Args:
        frame_data: Raw image data from the client
        user_id: Unique identifier for the user
        
    Returns:
        The detected sign language label or None if no valid detection
    """
    processing_start = time.time()
    
    # Simple rate limiting
    current_time = int(time.time())
    rate_key = f"{user_id}:{current_time}"
    
    if rate_key in rate_limit_counters:
        if rate_limit_counters[rate_key] >= MAX_FRAMES_PER_SECOND:
            logger.debug(f"Rate limit exceeded for user {user_id}")
            return None
        rate_limit_counters[rate_key] += 1
    else:
        # Clean up old entries
        for key in list(rate_limit_counters.keys()):
            if not key.endswith(str(current_time)):
                del rate_limit_counters[key]
        
        rate_limit_counters[rate_key] = 1
    
    logger.debug(f"Starting frame processing for user {user_id}")
    
    try:
        # Decode image
        nparr = np.frombuffer(frame_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            logger.warning(f"Failed to decode image for user {user_id}")
            return None
        
        H, W, _ = frame.shape
        logger.debug(f"Frame size: {W}x{H} for user {user_id}")
        
        # Convert to RGB for MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)
        
        # Check if hands were detected
        if not results.multi_hand_landmarks:
            logger.debug(f"No hands detected for user {user_id}")
            return None
            
        # Initialize arrays for hand landmark coordinates
        data_aux = []
        x_ = []
        y_ = []
        
        # Extract hand landmarks
        for hand_landmarks in results.multi_hand_landmarks:
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                x_.append(x)
                y_.append(y)
        
        # Validate we have landmarks
        if not x_ or not y_:
            logger.warning(f"No valid landmarks detected for user {user_id}")
            return None
        
        try:
            # Find bounding box to normalize coordinates
            x_min = min(x_)
            x_max = max(x_)
            y_min = min(y_)
            y_max = max(y_)
            
            # Handle possible division by zero if hand is just a point
            x_range = max(x_max - x_min, 0.001)
            y_range = max(y_max - y_min, 0.001)
            
            # Normalize coordinates to be invariant to scale and position
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                
                # Normalize by bounding box size
                data_aux.append((x - x_min) / x_range)
                data_aux.append((y - y_min) / y_range)
        except ValueError as e:
            logger.warning(f"Error processing landmarks for user {user_id}: {e}")
            return None
        
        # Handle both single-hand (42 features) and two-hand (84 features) cases
        if len(data_aux) == 42:
            logger.debug(f"Found 42 features, doubling for user {user_id}")
            data_aux = data_aux + data_aux  # duplicate to maintain expected input size
        elif len(data_aux) != 84:
            logger.warning(f"Unexpected number of features: {len(data_aux)}, expected 84 for user {user_id}")
            return None
        
        # Predict the sign
        prediction = model.predict([data_aux])
        predicted_class = int(prediction[0])
        
        # Get prediction probabilities if available
        confidence = None
        if hasattr(model, 'predict_proba'):
            try:
                probabilities = model.predict_proba([data_aux])[0]
                confidence = probabilities[predicted_class]
                logger.debug(f"Prediction confidence: {confidence:.2f} for class {predicted_class}")
            except:
                logger.debug(f"Could not get prediction probabilities")
        
        # Initialize prediction cache if needed
        if user_id not in prediction_cache:
            prediction_cache[user_id] = deque(maxlen=CACHE_SIZE)
        
        # Add current prediction to cache
        prediction_cache[user_id].append(predicted_class)
        
        # Count frequency of each prediction in the cache
        predictions_count = {}
        for pred in prediction_cache[user_id]:
            if pred in predictions_count:
                predictions_count[pred] += 1
            else:
                predictions_count[pred] = 1
        
        # Find the most frequent prediction
        max_count = 0
        smoothed_prediction = None
        for pred, count in predictions_count.items():
            if count > max_count:
                max_count = count
                smoothed_prediction = pred
        
        # Calculate confidence as ratio of most frequent prediction to cache size
        pred_confidence = max_count / len(prediction_cache[user_id])
        
        # Only return a prediction if we're confident enough
        if smoothed_prediction is not None and pred_confidence >= CONFIDENCE_THRESHOLD:
            result_class = smoothed_prediction
            result_label = labels_dict.get(result_class, "unknown")
            
            # Cache check - don't resend the same prediction repeatedly
            last_caption = caption_cache.get(user_id)
            if last_caption == result_label:
                logger.debug(f"Skipping duplicate caption '{result_label}' for user {user_id}")
                return None
            
            # Update caption cache
            caption_cache[user_id] = result_label
            
            logger.info(f"Detected sign '{result_label}' for user {user_id} with confidence {pred_confidence:.2f}")
            processing_time = time.time() - processing_start
            logger.debug(f"Processing time: {processing_time:.3f}s for user {user_id}")
            
            return result_label
        else:
            logger.debug(f"No confident prediction for user {user_id}")
            return None
            
    except Exception as e:
        logger.error(f"Error in frame processing for user {user_id}: {str(e)}", exc_info=True)
        return None

@socketio.on('connect')
def handle_connect():
    logger.info(f"New socket connection: {request.sid}")
    return True

@socketio.on('disconnect')
def handle_disconnect():
    user_id = None
    # Find user_id by session_id
    for uid, sid in list(active_connections.items()):
        if sid == request.sid:
            user_id = uid
            break
    
    if user_id:
        logger.info(f"Socket disconnected for user {user_id}")
        
        # Clean up connection and state
        del active_connections[user_id]
        if user_id in users_in_meeting:
            users_in_meeting.remove(user_id)
        if user_id in active_cameras:
            active_cameras.remove(user_id)
        if user_id in caption_cache:
            del caption_cache[user_id]
        if user_id in prediction_cache:
            del prediction_cache[user_id]
    else:
        logger.info(f"Socket disconnected for unknown user: {request.sid}")
    
    logger.info(f"Active connections: {len(active_connections)}, " +
               f"Active meetings: {len(users_in_meeting)}, " +
               f"Active cameras: {len(active_cameras)}")

@socketio.on('register')
def handle_register(data):
    user_id = data.get('userId')
    if not user_id:
        logger.warning(f"Registration attempt without user ID: {request.sid}")
        return False
    
    active_connections[user_id] = request.sid
    logger.info(f"Registered user {user_id} with session {request.sid}")
    
    # Join a room specific to this user
    join_room(user_id)
    
    return {'status': 'registered', 'message': 'User registered successfully'}

@socketio.on('ping')
def handle_ping():
    emit('pong')

@socketio.on('meeting_started')
def handle_meeting_started(data):
    user_id = data.get('userId')
    if not user_id:
        logger.warning(f"Meeting start signal without user ID: {request.sid}")
        return
    
    # Update session mapping if needed
    active_connections[user_id] = request.sid
    
    users_in_meeting.add(user_id)
    logger.info(f"User {user_id} joined a meeting. Active meetings: {len(users_in_meeting)}")
    
    emit('server_message', {'type': 'meeting_joined', 'message': 'Meeting joined successfully'})

@socketio.on('meeting_ended')
def handle_meeting_ended(data):
    user_id = data.get('userId')
    if not user_id:
        logger.warning(f"Meeting end signal without user ID: {request.sid}")
        return
    
    if user_id in users_in_meeting:
        users_in_meeting.remove(user_id)
    if user_id in active_cameras:
        active_cameras.remove(user_id)
    
    logger.info(f"User {user_id} left a meeting. Active meetings: {len(users_in_meeting)}")
    
    emit('server_message', {'type': 'meeting_ended', 'message': 'Meeting ended'})

@socketio.on('camera_on')
def handle_camera_on(data):
    user_id = data.get('userId')
    if not user_id:
        logger.warning(f"Camera on signal without user ID: {request.sid}")
        return
    
    active_cameras.add(user_id)
    logger.info(f"Camera activated for user {user_id}. Active cameras: {len(active_cameras)}")
    
    emit('server_message', {'type': 'camera_activated', 'message': 'Camera activated'})

@socketio.on('camera_off')
def handle_camera_off(data):
    user_id = data.get('userId')
    if not user_id:
        logger.warning(f"Camera off signal without user ID: {request.sid}")
        return
    
    if user_id in active_cameras:
        active_cameras.remove(user_id)
    
    logger.info(f"Camera deactivated for user {user_id}. Active cameras: {len(active_cameras)}")
    
    emit('server_message', {'type': 'camera_deactivated', 'message': 'Camera deactivated'})

@socketio.on('frame')
def handle_frame(data):
    user_id = data.get('userId')
    frame_data = data.get('frameData')
    
    if not user_id or not frame_data:
        logger.warning(f"Invalid frame data received")
        return
    
    if user_id not in active_cameras or user_id not in users_in_meeting:
        logger.debug(f"Ignoring frame from inactive user {user_id}")
        return
    
    try:
        # Convert base64 to bytes
        frame_bytes = base64.b64decode(frame_data.split(',')[1] if ',' in frame_data else frame_data)
        
        # Process the frame
        result = process_frame(frame_bytes, user_id)
        
        # Send the result back to the client if valid
        if result:
            emit('caption', {'text': result}, room=user_id)
            logger.debug(f"Sent caption '{result}' to user {user_id}")
    
    except Exception as e:
        logger.error(f"Error processing frame from {user_id}: {str(e)}", exc_info=True)

@app.route('/')
def root():
    """Root endpoint for health checks"""
    return jsonify({
        "status": "online", 
        "active_connections": len(active_connections),
        "active_meetings": len(users_in_meeting),
        "active_cameras": len(active_cameras)
    })

@app.route('/stats')
def stats():
    """Endpoint for detailed server statistics"""
    return jsonify({
        "active_connections": len(active_connections),
        "active_meetings": len(users_in_meeting),
        "active_cameras": len(active_cameras),
        "user_ids": list(active_connections.keys()),
        "users_in_meeting": list(users_in_meeting),
        "camera_active_users": list(active_cameras),
        "model_loaded": model is not None,
    })

if __name__ == "__main__":
    # Get port from environment or use default
    port = int(os.environ.get("PORT", 5000))
    
    # Start server
    logger.info(f"Starting sign language translation server on port {port}")
    socketio.run(app, host="0.0.0.0", port=port, debug=False)