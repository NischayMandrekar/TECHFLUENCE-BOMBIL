"use client";
import useUser from "@/hooks/useUser";
import { useRouter } from "next/navigation";
import { SignedIn, SignedOut, useAuth, useClerk } from "@clerk/nextjs";
import { ZegoUIKitPrebuilt } from "@zegocloud/zego-uikit-prebuilt";
import React, { use, useEffect, useRef, useState } from "react";
import { v4 as uuid } from "uuid";
import { io, Socket } from "socket.io-client";

// Add ImageCapture polyfill declaration
declare global {
  interface Window {
    ImageCapture?: any;
  }
}

const Page = ({ params }: { params: Promise<{ roomid: string }> }) => {
  const { isSignedIn, userId } = useAuth();
  const { fullName } = useUser();
  const { roomid } = use(params);
  const router = useRouter();
  const { openSignIn } = useClerk();

  const zpRef = useRef<ReturnType<typeof ZegoUIKitPrebuilt.create> | null>(null);
  const roomContainerRef = useRef<HTMLDivElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const socketRef = useRef<Socket | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const frameCountRef = useRef<number>(0);
  const captionCountRef = useRef<number>(0);
  const frameIntervalRef = useRef<number | null>(null);
  const pingIntervalRef = useRef<number | null>(null);
  const reconnectTimeoutRef = useRef<number | null>(null);
  const cameraCheckIntervalRef = useRef<number | null>(null);

  // Captions State
  const [captions, setCaptions] = useState<string>("");
  const [showCaptions, setShowCaptions] = useState<boolean>(true);
  const [inMeeting, setInMeeting] = useState(false);
  const [cameraActive, setCameraActive] = useState(false);
  const [lastCaptionTime, setLastCaptionTime] = useState<number | null>(null);

  // Connection state
  const [socketStatus, setSocketStatus] = useState<"connecting" | "connected" | "disconnected" | "error">("disconnected");
  const [reconnectAttempts, setReconnectAttempts] = useState(0);
  const maxReconnectAttempts = 5;

  // Notifications
  const [notification, setNotification] = useState<{message: string, type: 'success' | 'error' | 'info'} | null>(null);

  // Logging utility function with console and optional UI notification
  const log = (message: string, type: "info" | "error" | "warn" | "debug" = "info", notify: boolean = false) => {
    const timestamp = new Date().toISOString();
    const formattedMessage = `[${timestamp}] ${message}`;
    
    switch (type) {
      case "error":
        console.error(formattedMessage);
        if (notify) setNotification({message, type: 'error'});
        break;
      case "warn":
        console.warn(formattedMessage);
        if (notify) setNotification({message, type: 'info'});
        break;
      case "debug":
        console.debug(formattedMessage);
        break;
      default:
        console.log(formattedMessage);
        if (notify) setNotification({message, type: 'success'});
    }
  };

  // Function to establish Socket.io connection
  const connectSocket = () => {
    if (!userId) {
      log("No user ID available, cannot connect to Socket", "warn");
      return;
    }

    if (socketRef.current && socketRef.current.connected) {
      log("Socket already connected", "debug");
      return;
    }

    // Close any existing socket before creating a new one
    if (socketRef.current) {
      try {
        socketRef.current.disconnect();
      } catch (e) {
        log(`Error closing existing socket: ${e}`, "debug");
      }
    }

    log(`Initializing Socket.io connection for user ${userId}`);
    setSocketStatus("connecting");

    // Use secure WebSocket if in production
    const protocol = window.location.protocol === 'https:' ? 'https:' : 'http:';
    // Use relative URL to automatically handle different domain/localhost
    const socketBaseUrl = process.env.NEXT_PUBLIC_SOCKET_URL || `${protocol}//${window.location.hostname}:5000`;
    
    try {
      socketRef.current = io(socketBaseUrl, {
        reconnection: true,
        reconnectionAttempts: maxReconnectAttempts,
        reconnectionDelay: 1000,
        reconnectionDelayMax: 5000,
        timeout: 20000,
      });
    
      socketRef.current.on('connect', () => {
        log("Socket.io connection established successfully", "info", true);
        setSocketStatus("connected");
        setReconnectAttempts(0);
        
        // Register user ID with server
        socketRef.current?.emit('register', { userId });
        
        // Notify server about meeting state if already in meeting
        if (inMeeting) {
          socketRef.current?.emit('meeting_started', { userId });
          log("Sent 'meeting_started' signal to server");
          
          // Also send camera state if active
          if (cameraActive) {
            setTimeout(() => {
              socketRef.current?.emit('camera_on', { userId });
              log("Sent 'camera_on' signal to server");
            }, 500); // Small delay to ensure proper sequencing
          }
        }
        
        // Set up ping interval to keep connection alive
        if (pingIntervalRef.current) {
          clearInterval(pingIntervalRef.current);
        }
        
        pingIntervalRef.current = window.setInterval(() => {
          socketRef.current?.emit('ping');
        }, 30000); // ping every 30 seconds
      });

      socketRef.current.on('pong', () => {
        log("Received pong from server", "debug");
      });
      
      socketRef.current.on('server_message', (data) => {
        log(`Received server message: ${data.type} - ${data.message}`, "info");
      });
      
      socketRef.current.on('caption', (data) => {
        // Handle captions
        const message = data.text;
        log(`Received caption from server: "${message}"`);
        setCaptions(message);
        
        // Update statistics
        captionCountRef.current += 1;
        setLastCaptionTime(Date.now());
        
        log(`Caption statistics: Total received: ${captionCountRef.current}`);
        
        // Use TTS to speak the caption
        try {
          // Cancel any ongoing speech before starting new one
          window.speechSynthesis.cancel();
          
          const utterance = new SpeechSynthesisUtterance(message);
          utterance.rate = 1.0; // Normal speaking rate
          utterance.pitch = 1.0; // Normal pitch
          utterance.volume = 1.0; // Full volume
          window.speechSynthesis.speak(utterance);
          log(`TTS speaking: "${message}"`);
        } catch (error) {
          log(`TTS error: ${error}`, "error");
        }
      });

      socketRef.current.on('connect_error', (error) => {
        log(`Socket.io connection error: ${error}`, "error", true);
        setSocketStatus("error");
      });

      socketRef.current.on('disconnect', (reason) => {
        log(`Socket.io disconnected: ${reason}`, "warn");
        setSocketStatus("disconnected");
        
        // Clear ping interval
        if (pingIntervalRef.current) {
          clearInterval(pingIntervalRef.current);
          pingIntervalRef.current = null;
        }
      });
      
      socketRef.current.on('reconnect_attempt', (attemptNumber) => {
        log(`Socket.io reconnection attempt ${attemptNumber}/${maxReconnectAttempts}`, 
            "info", attemptNumber === 1);
        setReconnectAttempts(attemptNumber);
      });
      
      socketRef.current.on('reconnect_failed', () => {
        log("Max reconnection attempts reached. Please refresh the page.", "error", true);
      });
    } catch (e) {
      log(`Error creating Socket.io connection: ${e}`, "error", true);
      setSocketStatus("error");
    }
  };

  // Connect to Socket when userId is available and in meeting
  useEffect(() => {
    if (inMeeting && userId) {
      connectSocket();
    }
    
    return () => {
      // Clean up reconnect timeout
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
        reconnectTimeoutRef.current = null;
      }
    };
  }, [userId, inMeeting]);

  // Cleanup Socket on unmount
  useEffect(() => {
    return () => {
      // Send meeting_ended only once
      if (socketRef.current && socketRef.current.connected) {
        log("Sending 'meeting_ended' before closing Socket");
        try {
          socketRef.current.emit('meeting_ended', { userId });
          // Small delay to let the message get sent before closing
          setTimeout(() => {
            if (socketRef.current) {
              socketRef.current.disconnect();
              log("Socket.io connection closed on cleanup");
            }
          }, 100);
        } catch (e) {
          log(`Error during cleanup: ${e}`, "error");
          if (socketRef.current) {
            socketRef.current.disconnect();
          }
        }
      } else if (socketRef.current) {
        socketRef.current.disconnect();
        log("Socket.io connection closed on cleanup");
      }
      
      if (pingIntervalRef.current) {
        clearInterval(pingIntervalRef.current);
      }
      
      if (frameIntervalRef.current) {
        clearInterval(frameIntervalRef.current);
      }
      
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
      }
      
      if (cameraCheckIntervalRef.current) {
        clearInterval(cameraCheckIntervalRef.current);
      }
    };
  }, [userId]);

  // Redirect to sign in if not signed in
  useEffect(() => {
    if (!isSignedIn) {
      log("User not signed in, redirecting to sign-in", "warn");
      openSignIn();
    }
  }, [isSignedIn, openSignIn]);

  // Enhanced camera detection function
  const checkCameraStatus = () => {
    // Try multiple selector patterns to find the video element
    const videoSelectors = [
      '.ZegoLocalVideo video',
      '.zego-local-video video',
      '.ZegoRoom video',
      'video[data-zego-local="true"]',
      'video.zego-video'
    ];
    
    let videoElement: HTMLVideoElement | null = null;
    
    // Try each selector until we find a video element
    for (const selector of videoSelectors) {
      const element = document.querySelector(selector) as HTMLVideoElement;
      if (element && element.srcObject instanceof MediaStream) {
        videoElement = element;
        break;
      }
    }
    
    // If still not found, try to find any video element
    if (!videoElement) {
      const allVideos = document.querySelectorAll('video');
      for (const video of allVideos) {
        if (video.srcObject instanceof MediaStream) {
          videoElement = video;
          log("Found video element using fallback approach", "info");
          break;
        }
      }
    }
    
    if (videoElement && videoElement.srcObject instanceof MediaStream) {
      const tracks = (videoElement.srcObject as MediaStream).getVideoTracks();
      const isActive = tracks.length > 0 && tracks[0].enabled && tracks[0].readyState === 'live';
      
      // Only update if there's a change
      if (isActive !== cameraActive) {
        log(`Camera status changed to: ${isActive ? 'active' : 'inactive'}`, "info", true);
        setCameraActive(isActive);
        streamRef.current = videoElement.srcObject;
        
        // Notify server about camera state change
        if (socketRef.current && socketRef.current.connected) {
          socketRef.current.emit(isActive ? 'camera_on' : 'camera_off', { userId });
          log(`Sent '${isActive ? "camera_on" : "camera_off"}' signal to server`);
        }
      }
      
      return isActive;
    }
    
    log("No video element with active stream found", "debug");
    return false;
  };

  // Setup video frame capture and processing when camera is active
  useEffect(() => {
    // Clear any existing interval
    if (frameIntervalRef.current) {
      clearInterval(frameIntervalRef.current);
      frameIntervalRef.current = null;
    }
    
    if (!inMeeting || !cameraActive) {
      log("Not in meeting or camera not active, skipping frame capture setup", "debug");
      return;
    }
    
    if (!socketRef.current || !socketRef.current.connected) {
      log("Socket not connected, skipping frame capture setup", "warn");
      // Try to reconnect if disconnected
      if (socketStatus !== "connecting") {
        connectSocket();
      }
      return;
    }
    
    log("Setting up video frame capture");
    frameCountRef.current = 0;
    
    // Create canvas for frame capture if needed
    if (!canvasRef.current) {
      canvasRef.current = document.createElement('canvas');
      canvasRef.current.width = 640;  // Standard width
      canvasRef.current.height = 480; // Standard height
    }
    
    // Function to capture and send frames
    const captureAndSendFrame = async () => {
      if (!streamRef.current || !socketRef.current || !socketRef.current.connected || !canvasRef.current) {
        log("Missing required resources for frame capture", "warn");
        return;
      }
      
      try {
        const videoTrack = streamRef.current.getVideoTracks()[0];
        if (!videoTrack || videoTrack.readyState !== 'live' || !videoTrack.enabled) {
          log("Video track not available or disabled", "debug");
          return;
        }
        
        let imageCapture;
        
        // Try to use ImageCapture API if available
        if (window.ImageCapture) {
          try {
            imageCapture = new window.ImageCapture(videoTrack);
            const bitmap = await imageCapture.grabFrame();
            
            // Draw to canvas
            const ctx = canvasRef.current.getContext('2d');
            if (ctx) {
              ctx.drawImage(bitmap, 0, 0, canvasRef.current.width, canvasRef.current.height);
            }
          } catch (e) {
            log(`ImageCapture API failed: ${e}`, "debug");
            // Fall back to canvas method if ImageCapture fails
            imageCapture = null;
          }
        }
        
        // Fall back to canvas method if ImageCapture not available or failed
        if (!imageCapture) {
          try {
            // Try to find video element again as backup
            let videoEl = null;
            document.querySelectorAll('video').forEach((video) => {
              if (video.srcObject === streamRef.current) {
                videoEl = video;
              }
            });
            
            if (videoEl) {
              const ctx = canvasRef.current.getContext('2d');
              if (ctx) {
                ctx.drawImage(videoEl, 0, 0, canvasRef.current.width, canvasRef.current.height);
              }
            } else {
              log("Could not find video element for fallback capture", "warn");
              return;
            }
          } catch (e) {
            log(`Canvas frame capture failed: ${e}`, "error");
            return;
          }
        }
        
        // Get image data from canvas
        const imageData = canvasRef.current.toDataURL('image/jpeg', 0.7); // Lower quality for better performance
        
        // Send to server
        socketRef.current.emit('frame', {
          userId,
          frameData: imageData
        });
        
        // Update statistics
        frameCountRef.current += 1;
        
        if (frameCountRef.current % 100 === 0) {
          log(`Frames sent: ${frameCountRef.current}, Captions received: ${captionCountRef.current}`);
        }
      } catch (e) {
        log(`Error capturing frame: ${e}`, "error");
      }
    };
    
    // Start capturing and sending frames at a reasonable rate (5fps)
    frameIntervalRef.current = window.setInterval(captureAndSendFrame, 200);
    
    return () => {
      if (frameIntervalRef.current) {
        clearInterval(frameIntervalRef.current);
        frameIntervalRef.current = null;
      }
    };
  }, [inMeeting, cameraActive, socketStatus, userId]);

  // Set up camera status monitoring
  useEffect(() => {
    if (!inMeeting) {
      return;
    }
    
    log("Setting up camera monitoring");
    
    // Initial check after a short delay to allow ZegoCloud to initialize
    setTimeout(() => {
      checkCameraStatus();
    }, 2000);
    
    // Set up periodic checks
    cameraCheckIntervalRef.current = window.setInterval(() => {
      checkCameraStatus();
    }, 5000); // Check every 5 seconds
    
    return () => {
      if (cameraCheckIntervalRef.current) {
        clearInterval(cameraCheckIntervalRef.current);
        cameraCheckIntervalRef.current = null;
      }
    };
  }, [inMeeting]);

  // Initialize and join ZegoCloud meeting
  useEffect(() => {
    if (!isSignedIn || !roomid || !fullName || !userId) {
      log("Missing required information to join room", "warn");
      return;
    }
    
    const joinRoom = async () => {
      try {
        log(`Joining room ${roomid} as ${fullName} (${userId})`);
        
        // Get app credentials from environment
        const appID = process.env.NEXT_PUBLIC_ZEGO_APP_ID;
        const serverSecret = process.env.NEXT_PUBLIC_ZEGO_SERVER_SECRET;
        
        if (!appID || !serverSecret) {
          log("Missing ZegoCloud credentials", "error", true);
          return;
        }
        
        // Parse app ID as number
        const appIDNumber = parseInt(appID);
        
        // Initialize ZegoCloud
        const kitToken = ZegoUIKitPrebuilt.generateKitTokenForTest(
          appIDNumber,
          serverSecret,
          roomid,
          userId,
          fullName
        );
        
        if (!roomContainerRef.current) {
          log("Room container not found", "error");
          return;
        }
        
        // Create ZegoCloud instance
        zpRef.current = ZegoUIKitPrebuilt.create(kitToken);
        
        // Configure ZegoCloud
        zpRef.current.joinRoom({
          container: roomContainerRef.current,
          sharedLinks: [
            {
              name: "Copy Link",
              url: window.location.href,
            },
          ],
          scenario: {
            mode: ZegoUIKitPrebuilt.VideoConference,
          },
          turnOnMicrophoneWhenJoining: true,
          turnOnCameraWhenJoining: true,
          showMyCameraToggleButton: true,
          showMyMicrophoneToggleButton: true,
          showAudioVideoSettingsButton: true,
          onJoinRoom: () => {
            log("Successfully joined ZegoCloud room", "info", true);
            setInMeeting(true);
            
            // Connect to socket after joining room
            connectSocket();
          },
          onLeaveRoom: () => {
            log("Left ZegoCloud room", "info");
            setInMeeting(false);
            setCameraActive(false);
            
            // Notify server about meeting end
            if (socketRef.current && socketRef.current.connected) {
              socketRef.current.emit('meeting_ended', { userId });
              log("Sent 'meeting_ended' signal to server");
            }
          },
        });
      } catch (error) {
        log(`Error joining room: ${error}`, "error", true);
      }
    };
    
    joinRoom();
  }, [isSignedIn, roomid, fullName, userId]);

  // Handle notification dismiss
  useEffect(() => {
    if (notification) {
      const timer = setTimeout(() => {
        setNotification(null);
      }, 5000);
      
      return () => clearTimeout(timer);
    }
  }, [notification]);

  // Toggle captions
  const toggleCaptions = () => {
    setShowCaptions(!showCaptions);
    log(`Captions ${!showCaptions ? 'enabled' : 'disabled'}`);
  };

  return (
    <div className="flex flex-col h-screen bg-gray-100">
      {/* Notification */}
      {notification && (
        <div className={`fixed top-4 right-4 p-4 rounded-lg shadow-lg z-50 max-w-md
          ${notification.type === 'error' ? 'bg-red-500 text-white' :
            notification.type === 'success' ? 'bg-green-500 text-white' :
            'bg-blue-500 text-white'}`}>
          <span>{notification.message}</span>
          <button 
            onClick={() => setNotification(null)}
            className="ml-4 text-white hover:text-gray-200"
          >
            ✕
          </button>
        </div>
      )}
      
      {/* Connection Status Indicator */}
      <div className="absolute top-2 left-2 z-10 flex items-center space-x-2">
        <div className={`w-3 h-3 rounded-full ${
          socketStatus === 'connected' ? 'bg-green-500' :
          socketStatus === 'connecting' ? 'bg-yellow-500 animate-pulse' :
          socketStatus === 'error' ? 'bg-red-500' :
          'bg-gray-500'
        }`}></div>
        <span className="text-xs text-gray-700 bg-white/70 px-2 py-1 rounded">
          {socketStatus === 'connected' ? 'Connected' :
           socketStatus === 'connecting' ? 'Connecting...' :
           socketStatus === 'error' ? `Error (${reconnectAttempts}/${maxReconnectAttempts})` :
           'Disconnected'}
        </span>
      </div>
      
      {/* Main content */}
      <div className="relative flex-1 z-0">
        <SignedIn>
          <div ref={roomContainerRef} className="h-full w-full"></div>
        </SignedIn>
        <SignedOut>
          <div className="flex items-center justify-center h-full">
            <div className="bg-white p-8 rounded-lg shadow-lg">
              <h2 className="text-2xl font-bold mb-4">Sign In Required</h2>
              <p className="mb-4">Please sign in to join this meeting.</p>
              <button 
                onClick={() => openSignIn()}
                className="bg-blue-500 text-white px-4 py-2 rounded hover:bg-blue-600"
              >
                Sign In
              </button>
            </div>
          </div>
        </SignedOut>
      </div>
      
      {/* Captions area */}
      {showCaptions && inMeeting && (
        <div className="absolute bottom-24 left-0 right-0 flex justify-center z-10 pointer-events-none">
          <div className="bg-black/70 text-white px-6 py-3 rounded-lg max-w-3xl mx-4 text-center pointer-events-auto">
            {captions ? 
              <p className="text-2xl font-medium">{captions}</p> : 
              <p className="text-lg text-gray-300 italic">Waiting for sign language...</p>
            }
          </div>
        </div>
      )}
      
      {/* Controls */}
      <div className="absolute bottom-4 right-4 z-10">
        <button 
          onClick={toggleCaptions}
          className={`flex items-center justify-center p-3 rounded-full shadow-lg
            ${showCaptions ? 'bg-blue-500 text-white' : 'bg-gray-200 text-gray-700'}`}
          title={showCaptions ? 'Hide Captions' : 'Show Captions'}
        >
          <span className="material-icons">{showCaptions ? 'closed_caption' : 'closed_caption_off'}</span>
        </button>
      </div>
      
      {/* Stats (only in development) */}
      {process.env.NODE_ENV === 'development' && (
        <div className="absolute top-2 right-2 bg-white/80 p-2 rounded text-xs z-10">
          <div>Socket: {socketStatus}</div>
          <div>Meeting: {inMeeting ? 'Yes' : 'No'}</div>
          <div>Camera: {cameraActive ? 'Active' : 'Off'}</div>
          <div>Frames: {frameCountRef.current}</div>
          <div>Captions: {captionCountRef.current}</div>
          {lastCaptionTime && (
            <div>Last: {Math.round((Date.now() - lastCaptionTime) / 1000)}s ago</div>
          )}
        </div>
      )}
    </div>
  );
};

export default Page;