// Import application manager
import { ApplicationManager } from './application-manager.js';

// Generate UUID v4
function generateUID() {
    return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c) {
        const r = Math.random() * 16 | 0;
        const v = c === 'x' ? r : (r & 0x3 | 0x8);
        return v.toString(16);
    });
}

// Local storage keys
const STORAGE_KEYS = {
    MIRROR: 'adv-gestures-playground-mirror',
    CAMERA: 'adv-gestures-playground-camera'
};

// Maximum log entries
const MAX_LOGS = 100;

// Application state
const state = {
    uid: generateUID(),
    mirror: true, // Will be updated from server config or localStorage in init()
    selectedCamera: localStorage.getItem(STORAGE_KEYS.CAMERA) || null,
    stream: null,
    pc: null,
    eventSource: null,
    streamInfo: null,
    gestures: {
        active: {
            left: new Set(),
            right: new Set(),
            both: new Set()
        },
        added: {
            left: new Set(),
            right: new Set(),
            both: new Set()
        },
        removed: {
            left: new Set(),
            right: new Set(),
            both: new Set()
        }
    },
    appManager: null,
    handledAirTaps: new Set(),  // Set of IDs of air-taps that were handled
    reconnectAttempts: 0,
    reconnectTimeout: null
};

// DOM elements
const elements = {
    video: document.getElementById('video'),
    canvasContainer: document.getElementById('canvas-container'),
    videoContainer: document.getElementById('video-container'),
    videoWrapper: document.querySelector('.video-wrapper'),
    connectionStatus: document.getElementById('connection-status'),
    cameraSelector: document.getElementById('camera-selector'),
    cameraDropdown: document.getElementById('camera-dropdown'),
    previewVideo: document.getElementById('preview-video'),
    cameraSelectBtn: document.getElementById('camera-select-btn'),
    cameraSwitchBtn: document.getElementById('camera-switch'),
    mirrorToggle: document.getElementById('mirror-toggle'),
    logsToggle: document.getElementById('logs-toggle'),
    logsPanel: document.getElementById('logs-panel'),
    logsContent: document.getElementById('logs-content'),
    logsClear: document.getElementById('logs-clear')
};

// Logging functions
function log(level, message) {
    const timestamp = new Date();
    
    // Create log entry element
    const logEntry = document.createElement('div');
    logEntry.className = 'log-entry';
    
    const time = timestamp.toLocaleTimeString('en-US', {
        hour12: false,
        hour: '2-digit',
        minute: '2-digit',
        second: '2-digit',
        fractionalSecondDigits: 2
    });
    
    // Parse message to color gesture names in added/removed messages
    let formattedMessage = message;
    
    // Match "gesture(s) added: GESTURE_NAME1, GESTURE_NAME2 (active: ...)"
    const addedMatch = message.match(/gestures?\s+added:\s+([^(]+)\s+\(active:/);
    if (addedMatch) {
        const gesturesPart = addedMatch[1].trim();
        const coloredGestures = gesturesPart.split(',').map(g => 
            `<span style="color: #00ff00">${g.trim()}</span>`
        ).join(', ');
        formattedMessage = message.replace(gesturesPart, coloredGestures);
    }
    
    // Match "gesture(s) removed: GESTURE_NAME1, GESTURE_NAME2 (active: ...)"
    const removedMatch = message.match(/gestures?\s+removed:\s+([^(]+)\s+\(active:/);
    if (removedMatch) {
        const gesturesPart = removedMatch[1].trim();
        const coloredGestures = gesturesPart.split(',').map(g => 
            `<span style="color: #ff0000">${g.trim()}</span>`
        ).join(', ');
        formattedMessage = message.replace(gesturesPart, coloredGestures);
    }
    
    logEntry.innerHTML = `
        <span class="log-timestamp">${time}</span>
        <span class="log-level ${level}">${level}</span>
        <span class="log-message">${formattedMessage}</span>
    `;
    
    // Add to beginning of logs content
    elements.logsContent.insertBefore(logEntry, elements.logsContent.firstChild);
    
    // Remove old logs beyond MAX_LOGS
    while (elements.logsContent.children.length > MAX_LOGS) {
        elements.logsContent.removeChild(elements.logsContent.lastChild);
    }
}

// Create a special log entry for JSON parsing errors with expandable JSON display
function createJSONErrorLog(message, rawJSON) {
    const timestamp = new Date();
    
    // Create log entry element
    const logEntry = document.createElement('div');
    logEntry.className = 'log-entry log-entry-json-error';
    
    const time = timestamp.toLocaleTimeString('en-US', {
        hour12: false,
        hour: '2-digit',
        minute: '2-digit',
        second: '2-digit',
        fractionalSecondDigits: 2
    });
    
    const entryId = `json-error-${Date.now()}`;
    
    logEntry.innerHTML = `
        <span class="log-timestamp">${time}</span>
        <span class="log-level error">error</span>
        <span class="log-message">${message}</span>
        <button class="json-toggle-btn" data-target="${entryId}">Show JSON</button>
        <div id="${entryId}" class="json-content hidden" title="Click to copy JSON to clipboard">
            <pre>${escapeHtml(rawJSON)}</pre>
        </div>
    `;
    
    // Add click handler for the toggle button
    const toggleBtn = logEntry.querySelector('.json-toggle-btn');
    const jsonContent = logEntry.querySelector(`#${entryId}`);
    
    toggleBtn.addEventListener('click', () => {
        const isHidden = jsonContent.classList.contains('hidden');
        jsonContent.classList.toggle('hidden');
        toggleBtn.textContent = isHidden ? 'Hide JSON' : 'Show JSON';
    });
    
    // Add click handler for the JSON content to copy to clipboard
    jsonContent.addEventListener('click', async () => {
        try {
            await navigator.clipboard.writeText(rawJSON);
            
            // Visual feedback
            const originalBackground = jsonContent.style.background;
            jsonContent.style.background = 'rgba(0, 255, 0, 0.2)';
            
            // Show temporary message
            const copyMessage = document.createElement('div');
            copyMessage.className = 'copy-message';
            copyMessage.textContent = 'Copied to clipboard!';
            jsonContent.appendChild(copyMessage);
            
            setTimeout(() => {
                jsonContent.style.background = originalBackground;
                copyMessage.remove();
            }, 1000);
        } catch (err) {
            log('error', `Failed to copy JSON: ${err.message}`);
        }
    });
    
    return logEntry;
}

// Escape HTML for safe display
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Connection status management
function setConnectionStatus(status, text) {
    elements.connectionStatus.className = `status-${status}`;
    elements.connectionStatus.querySelector('.status-text').textContent = text;
    
    if (status === 'connected') {
        // Hide status after 3 seconds when connected
        setTimeout(() => {
            elements.connectionStatus.classList.add('fade-out');
        }, 3000);
    } else {
        elements.connectionStatus.classList.remove('fade-out');
    }
}

// Cleanup function for existing connections
async function cleanupConnections() {
    // Clear any pending reconnect timeout
    if (state.reconnectTimeout) {
        clearTimeout(state.reconnectTimeout);
        state.reconnectTimeout = null;
    }
    
    // Close EventSource
    if (state.eventSource) {
        state.eventSource.close();
        state.eventSource = null;
    }
    
    // Close peer connection
    if (state.pc) {
        state.pc.close();
        state.pc = null;
    }
    
    // Stop video tracks
    if (state.stream) {
        state.stream.getTracks().forEach(track => track.stop());
        state.stream = null;
    }
}

// Camera management
async function getCameras() {
    try {
        // Request camera permission first
        await navigator.mediaDevices.getUserMedia({ video: true }).then(s => s.getTracks().forEach(t => t.stop()));
        
        const devices = await navigator.mediaDevices.enumerateDevices();
        return devices.filter(d => d.kind === 'videoinput');
    } catch (error) {
        log('error', `Failed to get cameras: ${error.message}`);
        return [];
    }
}

async function selectCamera(deviceId) {
    try {
        // Stop existing stream
        if (state.stream) {
            state.stream.getTracks().forEach(track => track.stop());
        }
        
        // Get new stream with constraints
        // Request high resolution but let camera decide aspect ratio (portrait or landscape)
        const constraints = {
            video: {
                deviceId: deviceId ? { exact: deviceId } : undefined,
                width: { min: 720, ideal: 1280 },
                height: { min: 720, ideal: 1280 },
                // Frame rate - ideal and max at 30fps
                frameRate: { ideal: 30, max: 30 }
            }
        };
        
        state.stream = await navigator.mediaDevices.getUserMedia(constraints);
        
        // Log the actual resolution obtained
        const videoTrack = state.stream.getVideoTracks()[0];
        if (videoTrack) {
            const settings = videoTrack.getSettings();
            log('info', `Camera resolution: ${settings.width}x${settings.height} @ ${settings.frameRate}fps`);
            
            // Listen for track ended event
            videoTrack.addEventListener('ended', () => {
                log('error', 'Video track ended unexpectedly');
            });
        }
        
        return state.stream;
    } catch (error) {
        log('error', `Failed to access camera: ${error.message}`);
        throw error;
    }
}

async function setupCamera() {
    try {
        const cameras = await getCameras();
        
        if (cameras.length === 0) {
            throw new Error('No cameras found');
        }
        
        // Single camera: use it directly
        if (cameras.length === 1) {
            const stream = await selectCamera(cameras[0].deviceId);
            state.selectedCamera = cameras[0].deviceId;
            localStorage.setItem(STORAGE_KEYS.CAMERA, state.selectedCamera);
            log('info', `Camera selected: ${cameras[0].label || 'Camera 1'}`);
            return stream;
        }
        
        // Multiple cameras: check if we have a stored preference
        if (state.selectedCamera) {
            const camera = cameras.find(c => c.deviceId === state.selectedCamera);
            if (camera) {
                // Use stored camera
                const stream = await selectCamera(camera.deviceId);
                elements.cameraSwitchBtn.classList.remove('hidden');
                log('info', `Camera selected: ${camera.label || 'Camera'}`);
                return stream;
            }
        }
        
        // Show camera selector
        elements.cameraDropdown.innerHTML = cameras.map((camera, i) => 
            `<option value="${camera.deviceId}">${camera.label || `Camera ${i + 1}`}</option>`
        ).join('');
        
        // Preview first camera
        const previewStream = await selectCamera(cameras[0].deviceId);
        elements.previewVideo.srcObject = previewStream;
        
        // Handle dropdown change for preview
        elements.cameraDropdown.addEventListener('change', async (e) => {
            const stream = await selectCamera(e.target.value);
            elements.previewVideo.srcObject = stream;
        });
        
        // Show selector
        elements.cameraSelector.classList.remove('hidden');
        
        // Wait for selection
        return new Promise((resolve) => {
            elements.cameraSelectBtn.addEventListener('click', async () => {
                const deviceId = elements.cameraDropdown.value;
                state.selectedCamera = deviceId;
                localStorage.setItem(STORAGE_KEYS.CAMERA, deviceId);
                
                elements.cameraSelector.classList.add('hidden');
                elements.cameraSwitchBtn.classList.remove('hidden');
                
                const camera = cameras.find(c => c.deviceId === deviceId);
                log('info', `Camera selected: ${camera?.label || 'Camera'}`);
                
                resolve(state.stream);
            });
        });
    } catch (error) {
        setConnectionStatus('error', 'Camera access required');
        throw error;
    }
}

// WebRTC setup
async function setupWebRTC() {
    try {
        // Create peer connection
        state.pc = new RTCPeerConnection({
            iceServers: [{ urls: 'stun:stun.l.google.com:19302' }]
        });
        
        // Add video track with specific encoding parameters to prevent resolution changes
        state.stream.getTracks().forEach(track => {
            const sender = state.pc.addTrack(track, state.stream);
            
            // If it's a video track, set encoding parameters to prevent resolution adaptation
            if (track.kind === 'video') {
                const params = sender.getParameters();
                if (!params.encodings) {
                    params.encodings = [{}];
                }
                // Disable automatic scaling and set consistent parameters
                params.encodings[0].scaleResolutionDownBy = 1.0;  // No downscaling
                params.encodings[0].maxFramerate = 30;  // Consistent framerate
                // Set degradationPreference to maintain resolution over framerate
                params.degradationPreference = 'maintain-resolution';
                
                // Note: setParameters might not be supported by all browsers
                if (sender.setParameters) {
                    sender.setParameters(params).catch(err => {
                        log('warning', `Could not set video encoding parameters: ${err.message}`);
                    });
                }
            }
        });
        
        // Create offer with specific video constraints
        const offer = await state.pc.createOffer({
            offerToReceiveVideo: false,  // We're only sending, not receiving
            offerToReceiveAudio: false
        });
        await state.pc.setLocalDescription(offer);
        
        // Send offer to server
        const response = await fetch(`/webrtc/offer?uid=${state.uid}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                sdp: offer.sdp,
                mirror: state.mirror
            })
        });
        
        if (!response.ok) {
            throw new Error(`WebRTC handshake failed: ${response.statusText}`);
        }
        
        const answer = await response.json();
        await state.pc.setRemoteDescription(new RTCSessionDescription({
            type: 'answer',
            sdp: answer.sdp
        }));
        
        log('info', 'WebRTC established');
        
    } catch (error) {
        log('error', `WebRTC setup failed: ${error.message}`);
        throw error;
    }
}

// SSE setup
function setupSSE() {
    state.eventSource = new EventSource(`/sse?uid=${state.uid}&mirror=${state.mirror}`);
    
    const maxReconnectAttempts = 5;
    
    state.eventSource.addEventListener('connected', () => {
        log('info', 'SSE connected');
        setConnectionStatus('connected', 'Connected');
        state.reconnectAttempts = 0;  // Reset reconnect counter on successful connection
    });
    
    state.eventSource.addEventListener('snapshot', (event) => {
        try {
            const data = JSON.parse(event.data);
            processSnapshot(data);
        } catch (error) {
            const logEntry = createJSONErrorLog(`Failed to parse snapshot: ${error.message}`, event.data);
            elements.logsContent.insertBefore(logEntry, elements.logsContent.firstChild);
            
            // Remove old logs beyond MAX_LOGS
            while (elements.logsContent.children.length > MAX_LOGS) {
                elements.logsContent.removeChild(elements.logsContent.lastChild);
            }
        }
    });
    
    state.eventSource.addEventListener('error', (event) => {
        if (event.data) {
            try {
                const error = JSON.parse(event.data);
                log('error', `Server error: ${error.error}`);
            } catch {
                log('error', 'SSE connection error');
            }
        }
        // Check if the connection is closed or connecting (server down)
        if (state.eventSource.readyState === EventSource.CLOSED || 
            state.eventSource.readyState === EventSource.CONNECTING) {
            
            const stateText = state.eventSource.readyState === EventSource.CLOSED ? 'closed' : 'connecting';
            log('error', `SSE connection ${stateText}`);
            
            // Clear any existing reconnect timeout
            if (state.reconnectTimeout) {
                clearTimeout(state.reconnectTimeout);
            }
            
            if (state.reconnectAttempts < maxReconnectAttempts) {
                state.reconnectAttempts++;
                const delay = Math.min(1000 * Math.pow(2, state.reconnectAttempts - 1), 10000); // Exponential backoff, max 10s
                setConnectionStatus('error', `Connection lost - reconnecting in ${delay/1000}s...`);
                
                state.reconnectTimeout = setTimeout(async () => {
                    log('info', `Reconnection attempt ${state.reconnectAttempts}/${maxReconnectAttempts}`);
                    await reconnect();
                }, delay);
            } else {
                setConnectionStatus('error', 'Connection lost - reload page');
            }
        }
    });
}

// Process snapshot data
function processSnapshot(data) {

    data = enhanceData(data);

    // Store stream info and check for resolution changes
    const hadStreamInfo = !!state.streamInfo;
    const previousStreamInfo = state.streamInfo;
    state.streamInfo = data.stream_info;

    // Check if resolution changed
    if (hadStreamInfo && previousStreamInfo && state.streamInfo) {
        if (previousStreamInfo.width !== state.streamInfo.width || 
            previousStreamInfo.height !== state.streamInfo.height) {
            log('warning', `Resolution changed: ${previousStreamInfo.width}x${previousStreamInfo.height} → ${state.streamInfo.width}x${state.streamInfo.height}`);
            updateCanvasSize();
        }
    }

    // Only update canvas size on first stream info
    if (!hadStreamInfo) {
        if (state.streamInfo) {
            log('info', `Stream resolution: ${state.streamInfo.width}x${state.streamInfo.height}`);
        }
        updateCanvasSize();
    }

    // Check for gesture changes
    checkGestureChanges(data);
    
    // Pass data to application manager
    if (state.appManager) {
        state.appManager.update(data, state.gestures);
        state.appManager.draw();
    }
}

// Enhance data with additional properties if needed
function enhanceData(data) {
    if (!data) return null;
    // Save all visible hands. Never access via index 0 for left or index 1 for right
    // If direct access to left/right hands is needed, use data.left and data.right
    data.hands = [data.left, data.right].filter(hand => hand !== null && hand !== undefined && hand.is_visible);

    // Precompute pre-air-tap and air-tap data
    data.airTapData = extractAirTapData(data);
    data.preAirTapData = extractPreAirTapData(data);

    return data;
}

/** Returns pre-air-tap data for all hands
 *
 * @return {Object|null} Object keyed by handedness ('LEFT'/'RIGHT') containing:
 *   - duration {number}: Current duration of the pre-air-tap gesture (in seconds, starting from 0)
 *   - maxDuration {number}: Maximum duration allowed (in seconds)
 *   - tapPosition {Object}: Position where the tap will occur, with:
 *     - x {number}: X coordinate (0-1 normalized)
 *     - y {number}: Y coordinate (0-1 normalized)
 * @return {null} if no hands are in pre-air-tap state
 */
function extractPreAirTapData(data) {
    let result = {};
    let hasPreAirTap = false;
    for (const hand of data.hands) {
        if (!hand.gestures?.PRE_AIR_TAP) continue;
        const data = hand?.gestures_data?.PRE_AIR_TAP;
        if (!data) continue;
        result[hand.handedness] = {
            tapPosition: {x: data.tap_position[0], y: data.tap_position[1]},
            maxDuration: data.max_duration,
            duration: data.duration,
        }
        hasPreAirTap = true;
    }
    return hasPreAirTap ? result : null;
}

/** Returns air-tap data for all hands currently performing an air-tap
 *
 * @return {Object|null} Object keyed by handedness ('LEFT'/'RIGHT') containing:
 *   - tapPosition {Object}: Position of the air-tap, with:
 *     - x {number}: X coordinate (0-1 normalized)
 *     - y {number}: Y coordinate (0-1 normalized)
 *   - maxDuration {number}: Maximum duration the air tap gesture is active
 *   - elapsedSinceTap {number}: Time elapsed since the air tap occurred (in seconds)
 *   - tapId {string}: Unique identifier for this air tap gesture
 *   - alreadyHandled {boolean}: Whether this air tap has already been handled by the application
 * @return {null} if no hands are performing air-tap
 */
function extractAirTapData(data) {
    let result = {};
    let hasAirTap = false;
    for (const hand of data.hands) {
        if (!hand.gestures?.AIR_TAP) continue;
        const data = hand?.gestures_data?.AIR_TAP;
        if (!data) continue;
        result[hand.handedness] = {
            tapPosition: {x: data.tap_position[0], y: data.tap_position[1]},
            maxDuration: data.max_duration,
            elapsedSinceTap: data.elapsed_since_tap,
            tapId: data.tap_id,
            alreadyHandled: state.handledAirTaps.has(data.tap_id),
        };
        hasAirTap = true;
    }
    return hasAirTap ? result : null;
}


// Check for gesture changes and log them
function checkGestureChanges(data) {
    const currentGestures = {
        left: new Set(),
        right: new Set(),
        both: new Set()
    };
    
    // Helper function to extract gestures
    const extractGestures = (gestures, target) => {
        if (gestures) {
            Object.entries(gestures).forEach(([gesture, weight]) => {
                if (weight > 0) {
                    target.add(gesture);
                }
            });
        }
    };
    
    // Extract gestures from each hand
    ['left', 'right'].forEach(handedness => {
        const hand = data[handedness];
        if (hand) {
            extractGestures(hand.gestures, currentGestures[handedness]);
        }
    });
    
    // Extract two-hands gestures from hands level
    extractGestures(data.gestures, currentGestures.both);
    
    // Helper function to check and log changes
    const checkChanges = (category, label) => {
        const added = [...currentGestures[category]].filter(g => !state.gestures.active[category].has(g));
        const removed = [...state.gestures.active[category]].filter(g => !currentGestures[category].has(g));
        
        // Update added and removed gestures in state
        state.gestures.added[category] = new Set(added);
        state.gestures.removed[category] = new Set(removed);
        
        if (added.length > 0) {
            const active = [...currentGestures[category]].join(', ');
            const gesturesText = added.length > 1 ? 'gestures' : 'gesture';
            log('info', `[${label}] ${gesturesText} added: ${added.join(', ')} (active: ${active})`);
        }
        
        if (removed.length > 0) {
            const active = [...currentGestures[category]].join(', ') || 'none';
            const gesturesText = removed.length > 1 ? 'gestures' : 'gesture';
            log('info', `[${label}] ${gesturesText} removed: ${removed.join(', ')} (active: ${active})`);
        }
        
        state.gestures.active[category] = currentGestures[category];
    };
    
    // Check changes for each category
    checkChanges('left', 'left');
    checkChanges('right', 'right');
    checkChanges('both', 'both hands');
}

// Update video and canvas size to fill available space while maintaining aspect ratio
function updateCanvasSize() {
    const videoWidth = elements.video.videoWidth;
    const videoHeight = elements.video.videoHeight;
    
    if (!videoWidth || !videoHeight) return;
    
    // Get container dimensions
    const container = elements.videoContainer;
    const containerWidth = container.clientWidth;
    const containerHeight = container.clientHeight;
    
    // Calculate scale to fit video in container while maintaining aspect ratio
    const videoAspect = videoWidth / videoHeight;
    const containerAspect = containerWidth / containerHeight;
    
    let displayWidth, displayHeight;
    
    if (videoAspect > containerAspect) {
        // Video is wider - fit to width
        displayWidth = containerWidth;
        displayHeight = containerWidth / videoAspect;
    } else {
        // Video is taller - fit to height
        displayHeight = containerHeight;
        displayWidth = containerHeight * videoAspect;
    }
    
    // Set dimensions via CSS custom properties on body
    document.body.style.setProperty('--video-width', `${displayWidth}px`);
    document.body.style.setProperty('--video-height', `${displayHeight}px`);
    
    // Update application canvases
    if (state.appManager) {
        state.appManager.resize(displayWidth, displayHeight);
    }
}


// UI event handlers
elements.mirrorToggle.addEventListener('click', () => {
    state.mirror = !state.mirror;
    localStorage.setItem(STORAGE_KEYS.MIRROR, state.mirror);
    log('info', `Mirror mode: ${state.mirror ? 'ON' : 'OFF'}`);
    
    // Reload page as mirror setting needs to be sent at handshake
    window.location.reload();
});


elements.cameraSwitchBtn.addEventListener('click', () => {
    localStorage.removeItem(STORAGE_KEYS.CAMERA);
    window.location.reload();
});

elements.logsToggle.addEventListener('click', () => {
    const isHidden = elements.logsPanel.classList.contains('hidden');
    elements.logsPanel.classList.toggle('hidden');
    elements.logsToggle.classList.toggle('active', isHidden);
    elements.logsToggle.querySelector('.toggle-state').textContent = isHidden ? 'ON' : 'OFF';
});

elements.logsClear.addEventListener('click', () => {
    elements.logsContent.innerHTML = '';
});

// Keyboard shortcuts
document.addEventListener('keydown', (e) => {
    switch(e.key.toUpperCase()) {
        case 'M':
            elements.mirrorToggle.click();
            break;
    }
});

// Update UI based on initial state
function updateUI() {
    // Mirror toggle
    elements.mirrorToggle.classList.toggle('active', state.mirror);
    elements.mirrorToggle.querySelector('.toggle-state').textContent = state.mirror ? 'ON' : 'OFF';
    elements.video.classList.toggle('mirrored', state.mirror);
}

// Reconnect function
async function reconnect() {
    try {
        // Clean up existing connections
        await cleanupConnections();
        
        // Generate new UID
        state.uid = generateUID();
        log('info', `Generated new UID for reconnection: ${state.uid}`);
        
        // Clear handled air-taps since we have a new session
        state.handledAirTaps.clear();
        
        // Re-acquire camera stream
        const stream = await selectCamera(state.selectedCamera);
        state.stream = stream;
        elements.video.srcObject = stream;
        
        // Re-setup SSE connection with new UID
        setupSSE();
        
        // Wait a bit for SSE to establish
        await new Promise(resolve => setTimeout(resolve, 100));
        
        // Re-setup WebRTC
        await setupWebRTC();
        
        log('info', 'Reconnection successful');
    } catch (error) {
        log('error', `Reconnection failed: ${error.message}`);
        setConnectionStatus('error', 'Reconnection failed - reload page');
    }
}

// Main initialization
async function init() {
    try {
        // Get server default from data attribute
        const defaultMirror = document.body.dataset.defaultMirror === 'true';
        
        // Use localStorage value if it exists, otherwise use server default
        const localMirror = localStorage.getItem(STORAGE_KEYS.MIRROR);
        if (localMirror !== null) {
            state.mirror = localMirror !== 'false';
        } else {
            state.mirror = defaultMirror;
        }
        
        updateUI();
        
        // Initialize application manager with registered applications
        state.appManager = new ApplicationManager(elements.canvasContainer, state.handledAirTaps, state.mirror);
        
        // Setup camera
        const stream = await setupCamera();
        elements.video.srcObject = stream;
        
        // Update canvas size when video metadata is loaded
        elements.video.addEventListener('loadedmetadata', () => {
            // Force a reflow to ensure video dimensions are calculated
            setTimeout(() => {
                updateCanvasSize();
                // Create application canvases after we know the size
                const displayWidth = parseFloat(document.body.style.getPropertyValue('--video-width')) || 640;
                const displayHeight = parseFloat(document.body.style.getPropertyValue('--video-height')) || 480;
                state.appManager.createCanvases(displayWidth, displayHeight);
            }, 100);
        });
        
        // Update canvas size when video actually starts playing
        elements.video.addEventListener('playing', updateCanvasSize);
        
        // Update canvas size on window resize
        window.addEventListener('resize', updateCanvasSize);
        
        // Setup SSE first (creates recognizer on server)
        setupSSE();
        
        // Wait a bit for SSE to establish
        await new Promise(resolve => setTimeout(resolve, 100));
        
        // Setup WebRTC
        await setupWebRTC();
        
    } catch (error) {
        log('error', `Initialization failed: ${error.message}`);
        setConnectionStatus('error', error.message);
    }
}

// Start the application
init();
