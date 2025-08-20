import { DP, DrawingStyles } from '../drawing-primitives.js';

// Queue for pending registrations
const pendingRegistrations = [];
let isManagerReady = false;

export class BaseApplication {
    constructor(name, applicationManager) {
        this.name = name;
        this.applicationManager = applicationManager;
        this.canvas = null;
        this.ctx = null;
        this.isActive = false;
        this.width = 0;
        this.height = 0;
        this.showPointers = true;
        this.streamSize = null; // Will be {width, height} when stream info is available
        this.handsData = null;
        this.gestures = {};
        this.scale = null; // Will be {x, y} when stream info and canvas are available
        this.iconSvgPath = null; // SVG path for the icon (from FontAwesome). Use those urls to get the svg: https://site-assets.fontawesome.com/releases/v7.0.0/svgs-full/regular/{icon name}.svg
    }

    createCanvas(width, height) {
        this.canvas = document.createElement('canvas');
        this.canvas.id = `app-canvas-${this.name}`;
        this.canvas.className = 'app-canvas';
        this.canvas.width = width;
        this.canvas.height = height;
        this.ctx = this.canvas.getContext('2d');
        this.width = width;
        this.height = height;
        this.updateScale();

        document.getElementById('canvas-container').appendChild(this.canvas);
    }

    setStreamSize(streamSize) {
        this.streamSize = streamSize;
        this.updateScale();
    }

    updateScale() {
        if (!this.width || !this.height || !this.streamSize) return;
        this.scale = {
            x: this.width / this.streamSize.width,
            y: this.height / this.streamSize.height
        }
    }

    resize(width, height) {
        if (this.canvas) {
            this.canvas.width = width;
            this.canvas.height = height;
            this.width = width;
            this.height = height;
        }
        this.updateScale();
    }

    activate() {
        this.isActive = true;
        if (this.canvas) {
            this.canvas.classList.add('active');
        }
    }

    deactivate() {
        this.isActive = false;
        if (this.canvas) {
            this.canvas.classList.remove('active');
        }
    }

    drawIcon(ctx, x, y, size, isActive) {
        // Only draw if we have an icon SVG path
        if (!this.iconSvgPath) return;
        
        // Draw icon container
        DP.drawIconContainer(ctx, x, y, size, isActive, false);
        
        // Draw icon content
        ctx.save();
        // Translate to center of container
        ctx.translate(x + size / 2, y + size / 2);
        
        // If we have an SVG path, use it
        if (this.iconSvgPath) {
            this.drawSvgIcon(ctx, size, isActive);
        } else {
            // Fallback to custom drawing (for backwards compatibility)
            this.drawIconContent(ctx, size, isActive);
        }
        
        ctx.restore();
    }
    
    /**
     * Draw SVG icon from path
     * @param {CanvasRenderingContext2D} ctx - Canvas context (already translated to icon position)
     * @param {number} size - Size of the icon
     * @param {boolean} isActive - Whether the icon is active
     */
    drawSvgIcon(ctx, size, isActive) {
        const iconSize = size * DrawingStyles.metrics.iconSvgScale; // Use scale from constants
        const scale = iconSize / 640; // FontAwesome SVGs have viewBox of 640x640
        
        ctx.save();
        // Center the icon - translate to top-left corner of where icon should be
        ctx.translate(-320 * scale, -320 * scale);
        ctx.scale(scale, scale);
        
        // Create path from SVG path data
        const path = new Path2D(this.iconSvgPath);
        
        // Always use accent color for icons
        ctx.fillStyle = DrawingStyles.colors.accent;
        // Don't override globalAlpha - respect the opacity set by parent
        
        // Fill the path
        ctx.fill(path);
        
        ctx.restore();
    }
    
    /**
     * Draw the icon content - can be overridden by subclasses for custom icons
     * @param {CanvasRenderingContext2D} ctx - Canvas context (already translated to icon position)
     * @param {number} size - Size of the icon
     * @param {boolean} isActive - Whether the icon is active
     */
    drawIconContent(ctx, size, isActive) {
        // Default implementation - do nothing if no SVG path is set
    }

    /**
     * Check if we have a match for the exit gesture
     * @returns {boolean} True if the app should exit
     */
    hasExitMatch() {
        // Define the exit zone ranges (normalized 0-1)
        const horizontalRange = [0.75, 1.0];  // Right 25% of screen
        const verticalRange = [0.0, 0.33];    // Top 33% of screen
        
        // Check if the hand data is available
        if (!this.handsData || !this.streamSize) {
            return false;
        }
        
        // Check each hand
        for (const [handedness, hand] of [['left', this.handsData.left], ['right', this.handsData.right]]) {
            // Check if DOUBLE_SNAP was just added on this hand
            if (!this.isGestureJustAdded('DOUBLE_SNAP', handedness)) {
                continue;
            }
            
            // Check if this hand is in the exit zone
            if (hand && hand.palm && hand.palm.centroid) {
                const normalizedX = hand.palm.centroid.x / this.streamSize.width;
                const normalizedY = hand.palm.centroid.y / this.streamSize.height;

                if (normalizedX >= horizontalRange[0] && normalizedX <= horizontalRange[1] &&
                    normalizedY >= verticalRange[0] && normalizedY <= verticalRange[1]) {
                    return true;
                }
            }
        }
        
        return false;
    }

    shouldExit() {
        return this.hasExitMatch();
    }

    update(handsData, gestures) {
        // Store handsData and gestures for future use
        this.handsData = handsData;
        this.gestures = gestures;

        // Check if we should exit
        if (this.shouldExit()) {
            this.exit();
            return false; // Indicate that we should stop updating
        }

        // Override in subclasses if needed for additional functionality
        return true; // Indicate that we should continue updating
    }

    /**
     * Check if a gesture is currently active
     * @param {string} gestureName - Name of the gesture to check
     * @param {string} category - Category to check: 'left', 'right', 'both' (default: checks all three)
     * @returns {boolean} True if the gesture is active in the specified category
     */
    isGestureActive(gestureName, category) {
        if (!this.gestures || !this.gestures.active) return false;
        
        if (!category) {
            return this.gestures.active.left.has(gestureName) ||
                   this.gestures.active.right.has(gestureName) ||
                   this.gestures.active.both.has(gestureName);
        }
        
        return this.gestures.active[category]?.has(gestureName) || false;
    }

    /**
     * Check if a gesture was just added (triggered this frame)
     * @param {string} gestureName - Name of the gesture to check
     * @param {string} category - Category to check: 'left', 'right', 'both' (default: checks all three)
     * @returns {boolean} True if the gesture was just added in the specified category
     */
    isGestureJustAdded(gestureName, category) {
        if (!this.gestures || !this.gestures.added) return false;
        
        if (!category) {
            return this.gestures.added.left.has(gestureName) ||
                   this.gestures.added.right.has(gestureName) ||
                   this.gestures.added.both.has(gestureName);
        }
        
        return this.gestures.added[category]?.has(gestureName) || false;
    }

    draw() {
        if (!this.ctx || !this.isActive) return;
        // Override in subclasses
        // Clear canvas by default
        DP.clearCanvas(this.ctx);
    }

    /**
     * Exit the current application and return to the default app
     */
    exit() {
        if (this.applicationManager && this.applicationManager.defaultApp) {
            this.applicationManager.switchToApp(this.applicationManager.defaultApp.name);
        }
    }

    scaleX(value) {
        if (!this.scale) return value;
        return value * this.scale.x;
    }

    scaleY(value) {
        if (!this.scale) return value;
        return value * this.scale.y;
    }

    scalePoint(point) {
        if (!this.scale || !point) return point;
        return {
            x: point.x * this.scale.x,
            y: point.y * this.scale.y
        };
    }


    /**
     * Static method to register an application class
     * Handles timing issues - works even if ApplicationManager isn't loaded yet
     * @param {Function} ApplicationClass - The application class constructor
     * @param {boolean} isDefault - Whether this is the default application
     */
    static register(ApplicationClass, isDefault = false) {
        if (typeof window !== 'undefined' && window.ApplicationManager && window.ApplicationManager.register) {
            // ApplicationManager is ready, register immediately
            window.ApplicationManager.register(ApplicationClass, isDefault);
        } else {
            // Queue the registration for later
            console.log(`Queuing registration for application (default: ${isDefault})`);
            pendingRegistrations.push({ ApplicationClass, isDefault });
            
            // Set up a watcher if not already done
            if (!isManagerReady) {
                BaseApplication.waitForApplicationManager();
            }
        }
    }
    
    /**
     * Wait for ApplicationManager to be available and process pending registrations
     */
    static waitForApplicationManager() {
        const checkInterval = setInterval(() => {
            if (typeof window !== 'undefined' && window.ApplicationManager && window.ApplicationManager.register) {
                // ApplicationManager is now ready
                isManagerReady = true;
                clearInterval(checkInterval);
                
                // Process all pending registrations
                console.log(`ApplicationManager ready, processing ${pendingRegistrations.length} pending registrations`);
                while (pendingRegistrations.length > 0) {
                    const { ApplicationClass, isDefault } = pendingRegistrations.shift();
                    window.ApplicationManager.register(ApplicationClass, isDefault);
                }
            }
        }, 10); // Check every 10ms
        
        // Timeout after 5 seconds
        setTimeout(() => {
            if (!isManagerReady) {
                clearInterval(checkInterval);
                console.error('ApplicationManager not found after 5 seconds. Applications not registered.');
            }
        }, 5000);
    }
    
    /**
     * Notify that ApplicationManager is ready (called by ApplicationManager itself)
     */
    static notifyManagerReady() {
        if (!isManagerReady) {
            isManagerReady = true;
            
            // Process any pending registrations immediately
            while (pendingRegistrations.length > 0) {
                const { ApplicationClass, isDefault } = pendingRegistrations.shift();
                window.ApplicationManager.register(ApplicationClass, isDefault);
            }
        }
    }

    drawPointers(skipLeftHand = false, skipRightHand = false) {
        // Only draw if showPointers is true and we have context and hands data
        if (!this.showPointers || !this.ctx || !this.handsData) return;

        const ctx = this.ctx;
        const pointerRadius = 5;
        const pointerColor = DrawingStyles.colors.accent;

        // Process each hand
        for (const hand of this.handsData.hands) {
            if ((hand.handedness === 'LEFT' && skipLeftHand) || (hand.handedness === 'RIGHT' && skipRightHand)) continue;
            if (!hand.fingers || !hand.fingers.INDEX) continue;

            const indexFinger = hand.fingers.INDEX;
            
            // Check if we should show pointer based on AIR_TAP/pre-AIR_TAP or specific finger conditions
            const hasAirTapData = this.handsData.airTapData && this.handsData.airTapData[hand.handedness];
            const hasPreAirTapData = this.handsData.preAirTapData && this.handsData.preAirTapData[hand.handedness];
            
            // Check the specific finger conditions
            const middleFinger = hand.fingers.MIDDLE;
            const ringFinger = hand.fingers.RING;
            const pinkyFinger = hand.fingers.PINKY;
            
            const fingerConditionsMet = indexFinger.is_nearly_straight_or_straight &&
                                       middleFinger && middleFinger.is_not_straight_at_all &&
                                       ringFinger && ringFinger.is_not_straight_at_all &&
                                       pinkyFinger && pinkyFinger.is_not_straight_at_all;
            
            // Show pointer only if AIR_TAP/pre-AIR_TAP is active OR the specific finger conditions are met
            if (!(hasAirTapData || hasPreAirTapData || fingerConditionsMet)) continue;

            let x, y;

            // Draw AIR_TAP progress indicator if preAirTapData exists
            if (this.handsData.preAirTapData && this.handsData.preAirTapData[hand.handedness]) {
                const tapData = this.handsData.preAirTapData[hand.handedness];
                const duration = tapData.duration || 0;
                const maxDuration = tapData.maxDuration || 1;
                const progress = Math.min(duration / maxDuration, 1);
                const position = this.scalePoint(tapData.tapPosition);

                // Only draw if there's some progress
                if (progress > 0) {
                    DP.drawProgressArc(
                        ctx,
                        position.x,
                        position.y,
                        pointerRadius + 5,
                        progress,
                        { color: pointerColor }
                    );
                }

                // If preAirTap, we use this to draw the pointer position
                x = position.x;
                y = position.y;

            } else {
                if (!indexFinger.landmarks || indexFinger.landmarks.length < 4) continue;
                const tip = indexFinger.landmarks[3];
                if (!tip) continue;
                // Scale the tip coordinates
                const scaledTip = this.scalePoint(tip);
                x = scaledTip.x;
                y = scaledTip.y;
            }

            // Draw pointer circle
            ctx.save();
            ctx.beginPath();
            ctx.arc(x, y, pointerRadius, 0, Math.PI * 2);
            ctx.fillStyle = pointerColor;
            ctx.globalAlpha = 0.8;
            ctx.fill();

            // Add a subtle border
            ctx.strokeStyle = pointerColor;
            ctx.globalAlpha = 1;
            ctx.lineWidth = 2;
            ctx.stroke();

            ctx.restore();

        }

        // Draw ripple effects for air taps
        if (this.handsData.airTapData) {
            for (const [handedness, tapData] of Object.entries(this.handsData.airTapData)) {
                if (tapData.tapPosition && tapData.elapsedSinceTap !== undefined && tapData.maxDuration) {
                    const scaledPosition = this.scalePoint(tapData.tapPosition);
                    const progress = Math.min(tapData.elapsedSinceTap / tapData.maxDuration, 1);
                    DP.drawRippleEffect(
                        this.ctx,
                        scaledPosition.x,
                        scaledPosition.y,
                        progress
                    );
                }
            }
        }
    }

}

// Expose the notify method globally for ApplicationManager to call
if (typeof window !== 'undefined') {
    window.notifyApplicationManagerReady = BaseApplication.notifyManagerReady;
}
