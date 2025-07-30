import { DP, DrawingStyles } from '../drawing-primitives.js';

export class BaseApplication {
    constructor(name) {
        this.name = name;
        this.canvas = null;
        this.ctx = null;
        this.isActive = false;
        this.width = 0;
        this.height = 0;
        this.showCursors = true;
        this.handsData = null;
        this.scale = null; // Will be {x, y} when stream info is available
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
        
        document.getElementById('canvas-container').appendChild(this.canvas);
    }

    resize(width, height) {
        if (this.canvas) {
            this.canvas.width = width;
            this.canvas.height = height;
            this.width = width;
            this.height = height;
            if (this.handsData?.stream_info) {
                this.scale = {
                    x: this.width / this.handsData.stream_info.width,
                    y: this.height / this.handsData.stream_info.height
                }
            }
        }
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
        // Draw icon container
        DP.drawIconContainer(ctx, x, y, size, isActive, false);
        
        // Draw icon content
        ctx.save();
        ctx.translate(x, y);
        this.drawIconContent(ctx, size, isActive);
        ctx.restore();
    }
    
    /**
     * Draw the icon content - must be implemented by subclasses
     * @param {CanvasRenderingContext2D} ctx - Canvas context (already translated to icon position)
     * @param {number} size - Size of the icon
     * @param {boolean} isActive - Whether the icon is active
     */
    drawIconContent(ctx, size, isActive) {
        // Abstract method - must be implemented by subclasses
        throw new Error(`${this.constructor.name} must implement drawIconContent()`);
    }

    update(handsData) {
        // Store handsData for future use
        this.handsData = handsData;
        
        // Update scale if stream info is available
        if (!this.scale && handsData?.stream_info) {
            this.scale = {
                x: this.width / handsData.stream_info.width,
                y: this.height / handsData.stream_info.height
            };
        }
        
        // Override in subclasses if needed for additional functionality
    }

    draw() {
        // Override in subclasses
        // Clear canvas by default
        if (this.ctx) {
            DP.clearCanvas(this.ctx);
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

    drawCursors() {
        // Only draw if showCursors is true and we have context and hands data
        if (!this.showCursors || !this.ctx || !this.handsData) return;

        const ctx = this.ctx;
        const cursorRadius = 10;
        const cursorColor = DrawingStyles.colors.accent;

        // Process each hand
        for (const hand of this.handsData.hands) {
            if (!hand.fingers || !hand.fingers.INDEX) continue;

            const indexFinger = hand.fingers.INDEX;
            
            // Check if index finger is straight or nearly straight
            if (!indexFinger.is_fully_bent) {
                const tip = indexFinger.end_point;
                if (!tip) continue;

                // Scale the tip coordinates
                const scaledTip = this.scalePoint(tip);
                const x = scaledTip.x;
                const y = scaledTip.y;

                // Draw cursor circle
                ctx.save();
                ctx.beginPath();
                ctx.arc(x, y, cursorRadius, 0, Math.PI * 2);
                ctx.fillStyle = cursorColor;
                ctx.globalAlpha = 0.8;
                ctx.fill();
                
                // Add a subtle border
                ctx.strokeStyle = cursorColor;
                ctx.globalAlpha = 1;
                ctx.lineWidth = 2;
                ctx.stroke();
                
                ctx.restore();

                // Draw AIR_TAP progress indicator if preAirTapData exists
                const preAirTapData = this.preAirTapData();
                if (preAirTapData && preAirTapData[hand.handedness]) {
                    const airTapInfo = preAirTapData[hand.handedness];
                    const duration = airTapInfo.duration || 0;
                    const maxDuration = airTapInfo.max_duration || 1;
                    const progress = Math.min(duration / maxDuration, 1);
                    
                    // Only draw if there's some progress
                    if (progress > 0) {
                        DP.drawProgressArc(
                            ctx,
                            x,
                            y,
                            cursorRadius + 5,
                            progress,
                            { color: cursorColor }
                        );
                    }
                }
            }
        }

        // Draw ripple effects for air taps
        const airTapData = this.airTapData();
        if (airTapData) {
            for (const [handedness, tapData] of Object.entries(airTapData)) {
                if (tapData.tap_position && tapData.elapsed_since_tap !== undefined && tapData.max_duration) {
                    const scaledPosition = this.scalePoint(tapData.tap_position);
                    const progress = Math.min(tapData.elapsed_since_tap / tapData.max_duration, 1);
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

    /** Returns pre-air-tap data for all hands
     * 
     * @return {Object|null} Object keyed by handedness ('LEFT'/'RIGHT') containing:
     *   - duration {number}: Current duration of the pre-air-tap gesture (in seconds, starting from 0)
     *   - max_duration {number}: Maximum duration allowed (in seconds)
     *   - tap_position {Object}: Position where the tap will occur, with:
     *     - x {number}: X coordinate (0-1 normalized)
     *     - y {number}: Y coordinate (0-1 normalized)
     * @return {null} if no hands are in pre-air-tap state
     */
    preAirTapData() {
        let result = {};
        let hasPreAirTap = false;
        for (const hand of this.handsData.hands) {
            if (!hand.gestures?.PRE_AIR_TAP) continue;
            const data = hand?.gestures_data?.PRE_AIR_TAP
            result[hand.handedness] = data;
            result[hand.handedness].tap_position = {x: data.tap_position[0], y: data.tap_position[1]};
            hasPreAirTap = true;
        }
        return hasPreAirTap ? result : null;
    }

    /** Returns air-tap data for all hands currently performing an air-tap
     * 
     * @return {Object|null} Object keyed by handedness ('LEFT'/'RIGHT') containing:
     *   - tap_position {Object}: Position of the air-tap, with:
     *     - x {number}: X coordinate (0-1 normalized)
     *     - y {number}: Y coordinate (0-1 normalized)
     *   - max_duration {number}: Maximum duration the air tap gesture is active
     *   - elapsed_since_tap {number}: Time elapsed since the air tap occurred (in seconds)
     * @return {null} if no hands are performing air-tap
     */
    airTapData() {
        let result = {};
        let hasAirTap = false;
        for (const hand of this.handsData.hands) {
            if (!hand.gestures?.AIR_TAP) continue;
            const data = hand?.gestures_data?.AIR_TAP;
            if (!data) continue;
            result[hand.handedness] = data;
            result[hand.handedness].tap_position = {x: data.tap_position[0], y: data.tap_position[1]};
            hasAirTap = true;
        }
        return hasAirTap ? result : null;
    }



}
