import { BaseApplication } from './_base.js';
import { DP, DrawingStyles } from '../drawing-primitives.js';

export class DrawingApplication extends BaseApplication {
    constructor() {
        super('drawing');
        this.showCursors = false; // We'll handle our own cursors
        
        // Drawing state
        this.isDrawing = false;
        this.isErasing = false;
        this.isPaused = false;
        this.lastDrawPoint = null;
        
        // Drawing parameters
        this.currentColor = { h: 180, s: 100, l: 50 }; // HSL values
        this.currentStrokeSize = 5;
        
        // Constants
        this.MIN_STROKE_SIZE = 1;
        this.MAX_STROKE_SIZE = 50;
        this.INDICATOR_SIZE = 60;
        this.INDICATOR_MARGIN = 20;
        
        // Control margins for angle detection
        this.ANGLE_MARGIN = 20; // degrees
        
        // Drawing area margins
        this.DRAWING_AREA_MARGIN_SIDES = 100; // pixels from left and right edges
        this.DRAWING_AREA_MARGIN_TOP_BOTTOM = 40; // pixels from top and bottom edges
        
        // Drawing canvas (separate from display canvas)
        this.drawingCanvas = null;
        this.drawingCtx = null;
        
        // Control states
        this.rightHandHasClosedFist = false;
        this.leftHandHasClosedFist = false;
        this.rightHandControllingColor = false;
        this.leftHandControllingSize = false;
        
        // Cooldown after clearing
        this.clearCooldownDuration = 2000; // 2 seconds in milliseconds
        this.lastClearTimestamp = 0;
        
        // Cooldown after activation
        this.activationCooldownDuration = 2000; // 2 seconds in milliseconds
        this.lastActivationTimestamp = 0;
        
        // Per-hand cooldown after controlling
        this.handControlCooldownDuration = 2000; // 2 seconds in milliseconds
        this.lastRightHandControlTimestamp = 0;
        this.lastLeftHandControlTimestamp = 0;
    }

    activate() {
        super.activate();
        // Create drawing canvas if not exists
        if (!this.drawingCanvas && this.width && this.height) {
            this.createDrawingCanvas();
        }
        // Set activation timestamp
        this.lastActivationTimestamp = Date.now();
    }
    
    createDrawingCanvas() {
        this.drawingCanvas = document.createElement('canvas');
        this.drawingCanvas.width = this.width;
        this.drawingCanvas.height = this.height;
        this.drawingCtx = this.drawingCanvas.getContext('2d');
    }
    
    resize(width, height) {
        super.resize(width, height);
        // Resize drawing canvas and preserve content
        if (this.drawingCanvas) {
            const imageData = this.drawingCtx.getImageData(0, 0, this.drawingCanvas.width, this.drawingCanvas.height);
            this.drawingCanvas.width = width;
            this.drawingCanvas.height = height;
            this.drawingCtx.putImageData(imageData, 0, 0);
        }
    }
    
    drawIconContent(ctx, size, isActive) {
        // Draw a pencil icon
        const scale = size / 72;
        ctx.save();
        ctx.scale(scale, scale);
        
        // Pencil body
        ctx.fillStyle = '#FFD700'; // Gold color
        ctx.beginPath();
        ctx.moveTo(20, 50);
        ctx.lineTo(50, 20);
        ctx.lineTo(58, 28);
        ctx.lineTo(28, 58);
        ctx.closePath();
        ctx.fill();
        
        // Pencil tip
        ctx.fillStyle = '#444444';
        ctx.beginPath();
        ctx.moveTo(20, 50);
        ctx.lineTo(12, 58);
        ctx.lineTo(20, 66);
        ctx.lineTo(28, 58);
        ctx.closePath();
        ctx.fill();
        
        // Pencil point
        ctx.fillStyle = '#000000';
        ctx.beginPath();
        ctx.arc(16, 58, 2, 0, Math.PI * 2);
        ctx.fill();
        
        // Pencil eraser
        ctx.fillStyle = '#FF69B4';
        ctx.beginPath();
        ctx.moveTo(50, 20);
        ctx.lineTo(58, 12);
        ctx.lineTo(66, 20);
        ctx.lineTo(58, 28);
        ctx.closePath();
        ctx.fill();
        
        ctx.restore();
    }

    handHasGesture(hand, gesture) {
        return hand && hand.gestures && gesture in hand.gestures && hand.gestures[gesture] > 0;
    }

    getNormalizedYForFlatHand(hand, angleChecker) {
        if (!hand || hand.main_direction_angle === undefined) { return NaN; }
        const angle = hand.main_direction_angle;
        // Check if pointing right (around 0°)
        if (!angleChecker(angle)) { return NaN; }
        
        // Check that at least 3 fingers (excluding thumb) are straight
        if (!hand.fingers) { return NaN; }
        const straightFingers = [
            hand.fingers.INDEX?.is_nearly_straight_or_straight,
            hand.fingers.MIDDLE?.is_nearly_straight_or_straight,
            hand.fingers.RING?.is_nearly_straight_or_straight,
            hand.fingers.PINKY?.is_nearly_straight_or_straight
        ].filter(isStraight => isStraight === true).length;
        
        if (straightFingers < 3) { return NaN; }
        
        // Map y position to stroke size
        if (!hand.palm || !hand.palm.centroid) { return NaN; }

        const scaledCentroid = this.scalePoint(hand.palm.centroid);

        // Use drawing area margins as control bounds
        const minY = this.DRAWING_AREA_MARGIN_TOP_BOTTOM;
        const maxY = this.height - this.DRAWING_AREA_MARGIN_TOP_BOTTOM;
        const clampedY = Math.max(minY, Math.min(maxY, scaledCentroid.y));

        // Normalize within the drawing area (0-1)
        const normalizedY = (clampedY - minY) / (maxY - minY);

        // Return inverted Y (top = max, bottom = min)
        return 1 - normalizedY;
    }

    update(handsData) {
        super.update(handsData);
        
        if (!this.handsData || !this.handsData.hands || !this.drawingCtx) return;

        // Find left and right hands
        const leftHand = this.handsData.hands.find(h => h.handedness === 'LEFT');
        const rightHand = this.handsData.hands.find(h => h.handedness === 'RIGHT');

        // Check for CROSSED_FISTS to clear canvas
        if (this.handHasGesture(this.handsData, 'CROSSED_FISTS')) {  // this gesture uses both hands so it's on the handsData object
            this.clearDrawing();
            return;
        }
        
        // Check if we're in cooldown period after clearing or activation
        const now = Date.now();
        if (now - this.lastClearTimestamp < this.clearCooldownDuration ||
            now - this.lastActivationTimestamp < this.activationCooldownDuration) {
            return; // Ignore all interactions during cooldown
        }

        // Reset states
        this.isDrawing = false;
        this.isErasing = false;
        this.isPaused = false;

        // Reset control states
        this.rightHandHasClosedFist = this.handHasGesture(rightHand, 'CLOSED_FIST');
        this.leftHandHasClosedFist = this.handHasGesture(leftHand, 'CLOSED_FIST');
        this.rightHandControllingColor = false;
        this.leftHandControllingSize = false;

        this.isPaused = this.rightHandHasClosedFist || this.leftHandHasClosedFist;

        // Update color from right hand (pointing down ~180°)
        if (!this.rightHandHasClosedFist) {
            const normalizedY = this.getNormalizedYForFlatHand(rightHand, angle => Math.abs(angle) > (180 - this.ANGLE_MARGIN));
            if (!isNaN(normalizedY)) {
                this.rightHandControllingColor = true;
                this.lastRightHandControlTimestamp = now;
                // Map normalized Y (0-1) to hue (0-360)
                this.currentColor.h = Math.round(normalizedY * 360);
            }
        }

        // Update stroke size from left hand (pointing right ~0°)
        if (!this.leftHandHasClosedFist) {
            const normalizedY = this.getNormalizedYForFlatHand(leftHand, angle => Math.abs(angle) < this.ANGLE_MARGIN);
            if (!isNaN(normalizedY)) {
                this.leftHandControllingSize = true;
                this.lastLeftHandControlTimestamp = now;
                // Map normalized Y (0-1) to stroke size
                this.currentStrokeSize = this.MIN_STROKE_SIZE + normalizedY * (this.MAX_STROKE_SIZE - this.MIN_STROKE_SIZE);
            }
        }

        
        // Process drawing/erasing for each hand
        for (const hand of this.handsData.hands) {
            if (!hand.fingers) continue;

            // Skip hand if it's controlling color (right hand pointing down), has closed fist, or is in cooldown
            if (hand.handedness === 'RIGHT' && 
                (this.rightHandControllingColor || 
                 this.rightHandHasClosedFist || 
                 (now - this.lastRightHandControlTimestamp < this.handControlCooldownDuration))) continue;
            
            // Skip hand if it's controlling size (left hand pointing right), has closed fist, or is in cooldown
            if (hand.handedness === 'LEFT' && 
                (this.leftHandControllingSize || 
                 this.leftHandHasClosedFist ||
                 (now - this.lastLeftHandControlTimestamp < this.handControlCooldownDuration))) continue;
            
            // Check for eraser mode
            const index = hand.fingers.INDEX;
            const middle = hand.fingers.MIDDLE;
            const ring = hand.fingers.RING;
            const pinky = hand.fingers.PINKY;
            
            if (index && middle && ring && pinky &&
                index.is_nearly_straight_or_straight &&
                middle.is_nearly_straight_or_straight &&
                !ring.is_nearly_straight_or_straight &&
                !pinky.is_nearly_straight_or_straight) {
                
                this.isErasing = true;
                if (!this.isPaused) {
                    this.handleErasing(index, middle);
                }
                continue;
            }
            
            // Check for drawing mode (index tip on thumb)
            if (index && index.tip_on_thumb && !this.isPaused) {
                this.isDrawing = true;
                this.handleDrawing(hand);
            }
        }
        
        // Reset last draw point if not drawing
        if (!this.isDrawing) {
            this.lastDrawPoint = null;
        }
    }
    
    handleDrawing(hand) {
        const index = hand.fingers.INDEX;
        
        if (!index || !index.end_point) return;
        
        // Use index end point for drawing
        const scaledPoint = this.scalePoint(index.end_point);
        
        // Check if point is within drawing area
        if (!this.isPointInDrawingArea(scaledPoint)) {
            this.lastDrawPoint = null;
            return;
        }
        
        // Draw
        this.drawingCtx.save();
        this.drawingCtx.strokeStyle = `hsl(${this.currentColor.h}, ${this.currentColor.s}%, ${this.currentColor.l}%)`;
        this.drawingCtx.lineWidth = this.currentStrokeSize;
        this.drawingCtx.lineCap = 'round';
        this.drawingCtx.lineJoin = 'round';
        
        if (this.lastDrawPoint) {
            this.drawingCtx.beginPath();
            this.drawingCtx.moveTo(this.lastDrawPoint.x, this.lastDrawPoint.y);
            this.drawingCtx.lineTo(scaledPoint.x, scaledPoint.y);
            this.drawingCtx.stroke();
        }
        
        this.drawingCtx.restore();
        
        this.lastDrawPoint = scaledPoint;
    }
    
    handleErasing(indexFinger, middleFinger) {
        if (!indexFinger.landmarks || !middleFinger.landmarks) return;
        
        const indexTip = indexFinger.landmarks[3];
        const middleTip = middleFinger.landmarks[3];
        
        if (!indexTip || !middleTip) return;
        
        // Calculate center and radius
        const center = {
            x: (indexTip.x + middleTip.x) / 2,
            y: (indexTip.y + middleTip.y) / 2
        };
        
        // Calculate distance for radius
        const distance = Math.sqrt(
            Math.pow(indexTip.x - middleTip.x, 2) + 
            Math.pow(indexTip.y - middleTip.y, 2)
        );
        
        // Scale to canvas coordinates
        const scaledCenter = this.scalePoint(center);
        const scaledRadius = this.scaleX(distance) / 2;
        
        // Only erase within drawing area
        if (!this.isPointInDrawingArea(scaledCenter)) return;
        
        // Erase with clipping to drawing area
        this.drawingCtx.save();
        
        // Set clipping region to drawing area
        this.setDrawingAreaClip(this.drawingCtx);
        
        this.drawingCtx.globalCompositeOperation = 'destination-out';
        this.drawingCtx.beginPath();
        this.drawingCtx.arc(scaledCenter.x, scaledCenter.y, scaledRadius, 0, Math.PI * 2);
        this.drawingCtx.fill();
        this.drawingCtx.restore();
    }
    
    clearDrawing() {
        if (this.drawingCtx) {
            this.drawingCtx.clearRect(0, 0, this.drawingCanvas.width, this.drawingCanvas.height);
            this.lastClearTimestamp = Date.now();
        }
    }
    
    isPointInDrawingArea(point) {
        return point.x >= this.DRAWING_AREA_MARGIN_SIDES &&
               point.x <= this.width - this.DRAWING_AREA_MARGIN_SIDES &&
               point.y >= this.DRAWING_AREA_MARGIN_TOP_BOTTOM &&
               point.y <= this.height - this.DRAWING_AREA_MARGIN_TOP_BOTTOM;
    }
    
    setDrawingAreaClip(ctx) {
        const x = this.DRAWING_AREA_MARGIN_SIDES;
        const y = this.DRAWING_AREA_MARGIN_TOP_BOTTOM;
        const width = this.width - 2 * this.DRAWING_AREA_MARGIN_SIDES;
        const height = this.height - 2 * this.DRAWING_AREA_MARGIN_TOP_BOTTOM;
        
        ctx.beginPath();
        DP.roundedRect(ctx, x, y, width, height, 20);
        ctx.clip();
    }
    
    draw() {
        if (!this.ctx || !this.isActive) return;
        
        // Clear display canvas
        DP.clearCanvas(this.ctx);
        
        // Draw the drawing canvas
        if (this.drawingCanvas) {
            this.ctx.drawImage(this.drawingCanvas, 0, 0);
        }
        
        // Draw drawing area border
        this.drawDrawingAreaBorder();
        
        // Draw UI elements
        this.drawIndicators();
        this.drawCursors();
    }
    
    drawIndicators() {
        const ctx = this.ctx;
        
        // Color indicator position (top right)
        const colorX = this.width - this.INDICATOR_SIZE - this.INDICATOR_MARGIN;
        const colorY = this.INDICATOR_MARGIN;
        
        // Draw color indicator
        ctx.save();
        
        // Background
        ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
        DP.roundedRect(ctx, colorX, colorY, this.INDICATOR_SIZE, this.INDICATOR_SIZE, 10);
        ctx.fill();
        
        // Color preview
        ctx.fillStyle = `hsl(${this.currentColor.h}, ${this.currentColor.s}%, ${this.currentColor.l}%)`;
        DP.roundedRect(ctx, colorX + 5, colorY + 5, this.INDICATOR_SIZE - 10, this.INDICATOR_SIZE - 10, 8);
        ctx.fill();
        
        // Border (red if paused)
        ctx.strokeStyle = this.isPaused ? '#FF0000' : DrawingStyles.colors.accent;
        ctx.lineWidth = 2;
        DP.roundedRect(ctx, colorX, colorY, this.INDICATOR_SIZE, this.INDICATOR_SIZE, 10);
        ctx.stroke();
        
        // Stroke size indicator position (below color indicator)
        const sizeX = colorX;
        const sizeY = colorY + this.INDICATOR_SIZE + 10;
        
        // Background
        ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
        DP.roundedRect(ctx, sizeX, sizeY, this.INDICATOR_SIZE, this.INDICATOR_SIZE, 10);
        ctx.fill();
        
        // Stroke size preview (circle)
        ctx.fillStyle = '#FFFFFF';
        ctx.beginPath();
        ctx.arc(
            sizeX + this.INDICATOR_SIZE / 2,
            sizeY + this.INDICATOR_SIZE / 2,
            this.currentStrokeSize / 2,
            0,
            Math.PI * 2
        );
        ctx.fill();
        
        // Border (red if paused)
        ctx.strokeStyle = this.isPaused ? '#FF0000' : DrawingStyles.colors.accent;
        ctx.lineWidth = 2;
        DP.roundedRect(ctx, sizeX, sizeY, this.INDICATOR_SIZE, this.INDICATOR_SIZE, 10);
        ctx.stroke();
        
        ctx.restore();
    }
    
    drawDrawingAreaBorder() {
        const ctx = this.ctx;
        const x = this.DRAWING_AREA_MARGIN_SIDES;
        const y = this.DRAWING_AREA_MARGIN_TOP_BOTTOM;
        const width = this.width - 2 * this.DRAWING_AREA_MARGIN_SIDES;
        const height = this.height - 2 * this.DRAWING_AREA_MARGIN_TOP_BOTTOM;
        
        ctx.save();
        ctx.strokeStyle = DrawingStyles.colors.accent;
        ctx.lineWidth = 4;
        ctx.setLineDash([10, 10]);
        DP.roundedRect(ctx, x, y, width, height, 20);
        ctx.stroke();
        ctx.restore();
    }
    
    drawCursors() {
        if (!this.handsData || !this.handsData.hands) return;
        
        const ctx = this.ctx;
        
        // Use control states from update method
        
        for (const hand of this.handsData.hands) {
            if (!hand.fingers) continue;
            
            // Skip hand if it's controlling color (right hand pointing down) or has closed fist
            if (hand.handedness === 'RIGHT' && (this.rightHandControllingColor || this.rightHandHasClosedFist)) continue;
            
            // Skip hand if it's controlling size (left hand pointing right) or has closed fist
            if (hand.handedness === 'LEFT' && this.leftHandControllingSize || this.leftHandHasClosedFist) continue;
            
            // Draw eraser cursor
            const index = hand.fingers.INDEX;
            const middle = hand.fingers.MIDDLE;
            
            if (this.isErasing && index && middle &&
                index.is_nearly_straight_or_straight &&
                middle.is_nearly_straight_or_straight &&
                index.landmarks && middle.landmarks) {
                
                const indexTip = index.landmarks[3];
                const middleTip = middle.landmarks[3];
                
                if (indexTip && middleTip) {
                    const center = {
                        x: (indexTip.x + middleTip.x) / 2,
                        y: (indexTip.y + middleTip.y) / 2
                    };
                    
                    const distance = Math.sqrt(
                        Math.pow(indexTip.x - middleTip.x, 2) + 
                        Math.pow(indexTip.y - middleTip.y, 2)
                    );
                    
                    const scaledCenter = this.scalePoint(center);
                    const scaledRadius = this.scaleX(distance) / 2;
                    
                    // Draw eraser circle (red if paused)
                    ctx.save();
                    ctx.strokeStyle = this.isPaused ? '#FF0000' : '#FFFFFF';
                    ctx.lineWidth = 2;
                    ctx.setLineDash([5, 5]);
                    ctx.beginPath();
                    ctx.arc(scaledCenter.x, scaledCenter.y, scaledRadius, 0, Math.PI * 2);
                    ctx.stroke();
                    ctx.restore();
                }
            }
            
            // Draw drawing cursor only when index tip is on thumb
            if (!this.isErasing) {
                const thumb = hand.fingers.THUMB;
                const index = hand.fingers.INDEX;
                
                if (index && index.tip_on_thumb && index.end_point) {
                    const scaledPoint = this.scalePoint(index.end_point);
                        // Draw cursor
                        ctx.save();
                        
                        if (this.isPaused) {
                            // Red circle when paused
                            ctx.strokeStyle = '#FF0000';
                            ctx.lineWidth = 2;
                            ctx.beginPath();
                            ctx.arc(scaledPoint.x, scaledPoint.y, this.currentStrokeSize / 2, 0, Math.PI * 2);
                            ctx.stroke();
                        } else {
                            // Colored fill when actively drawing
                            ctx.fillStyle = `hsl(${this.currentColor.h}, ${this.currentColor.s}%, ${this.currentColor.l}%)`;
                            ctx.beginPath();
                            ctx.arc(scaledPoint.x, scaledPoint.y, this.currentStrokeSize / 2, 0, Math.PI * 2);
                            ctx.fill();
                        }
                        
                        ctx.restore();
                }
            }
        }
    }
}
