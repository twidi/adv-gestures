import { BaseApplication } from './_base.js';
import { DP, DrawingStyles } from '../drawing-primitives.js';
import { getStroke } from 'https://cdn.skypack.dev/perfect-freehand';

// Helper function to convert stroke points to SVG path data
const average = (a, b) => (a + b) / 2;

function getSvgPathFromStroke(points, closed = true) {
    const len = points.length;
    
    if (len < 4) {
        return ``;
    }
    
    let a = points[0];
    let b = points[1];
    const c = points[2];
    
    let result = `M${a[0].toFixed(2)},${a[1].toFixed(2)} Q${b[0].toFixed(
        2
    )},${b[1].toFixed(2)} ${average(b[0], c[0]).toFixed(2)},${average(
        b[1],
        c[1]
    ).toFixed(2)} T`;
    
    for (let i = 2, max = len - 1; i < max; i++) {
        a = points[i];
        b = points[i + 1];
        result += `${average(a[0], b[0]).toFixed(2)},${average(a[1], b[1]).toFixed(
            2
        )} `;
    }
    
    if (closed) {
        result += 'Z';
    }
    
    return result;
}

export class DrawingApplication extends BaseApplication {
    constructor() {
        super('drawing');
        this.showCursors = false; // We'll handle our own cursors
        
        // Drawing state
        this.isDrawing = false;
        this.isErasing = false;
        this.lastDrawPoint = null;
        
        // Drawing parameters
        this.currentColor = { h: 180, s: 100, l: 50 }; // HSL values
        this.currentStrokeSize = 10;
        
        // Perfect Freehand stroke management
        this.currentStroke = []; // Points being collected for current stroke
        this.completedStrokes = []; // Array of completed strokes
        this.lastPointTime = 0; // For velocity calculation
        
        // Perfect Freehand options
        this.strokeOptions = {
            size: this.currentStrokeSize,
            thinning: 0.6,        // Slightly more thinning for velocity effect
            smoothing: 0.8,       // Higher smoothing for MediaPipe jitter
            streamline: 0.7,      // More streamlining to reduce points
            simulatePressure: false,
            start: {
                taper: 0,
                cap: true,
            },
            end: {
                taper: 0,
                cap: true,
            },
            last: true,
        };
        
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
        
        // Control states
        this.rightHandControllingColor = false;
        this.leftHandControllingSize = false;
        
        // Track last drawing hand
        this.lastDrawingHandedness = null;
        
        // Unified cooldown system
        this.cooldownEndAt = 0;
        this.cooldownSource = null; // 'activation', 'clear', 'color', 'size'
    }

    activate() {
        super.activate();
        // Trigger cooldown on activation
        this.triggerCooldown(1000, 'activation');
    }
    
    resize(width, height) {
        super.resize(width, height);
        // Strokes will automatically adjust to new size
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
        
        if (!this.handsData || !this.handsData.hands) return;


        // Check for CROSSED_FISTS to clear canvas
        if (this.handHasGesture(this.handsData, 'CROSSED_FISTS')) {  // this gesture uses both hands so it's on the handsData object
            this.clearDrawing();
            return;
        }
        
        // Check if we're in cooldown period
        const now = Date.now();
        if (now < this.cooldownEndAt) {
            // During cooldown, only allow continuing the same action that triggered it
            if (this.cooldownSource === 'activation' || this.cooldownSource === 'clear') {
                return; // Block all interactions after activation or clear
            }
            // For color/size cooldowns, allow continuing that action but block others
        }

        // Reset states
        this.isDrawing = false;
        this.isErasing = false;

        // Reset control states
        this.rightHandControllingColor = false;
        this.leftHandControllingSize = false;
        
        // Process all visible hands
        let drawingHandedness = null;
        let hasMultipleDrawingHands = false;
        
        // First pass: check which hands want to draw
        for (const hand of this.handsData.hands) {
            if (!hand.fingers) continue;
            
            const index = hand.fingers.INDEX;
            const pinky = hand.fingers.PINKY;
            
            // Check if this hand wants to draw
            if (index && index.tip_on_thumb && pinky && !pinky.is_nearly_straight_or_straight) {
                if (drawingHandedness === null) {
                    drawingHandedness = hand.handedness;
                } else {
                    hasMultipleDrawingHands = true;
                }
            }
        }
        
        // If multiple hands want to draw, use the last one that was drawing
        if (hasMultipleDrawingHands && this.lastDrawingHandedness) {
            drawingHandedness = this.lastDrawingHandedness;
        }
        
        // Process each hand
        for (const hand of this.handsData.hands) {
            const handId = hand.handedness;
            const isRightHand = hand.handedness === 'RIGHT';
            const isLeftHand = hand.handedness === 'LEFT';

            // Update color from right hand (pointing down ~180°)
            if (isRightHand) {
                const normalizedY = this.getNormalizedYForFlatHand(hand, angle => Math.abs(angle) > (180 - this.ANGLE_MARGIN));
                if (!isNaN(normalizedY)) {
                    this.rightHandControllingColor = true;
                    // Map normalized Y (0-1) to hue (0-360)
                    const newHue = Math.round(normalizedY * 360);
                    if (newHue !== this.currentColor.h) {
                        this.currentColor.h = newHue;
                        this.triggerCooldown(1000, 'color');
                    }
                    continue; // Skip other processing for this hand
                }
            }

            // Update stroke size from left hand (pointing right ~0°)
            if (isLeftHand) {
                const normalizedY = this.getNormalizedYForFlatHand(hand, angle => Math.abs(angle) < this.ANGLE_MARGIN);
                if (!isNaN(normalizedY)) {
                    this.leftHandControllingSize = true;
                    // Map normalized Y (0-1) to stroke size
                    const newSize = this.MIN_STROKE_SIZE + normalizedY * (this.MAX_STROKE_SIZE - this.MIN_STROKE_SIZE);
                    if (Math.abs(newSize - this.currentStrokeSize) > 0.5) {
                        this.currentStrokeSize = newSize;
                        this.triggerCooldown(1000, 'size');
                    }
                    continue; // Skip other processing for this hand
                }
            }

            
            // Process drawing/erasing only if not controlling
            if (!hand.fingers) continue;
            
            // Block drawing/erasing during color/size cooldown
            if (now < this.cooldownEndAt && (this.cooldownSource === 'color' || this.cooldownSource === 'size')) {
                continue;
            }
            
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
                this.handleErasing(index, middle);
                continue;
            }
            
            // Check for drawing mode (index tip on thumb AND pinky not straight)
            if (index && index.tip_on_thumb && pinky && !pinky.is_nearly_straight_or_straight) {
                // Only draw if this is the selected drawing hand
                if (handId === drawingHandedness) {
                    this.isDrawing = true;
                    this.lastDrawingHandedness = handId;
                    this.handleDrawing(hand);
                }
            }
        }
        
        // Reset last draw point if not drawing
        if (!this.isDrawing) {
            this.lastDrawPoint = null;
            // Complete the current stroke if it has points
            if (this.currentStroke.length > 0) {
                this.completeCurrentStroke();
            }
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
            // Complete stroke if we exit drawing area
            if (this.currentStroke.length > 0) {
                this.completeCurrentStroke();
            }
            return;
        }
        
        // Add point to current stroke
        const now = Date.now();
        
        // Calculate pressure based on velocity (for Perfect Freehand)
        let pressure = 0.5; // Default pressure
        if (this.lastDrawPoint && this.lastPointTime) {
            const distance = Math.hypot(
                scaledPoint.x - this.lastDrawPoint.x,
                scaledPoint.y - this.lastDrawPoint.y
            );
            const timeDelta = now - this.lastPointTime;
            if (timeDelta > 0) {
                const velocity = distance / timeDelta;
                // Slower movement = higher pressure = thicker line
                // Adjusted for MediaPipe tracking characteristics
                const normalizedVelocity = velocity / 50; // Lower divisor for more sensitivity
                pressure = Math.max(0.2, Math.min(1, 1 - Math.pow(normalizedVelocity, 0.5)));
            }
        }
        
        // Add point with pressure
        this.currentStroke.push([scaledPoint.x, scaledPoint.y, pressure]);
        
        this.lastDrawPoint = scaledPoint;
        this.lastPointTime = now;
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
        if (!this.isPointInDrawingArea(scaledCenter)) {
            return;
        }
        
        // TODO: Implement erasing with Perfect Freehand strokes
        // For now, we'll need to detect which strokes intersect with the eraser circle
        // of scaledRadius at scaledCenter and remove them from completedStrokes
    }
    
    completeCurrentStroke() {
        if (this.currentStroke.length < 2) {
            // Need at least 2 points for a stroke
            this.currentStroke = [];
            return;
        }
        
        // Update stroke options with current size
        const options = {
            ...this.strokeOptions,
            size: this.currentStrokeSize
        };
        
        // Get stroke outline using Perfect Freehand
        const strokeOutline = getStroke(this.currentStroke, options);
        
        // Convert outline to SVG path data
        const pathData = getSvgPathFromStroke(strokeOutline);
        const path2D = new Path2D(pathData);
        
        // Store completed stroke with Path2D
        this.completedStrokes.push({
            outline: strokeOutline,
            path2D: path2D,
            color: { ...this.currentColor }, // Copy current color
            points: [...this.currentStroke] // Keep original points for erasing
        });

        // Clear current stroke
        this.currentStroke = [];
        this.lastPointTime = 0;
    }
    
    clearDrawing() {
        this.completedStrokes = [];
        this.currentStroke = [];
        this.triggerCooldown(1000, 'clear');
    }
    
    triggerCooldown(duration, source) {
        this.cooldownEndAt = Date.now() + duration;
        this.cooldownSource = source;
    }
    
    isPointInDrawingArea(point) {
        return point.x >= this.DRAWING_AREA_MARGIN_SIDES &&
               point.x <= this.width - this.DRAWING_AREA_MARGIN_SIDES &&
               point.y >= this.DRAWING_AREA_MARGIN_TOP_BOTTOM &&
               point.y <= this.height - this.DRAWING_AREA_MARGIN_TOP_BOTTOM;
    }
    
    draw() {
        if (!this.ctx || !this.isActive) return;
        
        // Clear display canvas
        DP.clearCanvas(this.ctx);
        
        // Draw all completed strokes
        this.drawStrokes();
        
        // Draw current stroke in progress
        this.drawCurrentStroke();
        
        // Draw drawing area border
        this.drawDrawingAreaBorder();
        
        // Draw UI elements
        this.drawIndicators();
        this.drawCursors();
    }
    
    drawStrokes() {
        const ctx = this.ctx;
        
        // Draw all completed strokes
        for (const stroke of this.completedStrokes) {
            this.drawStrokeOutline(ctx, stroke.outline, stroke.color, stroke.path2D);
        }
    }
    
    drawCurrentStroke() {
        if (this.currentStroke.length < 2) return;
        
        const ctx = this.ctx;
        
        // Update stroke options with current size
        const options = {
            ...this.strokeOptions,
            size: this.currentStrokeSize
        };
        
        // Get stroke outline using Perfect Freehand
        const strokeOutline = getStroke(this.currentStroke, options);
        
        // Draw with slight transparency to show it's in progress
        ctx.save();
        ctx.globalAlpha = 0.9;
        this.drawStrokeOutline(ctx, strokeOutline, this.currentColor);
        ctx.restore();
    }
    
    drawStrokeOutline(ctx, outline, color, path2D = null) {
        if (outline.length === 0) return;
        
        ctx.save();
        ctx.fillStyle = `hsl(${color.h}, ${color.s}%, ${color.l}%)`;
        
        if (path2D) {
            // Use pre-calculated Path2D if available
            ctx.fill(path2D);
        } else {
            // Create Path2D on the fly for current stroke
            const pathData = getSvgPathFromStroke(outline);
            const tempPath2D = new Path2D(pathData);
            ctx.fill(tempPath2D);
        }
        
        ctx.restore();
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
        
        // Border
        ctx.strokeStyle = DrawingStyles.colors.accent;
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
        
        // Border
        ctx.strokeStyle = DrawingStyles.colors.accent;
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
        
        // Check if any index finger is outside the drawing area
        let anyIndexOutsideDrawingArea = false;
        for (const hand of this.handsData.hands) {
            if (!hand.fingers || !hand.fingers.INDEX || !hand.fingers.INDEX.end_point) continue;
            const scaledPoint = this.scalePoint(hand.fingers.INDEX.end_point);
            if (!this.isPointInDrawingArea(scaledPoint)) {
                anyIndexOutsideDrawingArea = true;
                break;
            }
        }
        
        // If any index is outside drawing area, show normal cursors
        if (anyIndexOutsideDrawingArea) {
            // Temporarily enable showCursors to allow parent to draw
            const originalShowCursors = this.showCursors;
            this.showCursors = true;
            super.drawCursors();
            this.showCursors = originalShowCursors;
            return;
        }
        
        // Use control states from update method
        
        // Draw cursors for all visible hands
        for (const hand of this.handsData.hands) {
            if (!hand.fingers) continue;
            this.drawHandCursor(hand);
        }
    }
    
    drawHandCursor(hand) {
        const ctx = this.ctx;
        
        const isRightHand = hand.handedness === 'RIGHT';
        const isLeftHand = hand.handedness === 'LEFT';
        
        // Get current control states for this hand
        const isControllingColor = isRightHand && this.getNormalizedYForFlatHand(hand, angle => Math.abs(angle) > (180 - this.ANGLE_MARGIN)) !== undefined;
        const isControllingSize = isLeftHand && this.getNormalizedYForFlatHand(hand, angle => Math.abs(angle) < this.ANGLE_MARGIN) !== undefined;
        
        // Skip if controlling color or size
        if (isControllingColor || isControllingSize) return;
        
        // Draw eraser cursor
        const index = hand.fingers.INDEX;
        const middle = hand.fingers.MIDDLE;
        const ring = hand.fingers.RING;
        const pinky = hand.fingers.PINKY;
        
        // Check if hand is in eraser mode
        const isInEraserMode = index && middle && ring && pinky &&
            index.is_nearly_straight_or_straight &&
            middle.is_nearly_straight_or_straight &&
            !ring.is_nearly_straight_or_straight &&
            !pinky.is_nearly_straight_or_straight;
            
        if (isInEraserMode && index.landmarks && middle.landmarks) {
            
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
                
                // Draw eraser circle
                ctx.save();
                ctx.strokeStyle = '#FFFFFF';
                ctx.lineWidth = 2;
                ctx.setLineDash([5, 5]);
                ctx.beginPath();
                ctx.arc(scaledCenter.x, scaledCenter.y, scaledRadius, 0, Math.PI * 2);
                ctx.stroke();
                ctx.restore();
            }
        }
        
        // Draw drawing cursor only when index tip is on thumb AND pinky not straight
        const canDraw = index && index.tip_on_thumb && pinky && !pinky.is_nearly_straight_or_straight;
        
        if (!isInEraserMode && canDraw && index.end_point) {
            const scaledPoint = this.scalePoint(index.end_point);
            
            // Draw cursor with color fill
            ctx.save();
            ctx.fillStyle = `hsl(${this.currentColor.h}, ${this.currentColor.s}%, ${this.currentColor.l}%)`;
            ctx.beginPath();
            ctx.arc(scaledPoint.x, scaledPoint.y, this.currentStrokeSize / 2, 0, Math.PI * 2);
            ctx.fill();
            ctx.restore();
        }
    }
}
