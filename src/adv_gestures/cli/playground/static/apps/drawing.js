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
    constructor(applicationManager) {
        super('drawing', applicationManager);
        // FontAwesome pen icon SVG path (regular style)
        this.iconSvgPath = "M100.4 417.8C104.5 403.2 112.2 389.9 123 379.1L417 85.1C430.4 71.6 448.8 64 468 64C487.2 64 505.6 71.6 519.1 85.2L554.8 120.9C568.4 134.4 576 152.8 576 172C576 191.2 568.4 209.6 554.8 223.1L260.8 517.1C250.1 527.8 236.7 535.6 222.1 539.7L94.4 575.1C86.1 577.4 77.1 575.1 71 568.9C64.9 562.7 62.5 553.8 64.8 545.5L100.4 417.8zM450.8 119.1L397.9 172L468 242.1L520.9 189.2C525.5 184.6 528 178.5 528 172C528 165.5 525.4 159.4 520.9 154.8L485.2 119.1C480.6 114.6 474.4 112 468 112C461.6 112 455.4 114.6 450.8 119.1zM364 205.9L156.9 413.1C152 418 148.5 424 146.6 430.7L122.5 517.6L209.4 493.5C216 491.7 222.1 488.1 227 483.2L434.1 276L364 205.9z";

        // Drawing state
        this.isLeftHandAdjustingControl = false;
        this.isRightHandAdjustingControl = false;
        this.lastDrawingPoint = null;
        
        // Drawing parameters
        this.currentColor = { h: 180, s: 100, l: 50 }; // HSL values
        this.currentStrokeSize = 10;
        
        // Perfect Freehand stroke management
        this.strokes = []; // All strokes (last one is current if drawing)
        this.lastDrawingPointTime = 0; // For velocity calculation
        
        // activation and location tracking
        this.activatorHand = null; // Hand showing CLOSED_FIST
        this.drawingHand = null; // Hand that can draw (opposite of activator)
        this.drawingPoint = null; // Last drawing point
        this.erasingCircle = null; // Circle for erasing

        
        // Perfect Freehand options
        this.strokeOptions = {
            size: this.currentStrokeSize,
            thinning: 0.5,        // Slightly more thinning for velocity effect
            smoothing: 0.5,       // Higher smoothing for MediaPipe jitter
            streamline: 0.5,      // More streamlining to reduce points
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
        this.INDICATOR_HIDE_DELAY = 2000; // Delay in ms before hiding indicators
        this.SHOW_DETECTED_POINTS = false; // Toggle to show/hide white dots for detected points
        this.MIN_POINT_DISTANCE = 10; // Minimum distance in pixels between points
        this.MAX_POINT_DISTANCE = 100; // Maximum distance in pixels between points (to filter jumps)
        this.STROKE_CONTINUATION_TIME = 250; // Time in ms to continue a previous stroke

        // Control margins for angle detection
        this.ANGLE_MARGIN = 20; // degrees
        
        // Control height usage - only use the central portion of screen height for controls
        this.CONTROL_HEIGHT_USAGE = 0.8; // Use 80% of height (ignore top/bottom 10%)

        
        // Unified cooldown system
        this.cooldownEndAt = 0;
        this.cooldownSource = null; // 'activation', 'clear', 'color', 'size'
        
        // Indicator visibility tracking
        this.indicatorsVisibleUntil = 0;
        this.previousHandStates = {
            left: { visible: false, state: null },
            right: { visible: false, state: null }
        };
    }

    activate() {
        super.activate();
        // Trigger cooldown on activation
        this.triggerCooldown(1000, 'activation');
        // Show all indicators on activation
        this.showIndicators();
    }
    
    resize(width, height) {
        super.resize(width, height);
        // Strokes will automatically adjust to new size
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

        // Use only central portion of screen for controls
        const marginRatio = (1 - this.CONTROL_HEIGHT_USAGE) / 2; // e.g., 0.1 for top and bottom
        const minY = this.height * marginRatio;
        const maxY = this.height * (1 - marginRatio);
        const clampedY = Math.max(minY, Math.min(maxY, scaledCentroid.y));

        // Normalize within the control zone (0-1)
        const normalizedY = (clampedY - minY) / (maxY - minY);

        // Return inverted Y (top = max, bottom = min)
        return 1 - normalizedY;
    }

    update(handsData, gestures) {
        super.update(handsData, gestures);
        
        // Check for DOUBLE_SNAP gesture to exit the app
        if (this.isGestureJustAdded('DOUBLE_SNAP') && !this.isGestureActive('CLOSED_FIST')) {
            this.exit();
            return;
        }
        
        if (!this.handsData || !this.handsData.hands) return;


        // Check for CROSSED_FISTS to clear canvas
        if (this.isGestureJustAdded('CROSSED_FISTS', 'both')) {
            this.clearDrawing();
            return;
        }
        
        // Check if we're in cooldown period
        const now = Date.now();
        if (now < this.cooldownEndAt) {
            // During cooldown, only allow continuing the same action that triggered it
            if (this.cooldownSource === 'activation' || this.cooldownSource === 'clear' || this.cooldownSource === 'swipe') {
                return; // Block all interactions after activation, clear, or swipe
            }
            // For color/size cooldowns, allow continuing that action but block others
        }

        // Reset states
        this.activatorHand = null;
        this.drawingHand = null;
        this.isLeftHandAdjustingControl = false;
        this.isRightHandAdjustingControl = false;
        this.drawingPoint = null;
        this.erasingCircle = null;

        
        // Phase 1: Detect CLOSED_FIST activation
        const leftFist = this.isGestureActive('CLOSED_FIST', 'left');
        const rightFist = this.isGestureActive('CLOSED_FIST', 'right');
        
        if (leftFist && !rightFist) {
            this.activatorHand = this.handsData.left;
            this.drawingHand = this.handsData.right;
        } else if (rightFist && !leftFist) {
            this.activatorHand = this.handsData.right;
            this.drawingHand = this.handsData.left;
        } else {
            // No fist or multiple fists = no active drawing hand
            this.activatorHand = null;
            this.drawingHand = null;
        }

        // Check for SWIPE gesture when no activator hand
        if (!this.activatorHand) {
            for (const handSide of ['left', 'right']) {
                if (this.isGestureJustAdded('SWIPE', handSide)) {
                    const swipeData = this.handsData[handSide]?.gestures_data?.SWIPE;
                    if (swipeData && swipeData.mode) {
                        this.handleSwipe(swipeData.mode);
                        return; // Exit early to avoid processing other gestures during swipe
                    }
                }
            }
        }

        if (this.drawingHand) {
            // Block drawing/erasing during color/size cooldown
            if (!(now < this.cooldownEndAt && (this.cooldownSource === 'color' || this.cooldownSource === 'size'))) {
                this.erasingCircle = this.getErasingCircle(this.drawingHand);
                if (this.erasingCircle) {
                    this.handleErasing();
                } else {
                    this.drawingPoint = this.getDrawingPoint(this.drawingHand);
                    if (this.drawingPoint) {
                        this.handleDrawing();
                    }
                }
            }

        } else {
            // Only allow color/size controls if drawing mode is NOT active

            // Adjust color from right hand (pointing down ~180°)
            let normalizedY = this.getNormalizedYForFlatHand(this.handsData.right, angle => Math.abs(angle) > (180 - this.ANGLE_MARGIN));
            if (!isNaN(normalizedY)) {
                this.isRightHandAdjustingControl = true;
                // Map normalized Y (0-1) to hue (0-360)
                const newHue = Math.round(normalizedY * 360);
                if (newHue !== this.currentColor.h) {
                    this.currentColor.h = newHue;
                    this.triggerCooldown(1000, 'color');
                    this.showIndicators();
                }
            }

            normalizedY = this.getNormalizedYForFlatHand(this.handsData.left, angle => Math.abs(angle) < this.ANGLE_MARGIN);
            if (!isNaN(normalizedY)) {
                this.isLeftHandAdjustingControl = true;
                // Map normalized Y (0-1) to stroke size
                const newSize = this.MIN_STROKE_SIZE + normalizedY * (this.MAX_STROKE_SIZE - this.MIN_STROKE_SIZE);
                if (Math.abs(newSize - this.currentStrokeSize) > 0.5) {
                    this.currentStrokeSize = newSize;
                    this.triggerCooldown(1000, 'size');
                    this.showIndicators();
                }
            }
        }

        // Check for hand state changes
        this.checkHandStateChanges();
        
        // Reset last draw point if not drawing
        if (!this.drawingPoint) {
            this.lastDrawingPoint = null;
            // Mark current stroke as completed if there is one
            if (this.strokes.length > 0) {
                const lastStroke = this.strokes.at(-1);
                if (!lastStroke.completedAt && lastStroke.points.length > 0) {
                    lastStroke.completedAt = Date.now();
                }
            }
        }
    }
    
    checkHandStateChanges() {
        const now = Date.now();
        
        for (const handSide of ['left', 'right']) {
            const hand = this.handsData?.[handSide];
            const prevState = this.previousHandStates[handSide];
            
            // Determine current hand state
            let currentState = null;
            const isVisible = hand?.is_visible || false;
            
            if (isVisible) {
                const isFist = this.isGestureActive('CLOSED_FIST', handSide);
                const isDrawingHand = this.drawingHand && this.drawingHand.handedness === (handSide === 'left' ? 'LEFT' : 'RIGHT');
                const erasingCircle = hand ? (hand === this.drawingHand ? this.erasingCircle : this.getErasingCircle(hand)) : null;
                const drawingPoint = hand ? (hand === this.drawingHand ? this.drawingPoint : this.getDrawingPoint(hand)) : null;
                
                if (isFist) {
                    currentState = 'fist';
                } else if (erasingCircle) {
                    currentState = 'erasing';
                } else if (drawingPoint) {
                    currentState = 'drawing';
                } else {
                    currentState = 'normal';
                }
            }
            
            // Check if state changed
            if (prevState.visible !== isVisible || prevState.state !== currentState) {
                this.showIndicators();
                this.previousHandStates[handSide] = {
                    visible: isVisible,
                    state: currentState
                };
            }
        }
    }
    
    showIndicators() {
        this.indicatorsVisibleUntil = Date.now() + this.INDICATOR_HIDE_DELAY;
    }

    getDrawingPoint(hand) {
        if (!hand || !hand.is_visible || !hand.fingers || !hand.fingers.INDEX || !hand.fingers.INDEX.landmarks || hand.fingers.INDEX.landmarks.length < 4) return null;
        const scaledPoint = this.scalePoint(hand.fingers.INDEX.landmarks[3]);
        return scaledPoint;
    }

    getErasingCircle(hand) {
        if (!hand || !hand.is_visible || !hand.fingers) return null;

        const index = hand.fingers.INDEX;
        const middle = hand.fingers.MIDDLE;

        if (!index || !middle || !index.is_nearly_straight_or_straight || !middle.is_nearly_straight_or_straight) {
            return null; // Not in eraser position
        }

        if (!index.landmarks || index.landmarks.length < 4 || !middle.landmarks || middle.landmarks.length < 4) {
            return null; // Not enough landmarks to calculate eraser circle
        }

        const indexTip = index.landmarks[3];
        const middleTip = middle.landmarks[3];

        // Calculate center and radius
        const center = {
            x: (indexTip.x + middleTip.x) / 2,
            y: (indexTip.y + middleTip.y) / 2
        };

        // Calculate distance for radius first (before scaling)
        const distance = Math.sqrt(
            Math.pow(indexTip.x - middleTip.x, 2) +
            Math.pow(indexTip.y - middleTip.y, 2)
        );

        // Scale to canvas coordinates
        const scaledCenter = this.scalePoint(center);
        const scaledRadius = this.scaleX(distance) / 2;

        return {center: scaledCenter, radius: scaledRadius};

    }

    createStroke(points=[], color=this.currentColor, strokeSize=this.currentStrokeSize, completedAt = null) {
        return{
            points: points,
            color: { ...color },
            strokeSize: strokeSize,
            completedAt: completedAt,
            modifiedAt: Date.now(),
            path2D: null,
            path2DGeneratedAt: 0,
            fillStyle: `hsl(${color.h}, ${color.s}%, ${color.l}%)`,
        };
    }
    
    handleDrawing() {
        if (!this.drawingPoint || !this.ctx) return;

        // Add point to stroke
        const now = Date.now();
        
        // Check if we need to start a new stroke or continue the last one
        const needNewStroke = this.strokes.length === 0 || 
                             (this.lastDrawingPoint === null &&
                              this.strokes.at(-1).completedAt &&
                              (now - this.strokes.at(-1).completedAt) > this.STROKE_CONTINUATION_TIME);

        if (needNewStroke) {
            // Start a new stroke
            this.strokes.push(this.createStroke());
        }

        // Get current stroke
        const currentStroke = this.strokes.at(-1);
        
        // Check if we should add this point
        let shouldAddPoint = true;
        
        // If there's already a point in the current stroke, check distance
        if (currentStroke.points.length > 0) {
            const lastPoint = currentStroke.points.at(-1);
            const distance = Math.hypot(
                this.drawingPoint.x - lastPoint[0],
                this.drawingPoint.y - lastPoint[1]
            );
            
            // Skip if too close to last point
            if (distance < this.MIN_POINT_DISTANCE) {
                shouldAddPoint = false;
            }
            // Skip if too far (jump detection)
            else if (distance > this.MAX_POINT_DISTANCE) {
                shouldAddPoint = false;
                
                // If it's a large jump, mark current stroke as completed and start a new one
                if (currentStroke.points.length > 0) {
                    currentStroke.completedAt = now;
                    this.strokes.push(this.createStroke());
                }
            }
        }
        
        // Add point if it passes the distance check
        if (shouldAddPoint) {
            const targetStroke = this.strokes.at(-1);
            targetStroke.points.push([this.drawingPoint.x, this.drawingPoint.y]);
            targetStroke.modifiedAt = now;
        }
        
        this.lastDrawingPoint = this.drawingPoint;
        this.lastDrawingPointTime = now;
    }
    
    handleErasing(indexFinger, middleFinger) {
        if (!this.erasingCircle || !this.ctx) return;
        if (!this.strokes || this.strokes.length === 0) return;

        const centerX = this.erasingCircle.center.x;
        const centerY = this.erasingCircle.center.y;
        const radiusSquared = this.erasingCircle.radius * this.erasingCircle.radius;

        const newStrokes = [];

        for (const stroke of this.strokes) {
            if (stroke.points.length < 2) continue;

            const segments = [];
            let currentSegment = null;
            let strokeModified = false;

            for (const point of stroke.points) {
                const dx = point[0] - centerX;
                const dy = point[1] - centerY;
                const distanceSquared = dx * dx + dy * dy;

                if (distanceSquared > radiusSquared) {
                    if (!currentSegment) {
                        currentSegment = [];
                    }
                    currentSegment.push(point);
                } else {
                    strokeModified = true;
                    if (currentSegment && currentSegment.length > 0) {
                        segments.push(currentSegment);
                        currentSegment = null;
                    }
                }
            }

            if (currentSegment && currentSegment.length > 0) {
                segments.push(currentSegment);
            }

            if (!strokeModified) {
                newStrokes.push(stroke);
            } else {
                for (const segment of segments) {
                    if (segment.length >= 2) {
                        newStrokes.push(this.createStroke(segment, stroke.color, stroke.strokeSize, stroke.completedAt));
                    }
                }
            }
        }

        this.strokes = newStrokes;
    }
    
    
    clearDrawing() {
        this.strokes = [];
        this.triggerCooldown(1000, 'clear');
        this.lastDrawingPoint = null;
        this.lastDrawingPointTime = 0;
    }
    
    handleSwipe(mode) {
        if (!this.strokes || this.strokes.length === 0) return;
        
        if (mode === 'hand') {
            // Remove the last stroke
            this.strokes.pop();
            this.triggerCooldown(250, 'swipe');
        } else if (mode === 'index') {
            // Remove the last point from the last stroke
            const lastStroke = this.strokes.at(-1);
            if (lastStroke && lastStroke.points && lastStroke.points.length > 0) {
                lastStroke.points.pop();
                // If the stroke has no points left, remove it
                if (lastStroke.points.length === 0) {
                    this.strokes.pop();
                } else {
                    lastStroke.modifiedAt = Date.now();
                }
            }
            this.triggerCooldown(250, 'swipe');
        }
    }
    
    triggerCooldown(duration, source) {
        this.cooldownEndAt = Date.now() + duration;
        this.cooldownSource = source;
    }
    
    draw() {
        if (!this.ctx || !this.isActive) return;
        
        // Clear display canvas
        DP.clearCanvas(this.ctx);
        
        // Draw all strokes
        this.drawStrokes();
        
        // Draw UI elements
        this.drawIndicators();
    }
    
    drawStrokes() {
        const ctx = this.ctx;
        
        // Draw all strokes
        for (let i = 0; i < this.strokes.length; i++) {
            const stroke = this.strokes[i];
            
            // Skip strokes with less than 2 points
            if (stroke.points.length < 2) continue;

            // Check if this is the current stroke (last stroke and still drawing)
            const isCurrentStroke = i === this.strokes.length - 1 && this.drawingHand && !stroke.completedAt;
            
            // Draw with slight transparency if current stroke
            if (isCurrentStroke) {
                ctx.save();
                ctx.globalAlpha = 0.9;
            }

            if (stroke.modifiedAt > stroke.path2DGeneratedAt) {
                // Get stroke outline using Perfect Freehand
                const outline = getStroke(stroke.points, {...this.strokeOptions, size: stroke.strokeSize || this.currentStrokeSize});
                // Generate Path2D from stroke outline
                stroke.path2D = new Path2D(getSvgPathFromStroke(outline));
                stroke.path2DGeneratedAt = Date.now();
            }

            ctx.save();
            ctx.fillStyle = stroke.fillStyle;
            ctx.fill(stroke.path2D);
            ctx.restore();

            if (isCurrentStroke) {
                ctx.restore();
            }
            
            // Draw white dots for each detected point
            if (this.SHOW_DETECTED_POINTS) {
                ctx.save();
                ctx.fillStyle = '#FFFFFF';
                for (const point of stroke.points) {
                    ctx.beginPath();
                    ctx.arc(point[0], point[1], 2, 0, Math.PI * 2);
                    ctx.fill();
                }
                ctx.restore();
            }
        }
    }
    
    drawIndicators() {
        const ctx = this.ctx;
        const now = Date.now();
        
        // Check if indicators should be visible
        if (now >= this.indicatorsVisibleUntil) {
            return; // Don't draw any indicators
        }
        
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
        
        // Hand indicator position (below size indicator)
        const handX = sizeX;
        const handY = sizeY + this.INDICATOR_SIZE + 10;
        
        // Background
        ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
        DP.roundedRect(ctx, handX, handY, this.INDICATOR_SIZE, this.INDICATOR_SIZE, 10);
        ctx.fill();
        
        // Draw hand icons
        const iconSize = this.INDICATOR_SIZE * 0.35;
        const spacing = this.INDICATOR_SIZE * 0.1;
        const leftHandX = handX + this.INDICATOR_SIZE / 2 - iconSize - spacing / 2;
        const rightHandX = handX + this.INDICATOR_SIZE / 2 + spacing / 2;
        const handIconY = handY + this.INDICATOR_SIZE / 2 - iconSize / 2;
        
        // Helper function to draw a simple hand icon
        const drawHandIcon = (x, y, size, isLeft, handSide) => {
            ctx.save();
            ctx.translate(x + size / 2, y + size / 2);
            
            // Check if hand is visible
            const hand = this.handsData?.[handSide];
            const isVisible = hand?.is_visible || false;
            
            // Determine hand state and color
            let fillStyle = 'rgba(255, 255, 255, 0.1)'; // Very faint when not visible
            let handState = 'normal';
            
            if (isVisible) {
                // Check for closed fist
                const isFist = this.isGestureActive('CLOSED_FIST', handSide);
                
                // Check if this hand is the drawing hand (in drawing mode)
                const isDrawingHand = this.drawingHand && this.drawingHand.handedness === (handSide === 'left' ? 'LEFT' : 'RIGHT');
                
                // Check if hand can erase or draw
                const erasingCircle = hand ? (hand === this.drawingHand ? this.erasingCircle : this.getErasingCircle(hand)) : null;
                const drawingPoint = hand ? (hand === this.drawingHand ? this.drawingPoint : this.getDrawingPoint(hand)) : null;
                
                if (isFist) {
                    handState = 'fist';
                    fillStyle = isDrawingHand ? DrawingStyles.colors.accent : 'rgba(255, 255, 255, 0.5)';
                } else if (erasingCircle) {
                    handState = 'erasing';
                    fillStyle = isDrawingHand ? DrawingStyles.colors.accent : 'rgba(255, 255, 255, 0.5)';
                } else if (drawingPoint) {
                    handState = 'drawing';
                    fillStyle = isDrawingHand ? DrawingStyles.colors.accent : 'rgba(255, 255, 255, 0.5)';
                } else {
                    fillStyle = 'rgba(255, 255, 255, 0.3)'; // Normal visible hand
                }
            }
            
            // Draw palm
            ctx.fillStyle = fillStyle;
            ctx.beginPath();
            ctx.ellipse(0, size * 0.1, size * 0.35, size * 0.4, 0, 0, Math.PI * 2);
            ctx.fill();
            
            // Draw fingers based on hand state
            const fingerWidth = size * 0.12;
            const fingerSpacing = size * 0.18;
            const fingerHeight = size * 0.35;
            
            if (handState === 'fist') {
                // Closed fist - show small knuckles at top
                for (let i = -1.5; i <= 1.5; i++) {
                    ctx.beginPath();
                    ctx.ellipse(
                        i * fingerSpacing,
                        -size * 0.15,
                        fingerWidth * 0.8,
                        fingerWidth * 0.8,
                        0,
                        0,
                        Math.PI * 2
                    );
                    ctx.fill();
                }
            } else if (handState === 'drawing') {
                // Only index finger up
                ctx.beginPath();
                ctx.ellipse(
                    -0.5 * fingerSpacing,
                    -size * 0.25,
                    fingerWidth,
                    fingerHeight,
                    0,
                    0,
                    Math.PI * 2
                );
                ctx.fill();
            } else if (handState === 'erasing') {
                // Index and middle fingers up
                ctx.beginPath();
                ctx.ellipse(
                    -0.5 * fingerSpacing,
                    -size * 0.25,
                    fingerWidth,
                    fingerHeight,
                    0,
                    0,
                    Math.PI * 2
                );
                ctx.fill();
                
                ctx.beginPath();
                ctx.ellipse(
                    0.5 * fingerSpacing,
                    -size * 0.25,
                    fingerWidth,
                    fingerHeight,
                    0,
                    0,
                    Math.PI * 2
                );
                ctx.fill();
            } else {
                // All fingers (normal hand)
                for (let i = -1.5; i <= 1.5; i++) {
                    ctx.beginPath();
                    ctx.ellipse(
                        i * fingerSpacing,
                        -size * 0.25,
                        fingerWidth,
                        fingerHeight,
                        0,
                        0,
                        Math.PI * 2
                    );
                    ctx.fill();
                }
            }
            
            // Draw thumb (inner side for both hands) - for all hand types
            ctx.save();
            ctx.rotate(isLeft ? Math.PI / 4 : -Math.PI / 4);
            ctx.beginPath();
            ctx.ellipse(
                isLeft ? size * 0.35 : -size * 0.35,
                0,
                fingerWidth * 0.9,
                fingerHeight * 0.8,
                0,
                0,
                Math.PI * 2
            );
            ctx.fill();
            ctx.restore();
            
            ctx.restore();
        };
        
        // Draw left hand icon
        drawHandIcon(leftHandX, handIconY, iconSize, true, 'left');
        
        // Draw right hand icon
        drawHandIcon(rightHandX, handIconY, iconSize, false, 'right');
        
        // Border
        ctx.strokeStyle = DrawingStyles.colors.accent;
        ctx.lineWidth = 2;
        DP.roundedRect(ctx, handX, handY, this.INDICATOR_SIZE, this.INDICATOR_SIZE, 10);
        ctx.stroke();
        
        ctx.restore();
    }
    
    drawPointers(skipLeftHand = false, skipRightHand = false) {
        if (!this.showPointers || !this.ctx || !this.handsData) return;
        this.drawHandPointer(this.handsData.left);
        this.drawHandPointer(this.handsData.right);
    }
    
    drawHandPointer(hand) {
        if (!hand || !hand.is_visible || !this.ctx) return false;

        const ctx = this.ctx;

        if (hand === this.activatorHand) {
            return true;  // Don't draw pointer for the activator hand
        }
        if (hand.handedness === 'LEFT' && this.isLeftHandAdjustingControl || hand.handedness === 'RIGHT' && this.isRightHandAdjustingControl) {
            return true;  // Don't draw pointer for control adjusting hands
        }

        const erasingCircle = this.erasingCircle && this.drawingHand === hand ? this.erasingCircle : this.getErasingCircle(hand);
        if (erasingCircle) {
            // Draw eraser circle
            ctx.save();
            ctx.strokeStyle = '#FFFFFF';
            ctx.lineWidth = 2;
            ctx.setLineDash([5, 5]);
            ctx.beginPath();
            ctx.arc(erasingCircle.center.x, erasingCircle.center.y, erasingCircle.radius, 0, Math.PI * 2);
            ctx.stroke();
            ctx.restore();
            return true;
        } else {
            const drawingPoint = this.drawingHand && this.drawingHand === hand ? this.drawingPoint : this.getDrawingPoint(hand);
            if (drawingPoint) {
                // Draw pointer with color fill
                ctx.save();
                ctx.fillStyle = `hsl(${this.currentColor.h}, ${this.currentColor.s}%, ${this.currentColor.l}%)`;
                ctx.beginPath();
                ctx.arc(drawingPoint.x, drawingPoint.y, this.currentStrokeSize / 2, 0, Math.PI * 2);
                ctx.fill();
                ctx.restore();
                return true;
            }
        }

        return false;

    }
}
