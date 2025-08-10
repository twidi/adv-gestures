import { BaseApplication } from './_base.js';
import { DP, DrawingStyles } from '../drawing-primitives.js';

// Colors for drawing fingers (matching Python's FINGER_COLORS)
const FINGER_COLORS = [
    '#0000FF',  // Blue - THUMB
    '#00FF00',  // Green - INDEX
    '#FFFF00',  // Yellow - MIDDLE
    '#FF00FF',  // Magenta - RING
    '#00FFFF',  // Cyan - PINKY
];

export class DebugApplication extends BaseApplication {
    constructor(applicationManager) {
        super('debug', applicationManager);
        // FontAwesome bug icon SVG path (regular style)
        this.iconSvgPath = "M320 64C267 64 224 107 224 160L224 163.6C224 179.3 236.7 192 252.4 192L387.5 192C403.2 192 415.9 179.3 415.9 163.6L415.9 160C415.9 107 372.9 64 319.9 64zM432 344L432 416C432 469.6 394.3 514.4 344 525.4L344 360C344 346.7 333.3 336 320 336C306.7 336 296 346.7 296 360L296 525.4C245.7 514.4 208 469.6 208 416L208 344C208 313.1 233.1 288 264 288L376 288C406.9 288 432 313.1 432 344zM179.8 282.9C170.3 296 163.8 311.3 161.2 328L56 328C42.7 328 32 338.7 32 352C32 365.3 42.7 376 56 376L160 376L160 416C160 422.5 160.4 428.9 161.1 435.1L73.6 500.8C63 508.8 60.8 523.8 68.8 534.4C76.8 545 91.8 547.2 102.4 539.2L175.4 484.5C201 538.6 256.1 576 320 576C383.9 576 439 538.6 464.6 484.5L537.6 539.2C548.2 547.2 563.2 545 571.2 534.4C579.2 523.8 577 508.8 566.4 500.8L478.9 435.1C479.6 428.8 480 422.4 480 416L480 376L584 376C597.3 376 608 365.3 608 352C608 338.7 597.3 328 584 328L478.8 328C476.2 311.3 469.7 296 460.2 282.9L566.4 203.2C577 195.2 579.2 180.2 571.2 169.6C563.2 159 548.2 156.8 537.6 164.8L422.6 251C408.6 243.9 392.7 240 376 240L264 240C247.2 240 231.4 244 217.4 251L102.4 164.8C91.8 156.8 76.8 159 68.8 169.6C60.8 180.2 63 195.2 73.6 203.2L179.8 282.9z";
    }

    update(handsData, gestures) {
        super.update(handsData, gestures);
        
        // Check for DOUBLE_SNAP gesture to exit the app
        if (this.isGestureJustAdded('DOUBLE_SNAP')) {
            this.exit();
            return;
        }
    }

    draw() {
        if (!this.ctx || !this.isActive) return;
        
        // Clear canvas
        DP.clearCanvas(this.ctx);
        
        if (!this.handsData || !this.scale) return;
        
        // Draw hand-specific elements
        this.drawHand(this.handsData.left, 'LEFT');
        this.drawHand(this.handsData.right, 'RIGHT');
        
        // Draw frame box if available
        if (this.handsData.frame_box) {
            this.ctx.save();
            this.ctx.strokeStyle = '#0000AA'; // Dark blue
            this.ctx.lineWidth = 3;
            this.ctx.setLineDash([5, 5]);
            const { top_left, bottom_right } = this.handsData.frame_box;
            const scaledTopLeft = this.scalePoint(top_left);
            const scaledBottomRight = this.scalePoint(bottom_right);
            const displayX = scaledTopLeft.x;
            const displayY = scaledTopLeft.y;
            const displayWidth = scaledBottomRight.x - scaledTopLeft.x;
            const displayHeight = scaledBottomRight.y - scaledTopLeft.y;
            DP.roundedRect(this.ctx, displayX, displayY, displayWidth, displayHeight, DrawingStyles.metrics.borderRadius);
            this.ctx.stroke();
            
            // Draw "FRAME" label in dark blue
            DP.drawText(this.ctx, 'FRAME', displayX + 5, displayY - 5, '14px', '#0000AA');
            
            this.ctx.restore();
        }
        
        // Draw debug info panel in top right corner
        DP.drawInfoPanel(this.ctx, this.width - 220, 10, 210, 120);
        
        // Draw debug title
        DP.drawText(this.ctx, 'DEBUG INFO', this.width - 210, 25, 'bold 12px', DrawingStyles.colors.textTitle);
        
        const leftGestures = this.handsData.left?.gestures ? Object.keys(this.handsData.left.gestures).filter(g => this.handsData.left.gestures[g] > 0) : [];
        const rightGestures = this.handsData.right?.gestures ? Object.keys(this.handsData.right.gestures).filter(g => this.handsData.right.gestures[g] > 0) : [];
        const bothGestures = this.handsData.gestures ? Object.keys(this.handsData.gestures).filter(g => this.handsData.gestures[g] > 0) : [];
        
        // Draw debug information
        DP.drawText(this.ctx, `Left: ${leftGestures.join(', ') || 'none'}`, this.width - 210, 45, '12px');
        DP.drawText(this.ctx, `Right: ${rightGestures.join(', ') || 'none'}`, this.width - 210, 65, '12px');
        DP.drawText(this.ctx, `Both: ${bothGestures.join(', ') || 'none'}`, this.width - 210, 85, '12px');
        DP.drawText(this.ctx, `Hands: ${(this.handsData.left ? 1 : 0) + (this.handsData.right ? 1 : 0)}`, this.width - 210, 105, '12px');
    }
    
    drawHand(hand, label) {
        if (!hand) return;
        
        // Draw bounding box
        if (hand.bounding_box) {
            this.ctx.save();
            this.ctx.strokeStyle = DrawingStyles.colors.accent;
            this.ctx.lineWidth = 2;
            this.ctx.setLineDash([5, 5]);
            
            const { top_left, bottom_right } = hand.bounding_box;
            const scaledTopLeft = this.scalePoint(top_left);
            const scaledBottomRight = this.scalePoint(bottom_right);
            const displayX = scaledTopLeft.x;
            const displayY = scaledTopLeft.y;
            const displayWidth = scaledBottomRight.x - scaledTopLeft.x;
            const displayHeight = scaledBottomRight.y - scaledTopLeft.y;
            DP.roundedRect(this.ctx, displayX, displayY, displayWidth, displayHeight, DrawingStyles.metrics.borderRadius);
            this.ctx.stroke();
            
            // Draw label in same color as bounding box
            DP.drawText(this.ctx, label, displayX + 5, displayY - 5, '14px', DrawingStyles.colors.accent);
            this.ctx.restore();
        }
        
        // Draw pinch box if available
        if (hand.pinch_box) {
            this.ctx.save();
            // Red for PINCH_TOUCH, yellow for regular PINCH
            const hasPinchTouch = hand.gestures && 'PINCH_TOUCH' in hand.gestures && hand.gestures.PINCH_TOUCH > 0;
            this.ctx.strokeStyle = hasPinchTouch ? '#FF0000' : '#FFFF00';
            this.ctx.lineWidth = 2;
            this.ctx.setLineDash([5, 5]);
            
            const { top_left, bottom_right } = hand.pinch_box;
            const scaledTopLeft = this.scalePoint(top_left);
            const scaledBottomRight = this.scalePoint(bottom_right);
            const displayX = scaledTopLeft.x;
            const displayY = scaledTopLeft.y;
            const displayWidth = scaledBottomRight.x - scaledTopLeft.x;
            const displayHeight = scaledBottomRight.y - scaledTopLeft.y;
            DP.roundedRect(this.ctx, displayX, displayY, displayWidth, displayHeight, DrawingStyles.metrics.borderRadius);
            this.ctx.stroke();
            
            // Draw "PINCH" label in same color as box
            const labelColor = hasPinchTouch ? '#FF0000' : '#FFFF00';
            DP.drawText(this.ctx, 'PINCH', displayX + 5, displayY - 5, '14px', labelColor);
            
            this.ctx.restore();
        }
        
        // Draw palm center if available
        if (hand.palm && hand.palm.centroid) {
            const palmCenter = this.scalePoint(hand.palm.centroid);
            const palmColor = hand.is_facing_camera ? '#00FF00' : '#FF0000';
            this.ctx.fillStyle = palmColor;
            this.ctx.beginPath();
            this.ctx.arc(palmCenter.x, palmCenter.y, 5, 0, 2 * Math.PI);
            this.ctx.fill();
        }
        
        // Draw main direction arrow
        if (hand.main_direction && hand.wrist_landmark) {
            const wrist = this.scalePoint(hand.wrist_landmark);
            const direction = hand.main_direction;
            
            // Calculate arrow end point
            const arrowLength = 70;
            const endX = wrist.x + direction.x * arrowLength;
            const endY = wrist.y + direction.y * arrowLength;
            
            // Draw arrow in cyan
            this.ctx.strokeStyle = '#00FFFF';
            this.ctx.lineWidth = 3;
            this.ctx.setLineDash([]);
            
            // Draw arrow line
            this.ctx.beginPath();
            this.ctx.moveTo(wrist.x, wrist.y);
            this.ctx.lineTo(endX, endY);
            this.ctx.stroke();
            
            // Draw arrowhead
            const headLength = 15;
            const angle = Math.atan2(direction.y, direction.x);
            this.ctx.beginPath();
            this.ctx.moveTo(endX, endY);
            this.ctx.lineTo(endX - headLength * Math.cos(angle - Math.PI/6), endY - headLength * Math.sin(angle - Math.PI/6));
            this.ctx.moveTo(endX, endY);
            this.ctx.lineTo(endX - headLength * Math.cos(angle + Math.PI/6), endY - headLength * Math.sin(angle + Math.PI/6));
            this.ctx.stroke();
        }
        
        // Draw landmarks for all fingers
        if (hand.fingers) {
            const fingersArray = Object.values(hand.fingers);
            
            fingersArray.forEach((finger, fingerIndex) => {
                if (!finger.landmarks) return;
                
                const color = FINGER_COLORS[fingerIndex % FINGER_COLORS.length];
                
                // Draw all landmarks
                finger.landmarks.forEach(landmark => {
                    const scaled = this.scalePoint(landmark);
                    this.ctx.fillStyle = color;
                    this.ctx.beginPath();
                    this.ctx.arc(scaled.x, scaled.y, 3, 0, 2 * Math.PI);
                    this.ctx.fill();
                });
                
                // Draw lines for straight or nearly straight fingers
                if ((finger.is_straight || finger.is_nearly_straight) && finger.start_point && finger.end_point) {
                    const start = this.scalePoint(finger.start_point);
                    const end = this.scalePoint(finger.end_point);
                    
                    this.ctx.strokeStyle = color;
                    this.ctx.lineWidth = finger.is_straight ? 3 : 2;
                    
                    if (finger.is_straight) {
                        // Solid line for straight fingers
                        this.ctx.setLineDash([]);
                    } else {
                        // Dashed line for nearly straight fingers
                        this.ctx.setLineDash([10, 5]);
                    }
                    
                    this.ctx.beginPath();
                    this.ctx.moveTo(start.x, start.y);
                    this.ctx.lineTo(end.x, end.y);
                    this.ctx.stroke();
                }
                
                // Draw connections to adjacent touching fingers
                if (finger.touching_adjacent_fingers && finger.touching_adjacent_fingers.length > 0 && finger.end_point) {
                    const myTip = this.scalePoint(finger.end_point);

                    finger.touching_adjacent_fingers.forEach(touchingFingerName => {
                        // Find the touching finger
                        const touchingFinger = fingersArray.find(f => f.type === touchingFingerName);
                        if (touchingFinger && touchingFinger.end_point) {
                            const otherTip = this.scalePoint(touchingFinger.end_point);

                            // Draw solid white line between touching fingers
                            this.ctx.strokeStyle = '#FFFFFF';
                            this.ctx.lineWidth = 1;
                            this.ctx.setLineDash([]);

                            this.ctx.beginPath();
                            this.ctx.moveTo(myTip.x, myTip.y);
                            this.ctx.lineTo(otherTip.x, otherTip.y);
                            this.ctx.stroke();
                        }
                    });
                }
                
                // Draw red circle if this finger touches the thumb
                if (finger.type !== 'THUMB' && finger.tip_on_thumb) {
                    // Find the thumb finger
                    const thumb = fingersArray.find(f => f.type === 'THUMB');
                    if (thumb && thumb.end_point) {
                        // Calculate middle point between both tips
                        const fingerTip = this.scalePoint(finger.end_point);
                        const thumbTip = this.scalePoint(thumb.end_point);
                        
                        const middleX = (fingerTip.x + thumbTip.x) / 2;
                        const middleY = (fingerTip.y + thumbTip.y) / 2;
                        
                        // Draw small red filled circle
                        this.ctx.fillStyle = '#FF0000';
                        this.ctx.beginPath();
                        this.ctx.arc(middleX, middleY, 8, 0, 2 * Math.PI);
                        this.ctx.fill();
                    }
                }
            });
        }
        
        // Reset line dash
        this.ctx.setLineDash([]);
    }
}
