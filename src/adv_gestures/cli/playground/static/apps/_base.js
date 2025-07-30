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
            }
        }
    }
}
