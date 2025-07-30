import { DP } from '../drawing-primitives.js';

export class BaseApplication {
    constructor(name) {
        this.name = name;
        this.canvas = null;
        this.ctx = null;
        this.isActive = false;
        this.width = 0;
        this.height = 0;
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
        // Override in subclasses if needed
        // For now, does nothing as specified
    }

    draw() {
        // Override in subclasses
        // Clear canvas by default
        if (this.ctx) {
            DP.clearCanvas(this.ctx);
        }
    }
}