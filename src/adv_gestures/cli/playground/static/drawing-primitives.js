// Drawing style constants
export const DrawingStyles = {
    // Colors from CSS variables
    colors: {
        accent: '#00ffff',
        textPrimary: '#001616',
        textSecondary: '#003333',
        textTitle: '#000808',
        bgOverlay: 'rgba(110,234,234,0.5)',
        bgIcon: 'rgba(0, 0, 0, 0.3)',
        border: '#00ffff',
    },
    
    // Common measurements
    metrics: {
        borderRadius: 10,
        borderWidth: 1,
        borderWidthActive: 3,
        iconSize: 48,
        iconSpacing: 8,
        glowRadius: 10,
        glowOpacity: 0.3
    }
};

// Drawing primitives
export class DrawingPrimitives {
    /**
     * Draw an icon container (like the control buttons)
     * @param {CanvasRenderingContext2D} ctx - Canvas context
     * @param {number} x - X position
     * @param {number} y - Y position
     * @param {number} size - Size of the container
     * @param {boolean} isActive - Whether the container is active
     * @param {boolean} isHovered - Whether the container is hovered
     * @param {number} borderWidth - Border width (optional)
     */
    static drawIconContainer(ctx, x, y, size, isActive = false, isHovered = false, borderWidth = null) {
        const styles = DrawingStyles;
        const actualBorderWidth = borderWidth || (isActive ? styles.metrics.borderWidthActive : styles.metrics.borderWidth);
        
        ctx.save();
        
        // Background with rounded corners
        ctx.fillStyle = styles.colors.bgIcon;
        this.roundedRect(ctx, x, y, size, size, styles.metrics.borderRadius);
        ctx.fill();
        
        // Add subtle inner glow for depth
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.05)';
        ctx.lineWidth = 1;
        this.roundedRect(ctx, x + 1, y + 1, size - 2, size - 2, styles.metrics.borderRadius - 1);
        ctx.stroke();
        
        // Border
        ctx.strokeStyle = styles.colors.accent;
        ctx.lineWidth = actualBorderWidth;
        this.roundedRect(ctx, x, y, size, size, styles.metrics.borderRadius);
        ctx.stroke();
        
        // Glow effect when active
        if (isActive) {
            ctx.shadowColor = styles.colors.accent;
            ctx.shadowBlur = styles.metrics.glowRadius;
            ctx.globalAlpha = styles.metrics.glowOpacity;
            this.roundedRect(ctx, x, y, size, size, styles.metrics.borderRadius);
            ctx.stroke();
            ctx.globalAlpha = 1;
            ctx.shadowBlur = 0;
        }
        
        ctx.restore();
    }
    
    /**
     * Draw a rounded rectangle
     * @param {CanvasRenderingContext2D} ctx - Canvas context
     * @param {number} x - X position
     * @param {number} y - Y position
     * @param {number} width - Width
     * @param {number} height - Height
     * @param {number} radius - Corner radius
     */
    static roundedRect(ctx, x, y, width, height, radius) {
        ctx.save();
        ctx.beginPath();
        ctx.moveTo(x + radius, y);
        ctx.lineTo(x + width - radius, y);
        ctx.quadraticCurveTo(x + width, y, x + width, y + radius);
        ctx.lineTo(x + width, y + height - radius);
        ctx.quadraticCurveTo(x + width, y + height, x + width - radius, y + height);
        ctx.lineTo(x + radius, y + height);
        ctx.quadraticCurveTo(x, y + height, x, y + height - radius);
        ctx.lineTo(x, y + radius);
        ctx.quadraticCurveTo(x, y, x + radius, y);
        ctx.closePath();
        ctx.restore();
    }
    
    /**
     * Draw text with proper styling
     * @param {CanvasRenderingContext2D} ctx - Canvas context
     * @param {string} text - Text to draw
     * @param {number} x - X position
     * @param {number} y - Y position
     * @param {string} size - Font size (e.g., '14px', 'bold 16px')
     * @param {string} color - Text color
     * @param {string} align - Text alignment
     * @param {string} baseline - Text baseline
     */
    static drawText(ctx, text, x, y, size = '14px', color = null, align = 'left', baseline = 'alphabetic') {
        ctx.save();
        ctx.font = `${size} monospace`;
        ctx.fillStyle = color || DrawingStyles.colors.textPrimary;
        ctx.textAlign = align;
        ctx.textBaseline = baseline;
        ctx.fillText(text, x, y);
        ctx.restore();
    }
    
    /**
     * Draw a label (small text) like the ON/OFF states
     * @param {CanvasRenderingContext2D} ctx - Canvas context
     * @param {string} text - Text to draw
     * @param {number} x - X position
     * @param {number} y - Y position
     * @param {boolean} isActive - Whether the label is active
     */
    static drawLabel(ctx, text, x, y, isActive = false) {
        const color = isActive ? DrawingStyles.colors.accent : DrawingStyles.colors.textSecondary;
        this.drawText(ctx, text, x, y, 'bold 9px', color, 'center', 'middle');
    }
    
    /**
     * Draw an info panel background
     * @param {CanvasRenderingContext2D} ctx - Canvas context
     * @param {number} x - X position
     * @param {number} y - Y position
     * @param {number} width - Width
     * @param {number} height - Height
     */
    static drawInfoPanel(ctx, x, y, width, height) {
        ctx.save();
        ctx.fillStyle = DrawingStyles.colors.bgOverlay;
        this.roundedRect(ctx, x, y, width, height, DrawingStyles.metrics.borderRadius);
        ctx.fill();
        
        ctx.strokeStyle = DrawingStyles.colors.border;
        ctx.lineWidth = DrawingStyles.metrics.borderWidth;
        this.roundedRect(ctx, x, y, width, height, DrawingStyles.metrics.borderRadius);
        ctx.stroke();
        ctx.restore();
    }
    
    /**
     * Draw a ripple effect animation
     * @param {CanvasRenderingContext2D} ctx - Canvas context
     * @param {number} x - X position
     * @param {number} y - Y position
     * @param {number} progress - Animation progress (0-1)
     * @param {Object} options - Optional parameters
     * @param {number} options.rippleCount - Number of ripple circles (default: 3)
     * @param {number} options.maxRadius - Maximum radius (default: 50)
     * @param {string} options.color - Color (default: accent color)
     * @param {number} options.lineWidth - Line width (default: 3)
     */
    static drawRippleEffect(ctx, x, y, progress, options = {}) {
        
        // Default options
        const rippleCount = options.rippleCount || 3;
        const maxRadius = options.maxRadius || 50;
        const color = options.color || DrawingStyles.colors.accent;
        const lineWidth = options.lineWidth || 3;
        
        for (let i = 0; i < rippleCount; i++) {
            const rippleDelay = i * 0.15; // Stagger the ripples
            const rippleProgress = Math.max(0, Math.min((progress - rippleDelay) / (1 - rippleDelay), 1));
            
            if (rippleProgress > 0 && rippleProgress < 1) {
                const radius = rippleProgress * maxRadius;
                const opacity = (1 - rippleProgress) * 0.6;
                
                ctx.save();
                ctx.beginPath();
                ctx.arc(x, y, radius, 0, Math.PI * 2);
                ctx.strokeStyle = color;
                ctx.lineWidth = lineWidth;
                ctx.globalAlpha = opacity;
                ctx.stroke();
                ctx.restore();
            }
        }
    }
    
    /**
     * Draw a progress arc (circular progress indicator)
     * @param {CanvasRenderingContext2D} ctx - Canvas context
     * @param {number} x - Center X position
     * @param {number} y - Center Y position
     * @param {number} radius - Radius of the arc
     * @param {number} progress - Progress value (0-1)
     * @param {Object} options - Optional parameters
     * @param {string} options.color - Color (default: accent color)
     * @param {number} options.lineWidth - Line width (default: 3)
     * @param {number} options.startAngle - Starting angle in radians (default: -π/2 for 12 o'clock)
     * @param {boolean} options.clockwise - Direction of progress (default: true)
     * @param {number} options.opacity - Opacity (default: 0.8)
     */
    static drawProgressArc(ctx, x, y, radius, progress, options = {}) {
        if (progress <= 0) return;
        
        const color = options.color || DrawingStyles.colors.accent;
        const lineWidth = options.lineWidth || 3;
        const startAngle = options.startAngle !== undefined ? options.startAngle : -Math.PI / 2;
        const clockwise = options.clockwise !== undefined ? options.clockwise : true;
        const opacity = options.opacity !== undefined ? options.opacity : 0.8;
        
        const angleRange = Math.PI * 2 * progress;
        const endAngle = clockwise ? startAngle + angleRange : startAngle - angleRange;
        
        ctx.save();
        ctx.strokeStyle = color;
        ctx.lineWidth = lineWidth;
        ctx.globalAlpha = opacity;
        ctx.beginPath();
        ctx.arc(x, y, radius, startAngle, endAngle, !clockwise);
        ctx.stroke();
        ctx.restore();
    }
    
    /**
     * Clear the entire canvas
     * @param {CanvasRenderingContext2D} ctx - Canvas context
     */
    static clearCanvas(ctx) {
        ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
    }
}

// Export alias for shorter usage
export { DrawingPrimitives as DP };
