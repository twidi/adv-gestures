import { BaseApplication } from './_base.js';
import { DP, DrawingStyles } from '../drawing-primitives.js';

export class DebugApplication extends BaseApplication {
    constructor() {
        super('debug');
        this.streamInfo = null;
        this.handsData = null;
    }
    
    drawIconContent(ctx, size, isActive) {
        // Draw a D for Debug
        const color = DrawingStyles.colors.accent;
        DP.drawText(ctx, 'D', size/2, size/2, `bold ${size * 0.6}px`, color, 'center', 'middle');
    }

    setStreamInfo(streamInfo) {
        this.streamInfo = streamInfo;
    }

    update(handsData) {
        this.handsData = handsData;
        
        // Store stream info if available
        if (handsData.stream_info) {
            this.streamInfo = handsData.stream_info;
        }
    }

    draw() {
        if (!this.ctx || !this.isActive) return;
        
        // Clear canvas
        DP.clearCanvas(this.ctx);
        
        if (!this.streamInfo || !this.handsData) return;
        
        // Calculate scale factors - canvas size matches video element size
        const scaleX = this.width / this.streamInfo.width;
        const scaleY = this.height / this.streamInfo.height;
        
        // Draw bounding boxes for each hand
        this.ctx.strokeStyle = DrawingStyles.colors.accent;
        this.ctx.lineWidth = 2;
        this.ctx.setLineDash([5, 5]);
        
        // Draw left hand bounding box
        if (this.handsData.left && this.handsData.left.bounding_box) {
            const { top_left, bottom_right } = this.handsData.left.bounding_box;
            const displayX = top_left.x * scaleX;
            const displayY = top_left.y * scaleY;
            const displayWidth = (bottom_right.x - top_left.x) * scaleX;
            const displayHeight = (bottom_right.y - top_left.y) * scaleY;
            DP.roundedRect(this.ctx, displayX, displayY, displayWidth, displayHeight, DrawingStyles.metrics.borderRadius);
            this.ctx.stroke();
            
            // Draw label
            DP.drawText(this.ctx, 'LEFT', displayX + 5, displayY - 5, '14px');
        }
        
        // Draw right hand bounding box
        if (this.handsData.right && this.handsData.right.bounding_box) {
            const { top_left, bottom_right } = this.handsData.right.bounding_box;
            const displayX = top_left.x * scaleX;
            const displayY = top_left.y * scaleY;
            const displayWidth = (bottom_right.x - top_left.x) * scaleX;
            const displayHeight = (bottom_right.y - top_left.y) * scaleY;
            DP.roundedRect(this.ctx, displayX, displayY, displayWidth, displayHeight, DrawingStyles.metrics.borderRadius);
            this.ctx.stroke();
            
            // Draw label
            DP.drawText(this.ctx, 'RIGHT', displayX + 5, displayY - 5, '14px');
        }
        
        this.ctx.setLineDash([]);
        
        // Draw debug info panel in top right corner
        DP.drawInfoPanel(this.ctx, this.width - 220, 10, 210, 100);
        
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
}
