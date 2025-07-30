import { BaseApplication } from './_base.js';
import { DP } from '../drawing-primitives.js';

export class DefaultApplication extends BaseApplication {
    constructor() {
        super('default');
    }

    activate() {
        super.activate();
        // Clear canvas when activating
        if (this.ctx) {
            DP.clearCanvas(this.ctx);
        }
    }
    
    drawIconContent(ctx, size, isActive) {
        // Default app has no icon - it's always active in the background
        // This method should never be called as default app has no icon in the icon bar
    }

    draw() {
        if (!this.ctx || !this.isActive) return;
        
        // Clear canvas - default app shows nothing
        DP.clearCanvas(this.ctx);
    }
}
