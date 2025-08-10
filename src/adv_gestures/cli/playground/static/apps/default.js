import { BaseApplication } from './_base.js';
import { DP, DrawingStyles } from '../drawing-primitives.js';

export class DefaultApplication extends BaseApplication {
    constructor(applicationManager) {
        super('default', applicationManager);
        this.activeApp = null;
        this.iconSize = DrawingStyles.metrics.iconSize;
        this.iconSpacing = DrawingStyles.metrics.iconSpacing;
        this.iconRects = new Map(); // Store icon rectangles for click detection
        this.showIcons = false; // Icons hidden by default
        this.showPointers = this.showIcons; // Only show pointers if icons are visible
    }

    activate() {
        super.activate();
        // Clear canvas when activating
        if (this.ctx) {
            DP.clearCanvas(this.ctx);
        }
    }
    
    setActiveApp(activeApp) {
        this.activeApp = activeApp;
        if (this.width && this.height) {
            this.updateIconRects();
        }
    }
    
    updateIconRects() {
        if (!this.width || !this.height) return;
        
        this.iconRects.clear();
        
        // Calculate center position for icons
        const appCount = this.applicationManager.applications.size - 1; // Exclude default app
        const totalWidth = appCount * this.iconSize + (appCount - 1) * this.iconSpacing;
        const startX = (this.width - totalWidth) / 2;
        const y = (this.height - this.iconSize) / 2;
        
        let x = startX;
        for (const app of this.applicationManager.applications.values()) {
            if (app === this) continue; // Skip default app
            
            this.iconRects.set(app, {
                x: x,
                y: y,
                width: this.iconSize,
                height: this.iconSize,
                app: app
            });
            
            x += this.iconSize + this.iconSpacing;
        }
    }
    
    drawIconContent(ctx, size, isActive) {
        // Default app has no icon - it's always active in the background
        // This method should never be called as default app has no icon in the icon bar
    }

    draw() {
        if (!this.ctx || !this.isActive) return;
        
        // Clear canvas
        DP.clearCanvas(this.ctx);
        
        // Draw application icons in the center only if showIcons is true
        if (this.showIcons) {
            for (const [app, rect] of this.iconRects.entries()) {
                const isActive = app === this.activeApp;
                app.drawIcon(this.ctx, rect.x, rect.y, this.iconSize, isActive);
            }
        }
    }
    
    update(handsData, gestures) {
        super.update(handsData, gestures);
        
        // Check for SNAP gesture to toggle icons visibility
        if (this.isGestureJustAdded('SNAP')) {
            this.showIcons = !this.showIcons;
            this.showPointers = this.showIcons;
        }


        // Do nothing if icons are hidden
        if (!this.showIcons) return;
        
        if (handsData.airTapData) {
            this.handleAirTaps(handsData.airTapData);
        }
    }
    
    handleAirTaps(airTapData) {
        for (const [handedness, tapData] of Object.entries(airTapData)) {
            // Skip if already handled
            if (tapData.alreadyHandled) continue;
            
            // Check if tap is on any icon
            const tapPoint = this.scalePoint(tapData.tapPosition);
            if (!tapPoint) continue;
            
            for (const [app, rect] of this.iconRects.entries()) {
                if (tapPoint.x >= rect.x && tapPoint.x <= rect.x + rect.width &&
                    tapPoint.y >= rect.y && tapPoint.y <= rect.y + rect.height) {
                    
                    // Found clicked icon - request app switch through manager
                    if (this.applicationManager) {
                        if (app === this.activeApp) {
                            // Clicking active app switches back to default
                            this.applicationManager.switchToApp('default');
                        } else {
                            // Switch to clicked app
                            this.applicationManager.switchToApp(app.name);
                        }
                    }
                    
                    // Mark as handled
                    tapData.alreadyHandled = true;
                    break;
                }
            }
        }
    }
    
    resize(width, height) {
        super.resize(width, height);
        this.updateIconRects();
    }
}
