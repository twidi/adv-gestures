import { DP, DrawingStyles } from './drawing-primitives.js';
import { DefaultApplication } from './apps/default.js';
import { DebugApplication } from './apps/debug.js';

export class ApplicationManager {
    constructor(canvasContainer, handledAirTaps) {
        this.applications = new Map();
        this.activeApp = null;
        this.defaultApp = null;
        this.iconSize = DrawingStyles.metrics.iconSize;
        this.iconSpacing = DrawingStyles.metrics.iconSpacing;
        this.canvasContainer = canvasContainer;
        this.width = 0;
        this.height = 0;
        this.handledAirTaps = handledAirTaps;

        this.streamSize = null; // Will be set when stream info is available

        // Get existing icon canvas from HTML
        this.iconCanvas = document.getElementById('app-icons');
        this.iconCtx = this.iconCanvas.getContext('2d');
        
        // Add click and air-tap handlers for icon selection
        this.iconCanvas.addEventListener('click', (e) => this.handleIconClick(e));
        this.iconCanvas.addEventListener('air-tap', (e) => this.handleIconClick(e));

        // Register all applications
        this.registerApplications();
    }
    
    registerApplications() {
        // Create and register default application (always first, no icon)
        const defaultApp = new DefaultApplication();
        this.registerApplication(defaultApp, true);
        
        // Create and register other applications

        const debugApp = new DebugApplication();
        this.registerApplication(debugApp);
    }

    registerApplication(app, isDefault = false) {
        this.applications.set(app.name, app);
        
        if (isDefault) {
            this.defaultApp = app;
            this.activeApp = app;
        }
    }

    createCanvases(width, height) {
        // Set manager dimensions
        this.width = width;
        this.height = height;

        // Create canvases for all applications
        for (const app of this.applications.values()) {
            app.createCanvas(width, height);
        }
        
        // Update icon canvas size
        const iconCanvasWidth = this.iconSize + this.iconSpacing * 2;
        const iconCanvasHeight = height;
        this.iconCanvas.width = iconCanvasWidth;
        this.iconCanvas.height = iconCanvasHeight;
        
        // Activate the default app
        if (this.activeApp) {
            this.activeApp.activate();
        }
        
        this.drawIcons();
    }

    resize(width, height) {
        // Update manager dimensions
        this.width = width;
        this.height = height;

        // Update icon canvas size
        const iconCanvasWidth = this.iconSize + this.iconSpacing * 2;
        const iconCanvasHeight = height;
        this.iconCanvas.width = iconCanvasWidth;
        this.iconCanvas.height = iconCanvasHeight;
        
        // Resize all app canvases
        for (const app of this.applications.values()) {
            app.resize(width, height);
        }
        
        this.drawIcons();
    }

    switchToApp(appName) {
        const newApp = this.applications.get(appName);
        if (!newApp || newApp === this.activeApp) return;
        
        // Deactivate current app
        if (this.activeApp) {
            this.activeApp.deactivate();
        }
        
        // Activate new app
        this.activeApp = newApp;
        this.activeApp.activate();
        
        this.drawIcons();
    }

    update(handsData) {
        if (!this.streamSize || this.streamSize.width !== handsData.stream_info.width || this.streamSize.height !== handsData.stream_info.height) {
            // Update stream size if it has changed
            this.streamSize = {width: handsData.stream_info.width, height: handsData.stream_info.height};
            for (const app of this.applications.values()) {
                app.setStreamSize(this.streamSize);
            }
        }
        
        // Check for air taps and simulate clicks if not already handled
        if (handsData.airTapData) {
            this.handleAirTaps(handsData.airTapData);
        }
        
        // Update only the active app
        if (this.activeApp) {
            this.activeApp.update(handsData);
        }

        // Mark all air taps as handled (we receive them for many frames so we only handle them once)
        this.markAirTapsAsHandled(handsData.airTapData);
    }

    draw() {
        // Draw only the active app
        if (this.activeApp) {
            this.activeApp.draw();
            // Draw cursors after the app has drawn its content
            this.activeApp.drawCursors();
        }
    }

    drawIcons() {
        if (!this.iconCtx) return;
        
        const ctx = this.iconCtx;
        DP.clearCanvas(ctx);
        
        let y = this.iconSpacing;
        
        // Draw icons for all apps except default
        for (const app of this.applications.values()) {
            if (app === this.defaultApp) continue;
            
            const isActive = app === this.activeApp;
            app.drawIcon(ctx, this.iconSpacing, y, this.iconSize, isActive);
            
            y += this.iconSize + this.iconSpacing;
        }
    }

    handleIconClick(event) {
        const isAirTap = event.type === 'air-tap';
        const x = isAirTap ? event.detail.tapPosition.x : event.offsetX;
        const y = isAirTap ? event.detail.tapPosition.y : event.offsetY;

        // Check if click is within icon area
        if (x < this.iconSpacing || x > this.iconSpacing + this.iconSize) return;
        
        let currentY = this.iconSpacing;
        
        for (const app of this.applications.values()) {
            if (app === this.defaultApp) continue;
            
            if (y >= currentY && y < currentY + this.iconSize) {
                if (app === this.activeApp) {
                    // Clicking active app switches back to default
                    this.switchToApp(this.defaultApp.name);
                } else {
                    // Switch to clicked app
                    this.switchToApp(app.name);
                }
                event.stopImmediatePropagation();
                event.preventDefault();
                break;
            }
            
            currentY += this.iconSize + this.iconSpacing;
        }
    }

    getActiveApp() {
        return this.activeApp;
    }
    
    handleAirTaps(airTapData) {
        for (const [handedness, tapData] of Object.entries(airTapData)) {
            // Skip if already handled
            if (tapData.alreadyHandled) continue;

            // Scale the tap position to icon canvas coordinates
            const scaledPosition = this.scalePointToIconCanvas(tapData.tapPosition);
            
            // Create and dispatch custom air-tap event
            const event = new CustomEvent('air-tap', {
                bubbles: true,
                cancelable: true,
                detail: {
                    tapPosition: scaledPosition,
                    tapId: tapData.tapId
                }
            });
            
            // Dispatch the event on the icon canvas
            this.iconCanvas.dispatchEvent(event);
            
            // If the event was handled (propagation stopped), mark the air tap as handled right away
            if (event.defaultPrevented) {
                // Mark this tap as handled
                this.handledAirTaps.add(tapData.tapId);
                tapData.alreadyHandled = true;
            }
        }
    }
    
    scalePointToIconCanvas(point) {
        // Scale from normalized coordinates (0-1) to icon canvas pixel coordinates
        if (!this.streamSize || !point) return point;

        // Compute scale
        const scaleX = this.width / this.streamSize.width;
        const scaleY = this.height / this.streamSize.height;

        // First scale to stream size
        const streamX = point.x * scaleX;
        const streamY = point.y * scaleY;
        
        // Icon canvas is positioned at the left edge, so we only need to check if x is within bounds
        // No additional scaling needed since icon canvas uses absolute positioning
        return {
            x: streamX,
            y: streamY
        };
    }

    markAirTapsAsHandled(airTapData) {
        if (!airTapData) return;

        for (const [handedness, tapData] of Object.entries(airTapData)) {
            if (tapData && tapData.tapId) {
                this.handledAirTaps.add(tapData.tapId);
                tapData.alreadyHandled = true; // Mark as handled to avoid reprocessing
            }
        }
    }

}
