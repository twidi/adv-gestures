import { DP, DrawingStyles } from './drawing-primitives.js';
import { DefaultApplication } from './apps/default.js';
import { DebugApplication } from './apps/debug.js';

export class ApplicationManager {
    constructor(canvasContainer) {
        this.applications = new Map();
        this.activeApp = null;
        this.defaultApp = null;
        this.iconSize = DrawingStyles.metrics.iconSize;
        this.iconSpacing = DrawingStyles.metrics.iconSpacing;
        this.canvasContainer = canvasContainer;
        
        // Get existing icon canvas from HTML
        this.iconCanvas = document.getElementById('app-icons');
        this.iconCtx = this.iconCanvas.getContext('2d');
        
        // Add click handler for icon selection
        this.iconCanvas.addEventListener('click', (e) => this.handleIconClick(e));
        
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
        // Update only the active app
        if (this.activeApp) {
            this.activeApp.update(handsData);
        }
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
        const rect = this.iconCanvas.getBoundingClientRect();
        const x = event.clientX - rect.left;
        const y = event.clientY - rect.top;
        
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
                break;
            }
            
            currentY += this.iconSize + this.iconSpacing;
        }
    }

    getActiveApp() {
        return this.activeApp;
    }
}
