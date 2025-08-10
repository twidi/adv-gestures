import { DP, DrawingStyles } from './drawing-primitives.js';
import { DefaultApplication } from './apps/default.js';
import { DebugApplication } from './apps/debug.js';
import { DrawingApplication } from './apps/drawing.js';

export class ApplicationManager {
    constructor(canvasContainer, handledAirTaps) {
        this.applications = new Map();
        this.activeApp = null;
        this.defaultApp = null;
        this.canvasContainer = canvasContainer;
        this.width = 0;
        this.height = 0;
        this.handledAirTaps = handledAirTaps;

        this.streamSize = null; // Will be set when stream info is available

        // Register all applications
        this.registerApplications();
    }
    
    registerApplications() {
        // Create and register default application (always first, no icon)
        const defaultApp = new DefaultApplication(this);
        this.applications.set(defaultApp.name, defaultApp);
        this.defaultApp = defaultApp;
        this.activeApp = defaultApp;

        // Create and register other applications

        const debugApp = new DebugApplication(this);
        this.applications.set(debugApp.name, debugApp);

        const drawingApp = new DrawingApplication(this);
        this.applications.set(drawingApp.name, drawingApp);
    }

    createCanvases(width, height) {
        // Set manager dimensions
        this.width = width;
        this.height = height;

        // Create canvases for all applications
        for (const app of this.applications.values()) {
            app.createCanvas(width, height);
        }
        
        // Activate the default app
        if (this.activeApp) {
            this.activeApp.activate();
        }
        
        // Set the default app's active app reference
        this.defaultApp.setActiveApp(this.activeApp);
    }

    resize(width, height) {
        // Update manager dimensions
        this.width = width;
        this.height = height;
        
        // Resize all app canvases
        for (const app of this.applications.values()) {
            app.resize(width, height);
        }
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
        
        // Update default app's active app reference
        this.defaultApp.setActiveApp(this.activeApp);
    }

    update(handsData, gestures) {
        if (!this.streamSize || this.streamSize.width !== handsData.stream_info.width || this.streamSize.height !== handsData.stream_info.height) {
            // Update stream size if it has changed
            this.streamSize = {width: handsData.stream_info.width, height: handsData.stream_info.height};
            for (const app of this.applications.values()) {
                app.setStreamSize(this.streamSize);
            }
        }
        
        // Update only the active app
        if (this.activeApp) {
            this.activeApp.update(handsData, gestures);
        }

        // Mark all air taps as handled (we receive them for many frames so we only handle them once)
        this.markAirTapsAsHandled(handsData.airTapData);
    }

    draw() {
        // Draw only the active app
        if (this.activeApp) {
            this.activeApp.draw();
            // Draw pointers after the app has drawn its content
            this.activeApp.drawPointers();
        }
    }


    getActiveApp() {
        return this.activeApp;
    }
    
    getDefaultApp() {
        return this.defaultApp;
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
