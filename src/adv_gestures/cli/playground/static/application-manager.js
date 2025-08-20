import { DP, DrawingStyles } from './drawing-primitives.js';

// Static registry for self-registering applications
const applicationRegistry = {
    applications: new Map(),
    defaultAppClass: null,
    registrationOrder: []
};

export class ApplicationManager {

    static instance;

    constructor(canvasContainer, handledAirTaps, isMirrored = true) {
        this.applications = new Map();
        this.activeApp = null;
        this.defaultApp = null;
        this.canvasContainer = canvasContainer;
        this.width = 0;
        this.height = 0;
        this.handledAirTaps = handledAirTaps;
        this.applicationListeners = new Set(); // Listeners for app registration events
        this.isMirrored = isMirrored; // Store mirror state

        this.streamSize = null; // Will be set when stream info is available

        // Store singleton reference for late registrations
        ApplicationManager.instance = this;

        // Create instances from registered applications
        this.initializeApplications();
    }
    
    /**
     * Static method to register an application class
     * @param {Function} ApplicationClass - The application class constructor
     * @param {boolean} isDefault - Whether this is the default application
     */
    static register(ApplicationClass, isDefault = false) {
        const tempInstance = new ApplicationClass();
        const name = tempInstance.name;
        
        if (!name) {
            console.error('Application must have a name property:', ApplicationClass);
            return;
        }

        applicationRegistry.applications.set(name, ApplicationClass);
        
        if (isDefault) {
            applicationRegistry.defaultAppClass = ApplicationClass;
        } else {
            applicationRegistry.registrationOrder.push(ApplicationClass);
        }

        console.log(`Registered application: ${name}${isDefault ? ' (default)' : ''}`);
        
        // If manager instance exists, handle late registration
        if (ApplicationManager.instance) {
            ApplicationManager.instance.handleLateRegistration(ApplicationClass);
        }
    }
    
    initializeApplications() {
        // Create instance of default application first
        if (applicationRegistry.defaultAppClass) {
            const defaultApp = new applicationRegistry.defaultAppClass(this);
            this.applications.set(defaultApp.name, defaultApp);
            this.defaultApp = defaultApp;
            this.activeApp = defaultApp;
        }

        // Create instances of all other applications
        for (const [name, ApplicationClass] of applicationRegistry.applications) {
            // Skip if this is the default app (already created)
            if (ApplicationClass === applicationRegistry.defaultAppClass) continue;
            
            const app = new ApplicationClass(this);
            this.applications.set(app.name, app);
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
    
    /**
     * Add a listener for application registration events
     * @param {Function} listener - Callback function that receives the new app instance
     */
    addApplicationListener(listener) {
        this.applicationListeners.add(listener);
    }
    
    /**
     * Remove an application registration listener
     * @param {Function} listener - The listener to remove
     */
    removeApplicationListener(listener) {
        this.applicationListeners.delete(listener);
    }
    
    /**
     * Handle late registration of an application (after manager initialization)
     * @param {Function} ApplicationClass - The application class constructor
     */
    handleLateRegistration(ApplicationClass) {
        // Check if already registered
        const tempInstance = new ApplicationClass();
        const name = tempInstance.name;
        
        if (this.applications.has(name)) {
            return; // Already registered
        }
        
        console.log(`Handling late registration for: ${name}`);
        
        // Create the application instance
        const app = new ApplicationClass(this);
        this.applications.set(app.name, app);
        
        // Create canvas if we already have dimensions
        if (this.width > 0 && this.height > 0) {
            app.createCanvas(this.width, this.height);
        }
        
        // Set stream size if available
        if (this.streamSize) {
            app.setStreamSize(this.streamSize);
        }
        
        // Notify all listeners about the new application
        for (const listener of this.applicationListeners) {
            listener(app);
        }
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

// Expose ApplicationManager globally for self-registration
if (typeof window !== 'undefined') {
    window.ApplicationManager = ApplicationManager;
    
    // Notify the registrar that ApplicationManager is ready
    if (window.notifyApplicationManagerReady) {
        window.notifyApplicationManagerReady();
    }
}
