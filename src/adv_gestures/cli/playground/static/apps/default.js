import { BaseApplication } from './_base.js';
import { DP, DrawingStyles } from '../drawing-primitives.js';

// Cache for colored emoji canvases
const emojiCanvasCache = new Map();

// Create a colored version of an emoji
function createColoredEmoji(emoji, color, fontSize) {
    const cacheKey = `${emoji}_${color}_${fontSize}`;
    
    // Check if we already have this emoji in cache
    if (emojiCanvasCache.has(cacheKey)) {
        return emojiCanvasCache.get(cacheKey);
    }
    
    // Create temporary canvases
    const emojiCanvas = document.createElement('canvas');
    const emojiCtx = emojiCanvas.getContext('2d');
    
    // First canvas: render and colorize the emoji
    const emojiSize = Math.ceil(fontSize * 1.5);
    emojiCanvas.width = emojiSize;
    emojiCanvas.height = emojiSize;
    
    // Draw emoji in the center
    emojiCtx.font = `${fontSize}px sans-serif`;
    emojiCtx.textAlign = 'center';
    emojiCtx.textBaseline = 'middle';
    emojiCtx.fillText(emoji, emojiSize / 2, emojiSize / 2);
    
    // Get the image data and colorize
    const imageData = emojiCtx.getImageData(0, 0, emojiSize, emojiSize);
    const data = imageData.data;
    
    // Parse the hex color
    const r = parseInt(color.slice(1, 3), 16);
    const g = parseInt(color.slice(3, 5), 16);
    const b = parseInt(color.slice(5, 7), 16);
    
    // Replace all non-transparent pixels with accent color
    for (let i = 0; i < data.length; i += 4) {
        const alpha = data[i + 3];
        if (alpha > 0) {
            data[i] = r;
            data[i + 1] = g;
            data[i + 2] = b;
        }
    }

    // Put the colorized image data back
    emojiCtx.putImageData(imageData, 0, 0);
    
    // Final canvas with glow
    const glowRadius = 20;
    const padding = glowRadius * 2;
    const finalCanvas = document.createElement('canvas');
    const finalCtx = finalCanvas.getContext('2d');
    const finalSize = emojiSize + padding;
    finalCanvas.width = finalSize;
    finalCanvas.height = finalSize;
    
    const drawOffset = padding / 2;
    
    // First layer: strong outer glow
    finalCtx.save();
    finalCtx.shadowColor = color;
    finalCtx.shadowBlur = 20;
    finalCtx.shadowOffsetX = 0;
    finalCtx.shadowOffsetY = 0;
    finalCtx.drawImage(emojiCanvas, drawOffset, drawOffset);
    finalCtx.restore();
    
    // Second layer: tighter inner glow for more intensity
    finalCtx.save();
    finalCtx.shadowColor = color;
    finalCtx.shadowBlur = 10;
    finalCtx.shadowOffsetX = 0;
    finalCtx.shadowOffsetY = 0;
    finalCtx.globalAlpha = 0.8;
    finalCtx.drawImage(emojiCanvas, drawOffset, drawOffset);
    finalCtx.restore();
    
    // Cache the final canvas
    emojiCanvasCache.set(cacheKey, finalCanvas);
    
    return finalCanvas;
}

// Floating emoji class for gesture animations
class FloatingEmoji {
    constructor(emoji, x, canvasWidth, canvasHeight) {
        this.emoji = emoji;
        this.canvasWidth = canvasWidth;
        this.canvasHeight = canvasHeight;
        this.x = x;
        this.y = canvasHeight; // Start from bottom
        this.startX = x;
        this.startTime = Date.now();
        
        // Base configuration values
        this.baseSpeedPercent = 40; // % of canvas height per second
        this.baseWaveAmplitudePercent = 5; // % of canvas width for horizontal wave amplitude
        this.baseWaveFrequency = 0.5; // Number of complete sine waves per second
        this.baseFadeStartTime = 1; // Start fading after x seconds
        this.baseMaxLifetime = 2.5; // Total lifetime in seconds
        
        // Apply random variation of ±10% to create diversity (different for each parameter)
        const randomVariation = () => 0.8 + Math.random() * 0.4; // Returns value between 0.9 and 1.1
        
        this.speedPercent = this.baseSpeedPercent * randomVariation();
        this.waveAmplitudePercent = this.baseWaveAmplitudePercent * randomVariation();
        this.waveFrequency = this.baseWaveFrequency * randomVariation();
        this.fadeStartTime = this.baseFadeStartTime * randomVariation();
        this.maxLifetime = this.baseMaxLifetime * randomVariation();
        
        // Fixed values (no variation)
        this.fontSize = 48;
        this.opacity = 1;
        
        // Create colored emoji canvas
        this.coloredEmojiCanvas = createColoredEmoji(emoji, DrawingStyles.colors.accent, this.fontSize);
    }
    
    update() {
        const elapsedSeconds = (Date.now() - this.startTime) / 1000;
        
        // Calculate vertical position based on speed (% of canvas height per second)
        const pixelsPerSecond = (this.speedPercent / 100) * this.canvasHeight;
        this.y = this.canvasHeight - (pixelsPerSecond * elapsedSeconds);
        
        // Zigzag movement using sine wave
        // waveFrequency = complete sine waves per second
        // 2π radians = one complete wave, so multiply by 2π for full waves
        const waveAmplitudePixels = (this.waveAmplitudePercent / 100) * this.canvasWidth;
        this.x = this.startX + Math.sin(elapsedSeconds * this.waveFrequency * 2 * Math.PI) * waveAmplitudePixels;
        
        // Fade out near the end
        if (elapsedSeconds > this.fadeStartTime) {
            const fadeProgress = (elapsedSeconds - this.fadeStartTime) / (this.maxLifetime - this.fadeStartTime);
            this.opacity = Math.max(0, 1 - fadeProgress);
        }
        
        // Return false when lifetime exceeded or emoji is off screen
        return elapsedSeconds < this.maxLifetime && this.y > -50;
    }
    
    draw(ctx) {
        ctx.save();
        ctx.globalAlpha = this.opacity;
        
        // Draw the colored emoji canvas centered at the position
        const canvasSize = this.coloredEmojiCanvas.width;
        const drawX = this.x - canvasSize / 2;
        const drawY = this.y - canvasSize / 2;
        ctx.drawImage(this.coloredEmojiCanvas, drawX, drawY);
        
        ctx.restore();
    }
}

export class DefaultApplication extends BaseApplication {
    constructor(applicationManager) {
        super('default', applicationManager);
        this.activeApp = null;
        this.iconSize = DrawingStyles.metrics.iconSize;
        this.iconSpacing = DrawingStyles.metrics.iconSpacing;
        this.iconRects = new Map(); // Store icon rectangles for click detection
        this.showIcons = false; // Icons hidden by default
        this.showPointers = this.showIcons; // Only show pointers if icons are visible
        
        // Animation properties
        this.ICON_FADE_DURATION = 500; // Half second animation duration
        this.iconOpacity = 0; // Current opacity (0-1)
        this.targetIconOpacity = 0; // Target opacity to animate to
        this.opacityAnimationStart = null; // Timestamp when animation started
        
        // Emoji system properties
        this.floatingEmojis = [];
        this.activeGestures = new Set();
        this.gestureSpawnTimers = new Map();
        this.EMOJI_SPAWN_INTERVAL = 200; // Spawn new emoji every 500ms

        // Gesture to emoji mapping with blocking configuration
        this.gestureEmojiMap = {
            // Single hand gestures
            'CLOSED_FIST': { emoji: '✊' },
            'POINTING_UP': { emoji: '☝️' },
            'THUMB_DOWN': { emoji: '👎' },
            'THUMB_UP': { emoji: '👍' },
            'VICTORY': { emoji: '✌️' },
            'LOVE': { emoji: '🤟' },
            
            'MIDDLE_FINGER': { emoji: '🖕' },
            'SPOCK': { emoji: '🖖' },
            'ROCK': { emoji: '🤘' },
            'OK': { emoji: '👌' },
            // 'STOP': { emoji: '✋' },
            // 'GUN': { emoji: '🔫' },
            'WAVE': { emoji: '👋', blocks: ['STOP'] },
            'NO': { emoji: '🚫', blocks: ['POINTING_UP'] },
            
            // Two hands gestures
            'CLAP': { emoji: '👏' },
        };
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
        
        // Update and draw floating emojis
        this.updateAndDrawEmojis();
        
        // Update opacity animation
        this.updateOpacityAnimation();
        
        // Draw application icons with animated opacity
        if (this.iconOpacity > 0) {
            this.ctx.save();
            this.ctx.globalAlpha = this.iconOpacity;
            
            for (const [app, rect] of this.iconRects.entries()) {
                // In default app, all icons are drawn the same way (no active/inactive distinction)
                app.drawIcon(this.ctx, rect.x, rect.y, this.iconSize, false);
            }
            
            this.ctx.restore();
        }
    }
    
    update(handsData, gestures) {
        super.update(handsData, gestures);
        
        // Check for SNAP gesture to toggle icons visibility
        if (this.isGestureJustAdded('DOUBLE_SNAP')) {
            this.showIcons = !this.showIcons;
            this.showPointers = this.showIcons;
            
            // Start opacity animation
            this.targetIconOpacity = this.showIcons ? 1 : 0;
            this.opacityAnimationStart = Date.now();
            
            // If icons are now shown, clear all active emojis and timers
            if (this.showIcons) {
                this.clearAllEmojiTimers();
                this.floatingEmojis = [];
            }
        }
        
        // Handle emoji spawning for tracked gestures (only when icons are hidden)
        if (!this.showIcons) {
            this.handleEmojiGestures(gestures);
        }

        // Do nothing more if icons are hidden
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
    
    updateOpacityAnimation() {
        // Skip if already at target
        if (this.iconOpacity === this.targetIconOpacity) return;
        
        if (this.opacityAnimationStart) {
            const elapsed = Date.now() - this.opacityAnimationStart;
            const progress = Math.min(elapsed / this.ICON_FADE_DURATION, 1);
            
            // Linear interpolation
            const startOpacity = this.targetIconOpacity === 1 ? 0 : 1;
            this.iconOpacity = startOpacity + (this.targetIconOpacity - startOpacity) * progress;
            
            // Animation complete
            if (progress >= 1) {
                this.iconOpacity = this.targetIconOpacity;
                this.opacityAnimationStart = null;
            }
        }
    }
    
    handleEmojiGestures(gestures) {
        // Track currently active gestures that have emojis
        const currentEmojiGestures = new Set();
        
        if (gestures && gestures.active) {
            // Check all categories (left, right, both)
            for (const category of ['left', 'right', 'both']) {
                if (gestures.active[category]) {
                    for (const gesture of gestures.active[category]) {
                        if (this.gestureEmojiMap[gesture]) {
                            currentEmojiGestures.add(gesture);
                        }
                    }
                }
            }
        }
        
        // Get currently blocked gestures based on active emoji gestures
        const blockedGestures = this.getBlockedGestures();
        
        // Start spawning for newly active gestures (if not blocked)
        for (const gesture of currentEmojiGestures) {
            if (!this.activeGestures.has(gesture) && !blockedGestures.has(gesture)) {
                // New gesture started and not blocked
                this.activeGestures.add(gesture);
                this.spawnEmoji(gesture); // Spawn first emoji immediately
                
                // Set up periodic spawning
                this.gestureSpawnTimers.set(gesture, setInterval(() => {
                    // Only spawn if icons are still hidden and gesture is not blocked
                    if (!this.showIcons && !this.getBlockedGestures().has(gesture)) {
                        this.spawnEmoji(gesture);
                    }
                }, this.EMOJI_SPAWN_INTERVAL));
            }
        }
        
        // Stop spawning for gestures that are no longer active or are now blocked
        for (const gesture of this.activeGestures) {
            if (!currentEmojiGestures.has(gesture) || blockedGestures.has(gesture)) {
                // Gesture ended or is now blocked
                this.activeGestures.delete(gesture);
                
                // Clear the spawn timer
                const timer = this.gestureSpawnTimers.get(gesture);
                if (timer) {
                    clearInterval(timer);
                    this.gestureSpawnTimers.delete(gesture);
                }
            }
        }
    }
    
    getBlockedGestures() {
        const blocked = new Set();
        
        // Check which gestures are currently blocking others via active emojis
        for (const emoji of this.floatingEmojis) {
            // Find which gesture this emoji belongs to
            for (const [gestureName, config] of Object.entries(this.gestureEmojiMap)) {
                if (config.emoji === emoji.emoji && config.blocks) {
                    // Add all gestures blocked by this one
                    for (const blockedGesture of config.blocks) {
                        blocked.add(blockedGesture);
                    }
                }
            }
        }
        
        return blocked;
    }
    
    spawnEmoji(gesture) {
        if (!this.width || !this.height) return;
        
        const config = this.gestureEmojiMap[gesture];
        if (!config || !config.emoji) return;
        
        // Random x position in the middle 60% of the screen
        const x = this.width * 0.2 + Math.random() * this.width * 0.6;
        
        this.floatingEmojis.push(new FloatingEmoji(config.emoji, x, this.width, this.height));
    }
    
    updateAndDrawEmojis() {
        // Update emojis and remove dead ones
        this.floatingEmojis = this.floatingEmojis.filter(emoji => emoji.update());
        
        // Draw all emojis
        for (const emoji of this.floatingEmojis) {
            emoji.draw(this.ctx);
        }
    }
    
    clearAllEmojiTimers() {
        // Clear all spawn timers
        for (const timer of this.gestureSpawnTimers.values()) {
            clearInterval(timer);
        }
        this.gestureSpawnTimers.clear();
        this.activeGestures.clear();
    }
}
