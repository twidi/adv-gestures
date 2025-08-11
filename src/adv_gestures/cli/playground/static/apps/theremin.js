import { BaseApplication } from './_base.js';
import { DP, DrawingStyles } from '../drawing-primitives.js';

export class ThereminApplication extends BaseApplication {
    constructor(applicationManager) {
        super('theremin', applicationManager);
        this.showPointers = false;
        
        // Icon from FontAwesome - music note
        this.iconSvgPath = "M486.3 120.1L262.3 169.9C258.6 170.7 256 174 256 177.7L256 226.1L496 172.8L496 127.9C496 122.8 491.3 119 486.3 120.1zM496 330.7L496 221.9L256 275.2L256 463.9C256 508.1 213 543.9 160 543.9C107 543.9 64 508.1 64 463.9C64 419.7 107 383.9 160 383.9C177.5 383.9 193.9 387.8 208 394.6L208 177.6C208 151.4 226.2 128.6 251.9 122.9L475.9 73.1C510.9 65.3 544 91.9 544 127.8L544 400C544 444.2 501 480 448 480C395 480 352 444.2 352 400C352 355.8 395 320 448 320C465.5 320 481.9 323.9 496 330.7zM496 400C496 382.3 474.5 368 448 368C421.5 368 400 382.3 400 400C400 417.7 421.5 432 448 432C474.5 432 496 417.7 496 400zM208 464C208 446.3 186.5 432 160 432C133.5 432 112 446.3 112 464C112 481.7 133.5 496 160 496C186.5 496 208 481.7 208 464z";

        // Audio context and nodes
        this.audioContext = null;
        this.oscillator1 = null;
        this.oscillator2 = null;
        this.gainNode = null;
        this.filter = null;
        this.vibrato = null;
        this.vibratoGain = null;
        this.isPlaying = false;
        
        // Frequency range (in Hz) - 4 full octaves from C2 to C6
        this.minFreq = 65.41;  // C2
        this.maxFreq = 1046.5; // C6
        
        // Visual properties
        this.leftHandZone = { x: 0, width: 0.3 }; // Left 30% for volume (vertical control)
        this.rightHandZone = { x: 0.3, width: 0.7 }; // Right 70% for pitch (horizontal control)
        
        // Smoothing for audio parameters
        this.currentFreq = 0;
        this.currentVolume = 0;
        this.targetFreq = 0;
        this.targetVolume = 0;
        this.smoothingFactor = 1.0; // 1.0 = instant (no smoothing), 0.15 = smooth transitions
        
        // Visual feedback
        this.waveformPoints = [];
        this.waveformHistory = 50; // Number of points to keep
        
        // Hand tracking
        this.leftHand = null;
        this.rightHand = null;
        
        // Performance optimization
        this.lastUpdateTime = 0;
        this.updateInterval = 1000 / 60; // 60 FPS for audio updates
    }
    
    activate() {
        super.activate();
        this.initAudio();
    }
    
    deactivate() {
        super.deactivate();
        this.stopAudio();
        if (this.audioContext) {
            this.audioContext.close();
            this.audioContext = null;
        }
    }
    
    initAudio() {
        if (this.audioContext) return;
        
        try {
            this.audioContext = new (window.AudioContext || window.webkitAudioContext)();
            
            // Create two oscillators for a richer sound
            this.oscillator1 = this.audioContext.createOscillator();
            this.oscillator1.type = 'sine'; // Main tone
            this.oscillator1.frequency.value = this.minFreq;
            
            this.oscillator2 = this.audioContext.createOscillator();
            this.oscillator2.type = 'triangle'; // Adds warmth
            this.oscillator2.frequency.value = this.minFreq;
            
            // Create vibrato (low frequency oscillator for pitch modulation)
            this.vibrato = this.audioContext.createOscillator();
            this.vibrato.frequency.value = 5; // 5 Hz vibrato
            this.vibrato.type = 'sine';
            
            this.vibratoGain = this.audioContext.createGain();
            this.vibratoGain.gain.value = 3; // Subtle vibrato depth (±3 Hz)
            
            // Create filter for tone shaping (classic theremin warmth)
            this.filter = this.audioContext.createBiquadFilter();
            this.filter.type = 'lowpass';
            this.filter.frequency.value = 2000; // Cut harsh high frequencies
            this.filter.Q.value = 0.5;
            
            // Create gain node for volume control
            this.gainNode = this.audioContext.createGain();
            this.gainNode.gain.value = 0;
            
            // Connect vibrato to oscillator frequencies
            this.vibrato.connect(this.vibratoGain);
            this.vibratoGain.connect(this.oscillator1.frequency);
            this.vibratoGain.connect(this.oscillator2.frequency);
            
            // Mix oscillators through filter to gain
            this.oscillator1.connect(this.filter);
            this.oscillator2.connect(this.filter);
            this.filter.connect(this.gainNode);
            this.gainNode.connect(this.audioContext.destination);
            
            // Adjust oscillator 2 volume to be slightly quieter
            const osc2Gain = this.audioContext.createGain();
            osc2Gain.gain.value = 0.3;
            this.oscillator2.disconnect();
            this.oscillator2.connect(osc2Gain);
            osc2Gain.connect(this.filter);
            
            // Start all oscillators
            this.oscillator1.start();
            this.oscillator2.start();
            this.vibrato.start();
            this.isPlaying = true;
        } catch (error) {
            console.error('Failed to initialize audio:', error);
        }
    }
    
    stopAudio() {
        if (this.oscillator1) {
            this.oscillator1.stop();
            this.oscillator1 = null;
        }
        if (this.oscillator2) {
            this.oscillator2.stop();
            this.oscillator2 = null;
        }
        if (this.vibrato) {
            this.vibrato.stop();
            this.vibrato = null;
        }
        this.isPlaying = false;
    }
    
    update(handsData, gestures) {
        super.update(handsData, gestures);

        // Check for DOUBLE_SNAP gesture to exit the app
        if (this.isGestureJustAdded('DOUBLE_SNAP')) {
            this.exit();
            return;
        }

        // Track hands
        this.leftHand = handsData.left;
        this.rightHand = handsData.right;

        // Update audio parameters based on hand positions
        this.updateAudioFromHands();
    }
    
    updateAudioFromHands() {
        const now = Date.now();
        if (now - this.lastUpdateTime < this.updateInterval) return;
        this.lastUpdateTime = now;
        
        if (!this.audioContext || !this.isPlaying) return;
        
        // Calculate target frequency from right hand (pitch control)
        if (this.rightHand && this.rightHand.palm && this.streamSize) {
            // Use horizontal position for pitch (close to right edge = high, far = low)
            const x = this.rightHand.palm.centroid.x;
            // Normalize X coordinate (centroid is in absolute pixels)
            const normalizedX = x / this.streamSize.width;
            
            // Only respond if hand is in the right zone (right 70% of screen)
            if (normalizedX >= this.rightHandZone.x) {
                // Calculate position within the zone (0 = left edge of zone, 1 = right edge)
                const positionInZone = (normalizedX - this.rightHandZone.x) / this.rightHandZone.width;
                const pitchValue = Math.max(0, Math.min(1, positionInZone));
                
                // Linear scaling to match the visual grid
                const freqRange = this.maxFreq - this.minFreq;
                this.targetFreq = this.minFreq + pitchValue * freqRange;
            } else {
                this.targetFreq = 0; // No sound if hand is outside the zone
            }
        } else {
            this.targetFreq = 0; // No sound without right hand
            this.targetVolume = 0; // Also cut volume when right hand is missing
        }
        
        // Calculate target volume from left hand (volume control)
        // Only calculate volume if right hand is present (we need both hands for sound)
        if (this.rightHand && this.leftHand && this.leftHand.palm && this.streamSize) {
            const x = this.leftHand.palm.centroid.x;
            const y = this.leftHand.palm.centroid.y;
            // Normalize coordinates (centroid is in absolute pixels)
            const normalizedX = x / this.streamSize.width;
            const normalizedY = y / this.streamSize.height;

            // Only respond if hand is in the left zone (left 30% of screen)
            if (normalizedX <= this.leftHandZone.width) {
                // Use vertical position for volume (top = loud, bottom = quiet)
                // Invert so that higher position = louder
                const volumeNormalized = Math.max(0, Math.min(1, 1 - normalizedY));
                
                // Apply curve for more natural volume control
                this.targetVolume = Math.pow(volumeNormalized, 1.5) * 0.5; // Max volume 0.5
            } else {
                this.targetVolume = 0; // No volume if hand is outside the zone
            }
            
            // Additional control: closed fist = mute
            if (this.isGestureActive('CLOSED_FIST', 'left')) {
                this.targetVolume = 0;
            }
        } else {
            this.targetVolume = 0; // No volume without both hands
        }
        
        // Smooth the changes
        this.currentFreq += (this.targetFreq - this.currentFreq) * this.smoothingFactor;
        this.currentVolume += (this.targetVolume - this.currentVolume) * this.smoothingFactor;
        
        // Apply to audio nodes with exponential ramping for smooth transitions
        if (this.oscillator1 && this.oscillator2 && this.gainNode && this.filter) {
            const rampTime = this.audioContext.currentTime + 0.05; // 50ms ramp
            
            if (this.currentFreq > 0) {
                // Set main frequencies
                this.oscillator1.frequency.exponentialRampToValueAtTime(
                    Math.max(this.currentFreq, 20), // Avoid 0 frequency
                    rampTime
                );
                
                // Second oscillator slightly detuned for richness
                this.oscillator2.frequency.exponentialRampToValueAtTime(
                    Math.max(this.currentFreq * 1.005, 20), // Slight detune
                    rampTime
                );
                
                // Adjust filter cutoff based on frequency (brighter for higher notes)
                const filterFreq = Math.min(this.currentFreq * 4 + 500, 4000);
                this.filter.frequency.exponentialRampToValueAtTime(
                    filterFreq,
                    rampTime
                );
            }
            
            this.gainNode.gain.linearRampToValueAtTime(
                this.currentVolume,
                rampTime
            );
        }
        
        // Update waveform visualization data
        if (this.currentVolume > 0.01) {
            this.waveformPoints.push({
                freq: this.currentFreq,
                volume: this.currentVolume,
                time: now
            });
            
            // Keep only recent points
            if (this.waveformPoints.length > this.waveformHistory) {
                this.waveformPoints.shift();
            }
        }
    }
    
    draw() {
        if (!this.ctx || !this.isActive) return;
        
        // Clear canvas
        DP.clearCanvas(this.ctx);
        
        const ctx = this.ctx;
        
        // Draw background zones
        this.drawZones();
        
        // Draw frequency grid
        this.drawFrequencyGrid();
        
        // Draw waveform visualization
        this.drawWaveform();
        
        // Draw hand indicators
        this.drawHandIndicators();
        
        // Draw instructions
        this.drawInstructions();
    }
    
    drawZones() {
        const ctx = this.ctx;
        
        // Left zone (volume control - vertical)
        ctx.save();
        ctx.fillStyle = DrawingStyles.colors.primary;
        ctx.globalAlpha = 0.05;
        ctx.fillRect(0, 0, this.width * this.leftHandZone.width, this.height);
        
        // Add vertical gradient for volume indication (top = loud)
        const gradient = ctx.createLinearGradient(0, 0, 0, this.height);
        gradient.addColorStop(0, DrawingStyles.colors.accent);
        gradient.addColorStop(1, 'transparent');
        ctx.fillStyle = gradient;
        ctx.globalAlpha = 0.1;
        ctx.fillRect(0, 0, this.width * this.leftHandZone.width, this.height);
        ctx.restore();
        
        // Right zone (pitch control - horizontal)
        ctx.save();
        ctx.fillStyle = DrawingStyles.colors.primary;
        ctx.globalAlpha = 0.05;
        ctx.fillRect(this.width * this.rightHandZone.x, 0, this.width * this.rightHandZone.width, this.height);
        
        // Add horizontal gradient for pitch indication (right edge = antenna = high pitch)
        const pitchGradient = ctx.createLinearGradient(this.width * this.rightHandZone.x, 0, this.width, 0);
        pitchGradient.addColorStop(0, 'transparent');
        pitchGradient.addColorStop(1, DrawingStyles.colors.accent);
        ctx.fillStyle = pitchGradient;
        ctx.globalAlpha = 0.1;
        ctx.fillRect(this.width * this.rightHandZone.x, 0, this.width * this.rightHandZone.width, this.height);
        ctx.restore();
    }
    
    drawFrequencyGrid() {
        const ctx = this.ctx;
        const rightZoneStart = this.width * this.rightHandZone.x;
        const rightZoneWidth = this.width * this.rightHandZone.width;
        
        ctx.save();
        
        // Define octave boundaries (C notes mark octave changes)
        const octaves = [
            { freq: 65.41, label: 'C2', octave: '2', name: 'Bass' },
            { freq: 130.81, label: 'C3', octave: '3', name: 'Small' },
            { freq: 261.63, label: 'C4', octave: '4', name: 'Middle' },
            { freq: 523.25, label: 'C5', octave: '5', name: 'Treble' },
            { freq: 1046.5, label: 'C6', octave: '6', name: '' },
        ];
        
        // Draw octave regions with subtle background colors
        let prevX = rightZoneStart;
        for (let i = 0; i < octaves.length; i++) {
            const octaveData = octaves[i];
            const normalizedFreq = (octaveData.freq - this.minFreq) / (this.maxFreq - this.minFreq);
            const x = rightZoneStart + rightZoneWidth * normalizedFreq;
            
            // Draw octave background
            ctx.fillStyle = DrawingStyles.colors.primary;
            ctx.globalAlpha = i % 2 === 0 ? 0.03 : 0.06; // Alternate shading
            ctx.fillRect(prevX, 0, x - prevX, this.height);
            
            // Draw octave divider line
            ctx.strokeStyle = DrawingStyles.colors.accent;
            ctx.lineWidth = 2;
            ctx.globalAlpha = 0.4;
            ctx.beginPath();
            ctx.moveTo(x, 0);
            ctx.lineTo(x, this.height);
            ctx.stroke();
            
            // Add octave label
            ctx.save();
            ctx.fillStyle = DrawingStyles.colors.accent;
            ctx.globalAlpha = 0.7;
            ctx.font = 'bold 14px monospace';
            ctx.textAlign = 'center';
            ctx.fillText(octaveData.label, x, 20);
            
            // Show octave name if we have one for this range
            if (i > 0 && octaves[i-1].name) {
                const octaveName = octaves[i-1].name;
                ctx.fillText(octaveName, (prevX + x) / 2, this.height - 15);
            }
            ctx.restore();
            
            prevX = x;
        }
        
        // Fill the last octave region
        ctx.fillStyle = DrawingStyles.colors.primary;
        ctx.globalAlpha = octaves.length % 2 === 0 ? 0.03 : 0.06;
        ctx.fillRect(prevX, 0, rightZoneStart + rightZoneWidth - prevX, this.height);
        
        // No need for last region label since C6 is at the edge
        
        // Draw minor note lines (more subtle)
        ctx.strokeStyle = DrawingStyles.colors.primary;
        ctx.globalAlpha = 0.15;
        ctx.lineWidth = 1;
        const minorNotes = [
            // Octave 2
            73.42, 82.41, 87.31, 98.00, 110.00, 123.47,
            // Octave 3
            146.83, 164.81, 174.61, 196.00, 220.00, 246.94,
            // Octave 4
            293.66, 329.63, 349.23, 392.00, 440.00, 493.88,
            // Octave 5
            587.33, 659.25, 698.46, 783.99, 880.00, 987.77
        ];
        for (const freq of minorNotes) {
            const normalizedFreq = (freq - this.minFreq) / (this.maxFreq - this.minFreq);
            const x = rightZoneStart + rightZoneWidth * normalizedFreq;
            
            ctx.beginPath();
            ctx.moveTo(x, 30);
            ctx.lineTo(x, this.height - 20);
            ctx.stroke();
        }
        
        // Draw antenna indicator (right edge)
        ctx.save();
        ctx.strokeStyle = DrawingStyles.colors.accent;
        ctx.lineWidth = 3;
        ctx.globalAlpha = 0.5;
        ctx.beginPath();
        ctx.moveTo(this.width - 2, 0);
        ctx.lineTo(this.width - 2, this.height);
        ctx.stroke();
        ctx.restore();
        
        ctx.restore();
    }
    
    drawWaveform() {
        if (this.currentVolume < 0.01) return;
        
        const ctx = this.ctx;
        const centerY = this.height / 2;
        const maxAmplitude = this.height * 0.25;
        
        ctx.save();
        ctx.strokeStyle = DrawingStyles.colors.accent;
        ctx.lineWidth = 2;
        
        // Draw complex waveform that represents our actual sound
        ctx.beginPath();
        const now = Date.now() * 0.001; // Convert to seconds
        const samplesPerPixel = 2;
        
        for (let x = 0; x < this.width; x += samplesPerPixel) {
            // Time offset for this x position (creates traveling wave effect)
            const timeOffset = x / this.width * 0.01;
            const t = now - timeOffset;
            
            // Main oscillator (sine wave)
            const mainPhase = t * this.currentFreq * 2 * Math.PI;
            const mainWave = Math.sin(mainPhase);
            
            // Second oscillator (triangle wave simulation, slightly detuned)
            const detunedFreq = this.currentFreq * 1.005;
            const trianglePhase = t * detunedFreq * 2 * Math.PI;
            // Approximate triangle wave with harmonics
            const triangleWave = (Math.sin(trianglePhase) * 0.8 - 
                                 Math.sin(trianglePhase * 3) / 9 + 
                                 Math.sin(trianglePhase * 5) / 25) * 0.3;
            
            // Vibrato effect (modulates the combined wave)
            const vibratoPhase = t * 5 * 2 * Math.PI; // 5 Hz vibrato
            const vibratoEffect = 1 + Math.sin(vibratoPhase) * 0.01; // ±1% modulation
            
            // Combine waves (main + triangle with vibrato)
            const combinedWave = (mainWave + triangleWave) * vibratoEffect;
            
            // Apply amplitude based on current volume
            const amplitude = maxAmplitude * this.currentVolume * 2;
            const y = centerY + combinedWave * amplitude;
            
            if (x === 0) {
                ctx.moveTo(x, y);
            } else {
                ctx.lineTo(x, y);
            }
        }
        
        // Add subtle glow effect
        ctx.globalAlpha = Math.min(this.currentVolume * 2, 0.8);
        ctx.stroke();
        
        // Second pass with slight blur for glow
        ctx.globalAlpha = Math.min(this.currentVolume, 0.3);
        ctx.lineWidth = 4;
        ctx.stroke();
        
        ctx.restore();
    }
    
    drawHandIndicators() {
        const ctx = this.ctx;
        
        // Draw left hand indicator (volume)
        if (this.leftHand && this.leftHand.palm) {
            const scaledPalm = this.scalePoint(this.leftHand.palm.centroid);
            
            // Draw volume indicator
            ctx.save();
            ctx.fillStyle = DrawingStyles.colors.accent;
            ctx.globalAlpha = 0.8;
            
            // Draw a circle that grows with volume
            const radius = 10 + this.currentVolume * 30;
            ctx.beginPath();
            ctx.arc(scaledPalm.x, scaledPalm.y, radius, 0, Math.PI * 2);
            ctx.fill();
            
            // Add volume percentage text
            ctx.fillStyle = DrawingStyles.colors.text;
            ctx.font = 'bold 14px monospace';
            ctx.textAlign = 'center';
            ctx.fillText(
                `${Math.round(this.currentVolume * 200)}%`,
                scaledPalm.x,
                scaledPalm.y - radius - 10
            );
            
            ctx.restore();
        }
        
        // Draw right hand indicator (pitch)
        if (this.rightHand && this.rightHand.palm) {
            const scaledPalm = this.scalePoint(this.rightHand.palm.centroid);
            
            ctx.save();
            
            // Draw pitch indicator line (vertical)
            ctx.strokeStyle = DrawingStyles.colors.accent;
            ctx.lineWidth = 3;
            ctx.globalAlpha = 0.8;
            ctx.setLineDash([5, 5]);
            
            ctx.beginPath();
            ctx.moveTo(scaledPalm.x, 0);
            ctx.lineTo(scaledPalm.x, this.height);
            ctx.stroke();
            
            // Draw frequency text
            ctx.fillStyle = DrawingStyles.colors.accent;
            ctx.font = 'bold 16px monospace';
            ctx.textAlign = 'center';
            ctx.setLineDash([]);
            ctx.fillText(
                `${Math.round(this.currentFreq)} Hz`,
                scaledPalm.x,
                scaledPalm.y - 20
            );
            
            // Draw note names if applicable (both notations)
            const noteData = this.frequencyToNote(this.currentFreq);
            if (noteData) {
                ctx.font = 'bold 20px monospace';
                ctx.fillText(noteData.note, scaledPalm.x, scaledPalm.y + 30);
                ctx.font = '16px monospace';
                ctx.fillText(noteData.solfege, scaledPalm.x, scaledPalm.y + 50);
            }
            
            ctx.restore();
        }
    }
    
    drawInstructions() {
        const ctx = this.ctx;
        
        ctx.save();
        ctx.fillStyle = DrawingStyles.colors.text;
        ctx.font = '14px monospace';
        ctx.globalAlpha = 0.6;
        
        // Left hand instructions
        ctx.textAlign = 'left';
        ctx.fillText('LEFT HAND: Volume (move ↑↓)', 10, 30);
        ctx.fillText('Close fist to mute', 10, 50);
        
        // Right hand instructions
        ctx.textAlign = 'right';
        ctx.fillText('RIGHT HAND: Pitch (move ←→)', this.width - 10, 30);
        ctx.fillText('Closer to edge = higher', this.width - 10, 50);

        ctx.restore();
    }
    
    frequencyToNote(freq) {
        // Note mapping with both notations - 4 full octaves
        const notes = [
            // Octave 2
            { freq: 65.41, note: 'C2', solfege: 'DO' },
            { freq: 73.42, note: 'D2', solfege: 'RE' },
            { freq: 82.41, note: 'E2', solfege: 'MI' },
            { freq: 87.31, note: 'F2', solfege: 'FA' },
            { freq: 98.00, note: 'G2', solfege: 'SOL' },
            { freq: 110.00, note: 'A2', solfege: 'LA' },
            { freq: 123.47, note: 'B2', solfege: 'SI' },
            // Octave 3
            { freq: 130.81, note: 'C3', solfege: 'DO' },
            { freq: 146.83, note: 'D3', solfege: 'RE' },
            { freq: 164.81, note: 'E3', solfege: 'MI' },
            { freq: 174.61, note: 'F3', solfege: 'FA' },
            { freq: 196.00, note: 'G3', solfege: 'SOL' },
            { freq: 220.00, note: 'A3', solfege: 'LA' },
            { freq: 246.94, note: 'B3', solfege: 'SI' },
            // Octave 4
            { freq: 261.63, note: 'C4', solfege: 'DO' },
            { freq: 293.66, note: 'D4', solfege: 'RE' },
            { freq: 329.63, note: 'E4', solfege: 'MI' },
            { freq: 349.23, note: 'F4', solfege: 'FA' },
            { freq: 392.00, note: 'G4', solfege: 'SOL' },
            { freq: 440.00, note: 'A4', solfege: 'LA' },
            { freq: 493.88, note: 'B4', solfege: 'SI' },
            // Octave 5
            { freq: 523.25, note: 'C5', solfege: 'DO' },
            { freq: 587.33, note: 'D5', solfege: 'RE' },
            { freq: 659.25, note: 'E5', solfege: 'MI' },
            { freq: 698.46, note: 'F5', solfege: 'FA' },
            { freq: 783.99, note: 'G5', solfege: 'SOL' },
            { freq: 880.00, note: 'A5', solfege: 'LA' },
            { freq: 987.77, note: 'B5', solfege: 'SI' },
            // Octave 6
            { freq: 1046.50, note: 'C6', solfege: 'DO' },
        ];
        
        // Find closest note
        let closestNote = null;
        let minDiff = Infinity;
        
        for (const noteData of notes) {
            const diff = Math.abs(freq - noteData.freq);
            if (diff < minDiff && diff < 20) { // Within 20Hz tolerance
                minDiff = diff;
                closestNote = noteData;
            }
        }
        
        return closestNote;
    }
}
