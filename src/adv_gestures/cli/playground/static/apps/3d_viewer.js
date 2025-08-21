import { BaseApplication } from './_base.js';
import { DP, DrawingStyles } from '../drawing-primitives.js';

// Three.js will be loaded lazily
let THREE = null;

export class ThreeDViewerApplication extends BaseApplication {
    constructor(applicationManager) {
        super('3d_viewer', applicationManager);
        // FontAwesome 360-degrees icon SVG path
        this.iconSvgPath = "M608 128C625.7 128 640 113.7 640 96C640 78.3 625.7 64 608 64C590.3 64 576 78.3 576 96C576 113.7 590.3 128 608 128zM24 128C10.7 128 0 138.7 0 152C0 165.3 10.7 176 24 176L107.2 176L36.1 282.7C31.2 290.1 30.7 299.5 34.9 307.3C39.1 315.1 47.1 320 56 320L88 320C118.9 320 144 345.1 144 376L144 416C144 442.5 122.5 464 96 464L94.5 464C78.5 464 63.5 456 54.6 442.6L44 426.7C36.6 415.7 21.7 412.7 10.7 420C-.3 427.3-3.3 442.3 4 453.3L14.6 469.2C32.5 496 62.4 512 94.5 512L96 512C149 512 192 469 192 416L192 376C192 322.7 152 278.8 100.4 272.7L172 165.3C176.9 157.9 177.4 148.5 173.2 140.7C169 132.9 160.9 128 152 128L24 128zM464 208C464 190.3 478.3 176 496 176C513.7 176 528 190.3 528 208L528 432C528 449.7 513.7 464 496 464C478.3 464 464 449.7 464 432L464 208zM576 432L576 208C576 163.8 540.2 128 496 128C451.8 128 416 163.8 416 208L416 432C416 476.2 451.8 512 496 512C540.2 512 576 476.2 576 432zM272 224C272 197.5 293.5 176 320 176C333.3 176 344 165.3 344 152C344 138.7 333.3 128 320 128C267 128 224 171 224 224L224 352L224 352.2L224 432C224 476.2 259.8 512 304 512C348.2 512 384 476.2 384 432L384 336C384 291.8 348.2 256 304 256C292.6 256 281.8 258.4 272 262.7L272 224zM304 304C321.7 304 336 318.3 336 336L336 432C336 449.7 321.7 464 304 464C286.3 464 272 449.7 272 432L272 336C272 318.3 286.3 304 304 304z";
        
        // Three.js setup
        this.scene = null;
        this.camera = null;
        this.renderer = null;
        this.landmarkSpheres = [];
        this.lineSegments = [];
        this.leftContainer = null;
        
        // Mouse controls
        this.isMouseDown = false;
        this.mouseX = 0;
        this.mouseY = 0;
        this.rotationX = 0;
        this.rotationY = 0;
        
        // Which hand to display (default: left, since most people use mouse with right hand)
        this.displayHand = 'left'; // 'right' or 'left'
    }
    
    getLandmarkColor(index) {
        const colors = {
            // Wrist and palm (red)
            0: [1.0, 0.0, 0.0],  // WRIST
            1: [1.0, 0.0, 0.0],  // THUMB_CMC
            2: [1.0, 0.0, 0.0], // THUMB_MCP
            // Thumb (cream white)
            3: [1.0, 0.99, 0.82], // THUMB_IP
            4: [1.0, 0.99, 0.82], // THUMB_TIP
            // Palm (red)
            5: [1.0, 0.0, 0.0],   // INDEX_MCP
            // Index (magenta)
            6: [1.0, 0.0, 1.0],   // INDEX_PIP
            7: [1.0, 0.0, 1.0],   // INDEX_DIP
            8: [1.0, 0.0, 1.0],   // INDEX_TIP
            // Palm (red)
            9: [1.0, 0.0, 0.0],   // MIDDLE_MCP
            // Middle (yellow)
            10: [1.0, 1.0, 0.0],  // MIDDLE_PIP
            11: [1.0, 1.0, 0.0],  // MIDDLE_DIP
            12: [1.0, 1.0, 0.0],  // MIDDLE_TIP
            // Palm (red)
            13: [1.0, 0.0, 0.0],  // RING_MCP
            // Ring (green)
            14: [0.0, 1.0, 0.0],  // RING_PIP
            15: [0.0, 1.0, 0.0],  // RING_DIP
            16: [0.0, 1.0, 0.0],  // RING_TIP
            // Palm (red)
            17: [1.0, 0.0, 0.0],  // PINKY_MCP
            // Pinky (blue)
            18: [0.0, 0.5, 1.0],  // PINKY_PIP
            19: [0.0, 0.5, 1.0],  // PINKY_DIP
            20: [0.0, 0.5, 1.0],  // PINKY_TIP
        };
        return colors[index] || [1.0, 1.0, 1.0];
    }
    
    get landmarkColors() {
        const colors = {};
        for (let i = 0; i <= 20; i++) {
            colors[i] = this.getLandmarkColor(i);
        }
        return colors;
    }
    
    async createCanvas(width, height) {
        // Create the regular canvas (right side)
        super.createCanvas(width, height);
        
        // Create container for Three.js
        this.leftContainer = document.createElement('div');
        this.leftContainer.id = `three-container-${this.name}`;
        this.leftContainer.className = 'three-container';
        this.leftContainer.style.position = 'absolute';
        this.leftContainer.style.top = '0';
        this.leftContainer.style.width = `${width / 2}px`;
        this.leftContainer.style.height = `${height}px`;
        this.leftContainer.style.backgroundColor = '#000';
        this.leftContainer.style.display = 'none';
        this.updateLayout();  // Set initial position
        
        document.getElementById('canvas-container').appendChild(this.leftContainer);
        
        // Load Three.js lazily if not already loaded
        if (!THREE) {
            console.log('Loading Three.js...');
            THREE = await import('https://unpkg.com/three@0.161.0/build/three.module.js');
            console.log('Three.js loaded');
        }
        
        // Initialize Three.js
        this.initThree(width / 2, height);
        
        // Add mouse event listeners
        this.setupMouseControls();
    }
    
    initThree(width, height) {
        // Scene
        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0x000000);
        
        // Camera - adjusted for normalized world coordinates
        this.camera = new THREE.PerspectiveCamera(60, width / height, 0.01, 10);
        this.camera.position.z = 2.5; // Further back to see entire hand
        
        // Renderer
        this.renderer = new THREE.WebGLRenderer({ antialias: true });
        this.renderer.setSize(width, height);
        this.leftContainer.appendChild(this.renderer.domElement);
        
        // Lighting
        const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
        this.scene.add(ambientLight);
        
        const directionalLight = new THREE.DirectionalLight(0xffffff, 0.4);
        directionalLight.position.set(0, 1, 1);
        this.scene.add(directionalLight);
        
        // Create spheres for landmarks instead of points
        this.landmarkSpheres = [];
        const sphereGeometry = new THREE.SphereGeometry(0.03, 16, 16); // Doubled radius for spheres
        
        for (let i = 0; i < 21; i++) {
            const color = this.getLandmarkColor(i);
            const material = new THREE.MeshPhongMaterial({
                color: new THREE.Color(color[0], color[1], color[2]),
                emissive: new THREE.Color(color[0] * 0.3, color[1] * 0.3, color[2] * 0.3),
                shininess: 100
            });
            const sphere = new THREE.Mesh(sphereGeometry, material);
            this.scene.add(sphere);
            this.landmarkSpheres.push(sphere);
        }
        
        // Create cylinders for connections with different thicknesses
        this.connectionCylinders = [];
        
        // Define connection groups with their colors and thicknesses
        const connectionGroups = [
            // Palm connections (gray, thicker)
            { connections: [[0, 1], [1, 2], [2, 5], [5, 9], [9, 13], [13, 17], [0, 17]], color: 0x808080, radius: 0.012 },
            // Thumb (cream white)
            { connections: [[2, 3], [3, 4]], color: 0xFFFDD0, radius: 0.008 },
            // Index (magenta)
            { connections: [[5, 6], [6, 7], [7, 8]], color: 0xFF00FF, radius: 0.008 },
            // Middle (yellow)
            { connections: [[9, 10], [10, 11], [11, 12]], color: 0xFFFF00, radius: 0.008 },
            // Ring (green)
            { connections: [[13, 14], [14, 15], [15, 16]], color: 0x00FF00, radius: 0.008 },
            // Pinky (blue)
            { connections: [[17, 18], [18, 19], [19, 20]], color: 0x0080FF, radius: 0.008 }
        ];
        
        // Create cylinders for each connection
        for (const group of connectionGroups) {
            const material = new THREE.MeshPhongMaterial({
                color: group.color,
                emissive: new THREE.Color(group.color).multiplyScalar(0.3),
                shininess: 100
            });
            
            for (const connection of group.connections) {
                const cylinderGeometry = new THREE.CylinderGeometry(group.radius, group.radius, 1, 8);
                const cylinder = new THREE.Mesh(cylinderGeometry, material);
                this.scene.add(cylinder);
                this.connectionCylinders.push({ 
                    mesh: cylinder, 
                    connection: connection,
                    baseRadius: group.radius
                });
            }
        }
    }
    
    setupMouseControls() {
        if (!this.renderer) return;
        const canvas = this.renderer.domElement;
        
        // Double-click to reset rotation
        canvas.addEventListener('dblclick', () => {
            this.rotationX = 0;
            this.rotationY = 0;
        });
        
        canvas.addEventListener('mousedown', (e) => {
            this.isMouseDown = true;
            this.mouseX = e.clientX;
            this.mouseY = e.clientY;
        });
        
        canvas.addEventListener('mousemove', (e) => {
            if (!this.isMouseDown) return;
            
            const deltaX = e.clientX - this.mouseX;
            const deltaY = e.clientY - this.mouseY;
            
            this.rotationY += deltaX * 0.01;
            this.rotationX += deltaY * 0.01;
            
            this.mouseX = e.clientX;
            this.mouseY = e.clientY;
        });
        
        canvas.addEventListener('mouseup', () => {
            this.isMouseDown = false;
        });
        
        canvas.addEventListener('mouseleave', () => {
            this.isMouseDown = false;
        });
        
        // Touch support for mobile
        canvas.addEventListener('touchstart', (e) => {
            if (e.touches.length === 1) {
                this.isMouseDown = true;
                this.mouseX = e.touches[0].clientX;
                this.mouseY = e.touches[0].clientY;
            }
        });
        
        canvas.addEventListener('touchmove', (e) => {
            if (!this.isMouseDown || e.touches.length !== 1) return;
            
            const deltaX = e.touches[0].clientX - this.mouseX;
            const deltaY = e.touches[0].clientY - this.mouseY;
            
            this.rotationY += deltaX * 0.01;
            this.rotationX += deltaY * 0.01;
            
            this.mouseX = e.touches[0].clientX;
            this.mouseY = e.touches[0].clientY;
        });
        
        canvas.addEventListener('touchend', () => {
            this.isMouseDown = false;
        });
    }
    
    resize(width, height) {
        super.resize(width, height);
        
        if (this.leftContainer) {
            this.leftContainer.style.width = `${width / 2}px`;
            this.leftContainer.style.height = `${height}px`;
        }
        
        if (this.camera && this.renderer) {
            this.camera.aspect = (width / 2) / height;
            this.camera.updateProjectionMatrix();
            this.renderer.setSize(width / 2, height);
        }
    }
    
    activate() {
        super.activate();
        if (this.leftContainer) {
            this.leftContainer.style.display = 'block';
        }
    }
    
    deactivate() {
        super.deactivate();
        if (this.leftContainer) {
            this.leftContainer.style.display = 'none';
        }
    }
    
    update(handsData, gestures) {
        // Call parent update and check if we should continue
        if (!super.update(handsData, gestures)) {
            return;
        }
        
        // Check for air tap on the 3D viewer area to swap hands
        this.checkForSwapTap();
        
        // Update 3D visualization
        this.update3D();
    }
    
    checkForSwapTap() {
        if (!this.handsData || !this.handsData.airTapData) return;

        // Define info panel bounds
        const panelX = this.displayHand === 'right' ? this.width - 220 : 10;
        const panelY = 10;
        const panelWidth = 220;
        const panelHeight = 110;
        
        // Check air taps from both hands
        for (const [handedness, tapData] of Object.entries(this.handsData.airTapData)) {
            if (this.isGestureJustAdded('AIR_TAP', handedness.toLowerCase())) {
                // Get tap position
                const scaled = this.scalePoint(tapData.tapPosition);

                // Check if tap is in the info panel area
                const tapInPanel = scaled.x >= panelX && 
                                  scaled.x <= panelX + panelWidth &&
                                  scaled.y >= panelY && 
                                  scaled.y <= panelY + panelHeight;
                
                if (tapInPanel) {
                    // Swap hands and reset rotation
                    this.displayHand = this.displayHand === 'right' ? 'left' : 'right';
                    this.rotationX = 0;
                    this.rotationY = 0;
                    this.updateLayout();
                    break; // Exit loop after handling the tap
                }
            }
        }
    }
    
    update3D() {
        if (!this.handsData || !this.landmarkSpheres || !this.connectionCylinders) return;
        
        const hand = this.displayHand === 'right' ? this.handsData.right : this.handsData.left;
        
        if (!hand || !hand.all_landmarks) {
            // Hide spheres and cylinders if no hand
            for (const sphere of this.landmarkSpheres) {
                sphere.visible = false;
            }
            for (const cylinderData of this.connectionCylinders) {
                cylinderData.mesh.visible = false;
            }
            // Clear the 3D scene when no hand is visible
            if (this.renderer) {
                this.renderer.render(this.scene, this.camera);
            }
            return;
        }
        
        // Show all meshes
        for (const sphere of this.landmarkSpheres) {
            sphere.visible = true;
        }
        for (const cylinderData of this.connectionCylinders) {
            cylinderData.mesh.visible = true;
        }
        
        // Simple fixed scale for world coordinates
        const scale = 10.0; // Scale up to make visible
        
        // Create a group to apply rotation to everything
        const rotationMatrix = new THREE.Matrix4();
        rotationMatrix.makeRotationFromEuler(new THREE.Euler(this.rotationX, this.rotationY, 0));
        
        // Update sphere positions
        for (let i = 0; i < hand.all_landmarks.length && i < this.landmarkSpheres.length; i++) {
            const landmark = hand.all_landmarks[i];
            const sphere = this.landmarkSpheres[i];
            if (landmark?.world) {
                // Set base position
                const basePos = new THREE.Vector3(
                    landmark.world.x * scale,
                    -landmark.world.y * scale,
                    landmark.world.z * scale
                );
                
                // Apply rotation
                basePos.applyMatrix4(rotationMatrix);
                sphere.position.copy(basePos);
            }
        }
        
        // Update cylinder positions and orientations
        for (const cylinderData of this.connectionCylinders) {
            const [startIdx, endIdx] = cylinderData.connection;
            const startLandmark = hand.all_landmarks[startIdx];
            const endLandmark = hand.all_landmarks[endIdx];
            
            if (startLandmark?.world && endLandmark?.world) {
                // Create start and end vectors
                const start = new THREE.Vector3(
                    startLandmark.world.x * scale,
                    -startLandmark.world.y * scale,
                    startLandmark.world.z * scale
                );
                const end = new THREE.Vector3(
                    endLandmark.world.x * scale,
                    -endLandmark.world.y * scale,
                    endLandmark.world.z * scale
                );
                
                // Apply rotation to both points
                start.applyMatrix4(rotationMatrix);
                end.applyMatrix4(rotationMatrix);
                
                // Position cylinder at midpoint
                cylinderData.mesh.position.copy(start.clone().add(end).multiplyScalar(0.5));
                
                // Calculate length and set scale
                const length = start.distanceTo(end);
                cylinderData.mesh.scale.set(1, length, 1);
                
                // Orient cylinder to point from start to end
                const direction = end.clone().sub(start).normalize();
                const up = new THREE.Vector3(0, 1, 0);
                const quaternion = new THREE.Quaternion().setFromUnitVectors(up, direction);
                cylinderData.mesh.quaternion.copy(quaternion);
            }
        }
        
        // Render the scene
        this.renderer.render(this.scene, this.camera);
    }
    
    draw() {
        if (!this.ctx || !this.isActive) return;
        
        // Clear canvas
        DP.clearCanvas(this.ctx);
        
        if (!this.handsData || !this.scale) return;
        
        // Clip to the appropriate half based on which hand is displayed
        this.ctx.save();
        this.ctx.beginPath();
        if (this.displayHand === 'right') {
            // Camera view on right when showing right hand
            this.ctx.rect(this.width / 2, 0, this.width / 2, this.height);
        } else {
            // Camera view on left when showing left hand
            this.ctx.rect(0, 0, this.width / 2, this.height);
        }
        this.ctx.clip();
        
        // Draw hands (camera view)
        this.drawHand(this.handsData.left, 'LEFT');
        this.drawHand(this.handsData.right, 'RIGHT');
        
        this.ctx.restore();
        
        // Draw info panel (no clipping)
        this.drawInfoPanel();
    }
    
    drawHand(hand, label) {
        if (!hand || !hand.all_landmarks) return;
        
        // Convert landmark colors from RGB array to hex strings
        const colorToHex = (color) => {
            const r = Math.floor(color[0] * 255);
            const g = Math.floor(color[1] * 255);
            const b = Math.floor(color[2] * 255);
            return `#${r.toString(16).padStart(2, '0')}${g.toString(16).padStart(2, '0')}${b.toString(16).padStart(2, '0')}`;
        };
        
        // Define connection groups for canvas drawing
        const connectionGroups = [
            // Palm connections (gray, thicker)
            { connections: [[0, 1], [1, 2], [2, 5], [5, 9], [9, 13], [13, 17], [17, 0]], color: '#808080', linewidth: 3 },
            // Thumb (cream white)
            { connections: [[2, 3], [3, 4]], color: '#FFFDD0', linewidth: 2 },
            // Index (magenta)
            { connections: [[5, 6], [6, 7], [7, 8]], color: '#FF00FF', linewidth: 2 },
            // Middle (yellow)
            { connections: [[9, 10], [10, 11], [11, 12]], color: '#FFFF00', linewidth: 2 },
            // Ring (green)
            { connections: [[13, 14], [14, 15], [15, 16]], color: '#00FF00', linewidth: 2 },
            // Pinky (blue)
            { connections: [[17, 18], [18, 19], [19, 20]], color: '#0080FF', linewidth: 2 }
        ];
        
        // Draw connections first (so points are on top)
        for (const group of connectionGroups) {
            this.ctx.strokeStyle = group.color;
            this.ctx.lineWidth = group.linewidth;
            for (const [start, end] of group.connections) {
                const startLandmark = hand.all_landmarks[start];
                const endLandmark = hand.all_landmarks[end];
                if (startLandmark && endLandmark) {
                    const startScaled = this.scalePoint(startLandmark);
                    const endScaled = this.scalePoint(endLandmark);
                    this.ctx.beginPath();
                    this.ctx.moveTo(startScaled.x, startScaled.y);
                    this.ctx.lineTo(endScaled.x, endScaled.y);
                    this.ctx.stroke();
                }
            }
        }
        
        // Draw landmarks with their colors
        for (let i = 0; i < hand.all_landmarks.length; i++) {
            const landmark = hand.all_landmarks[i];
            const color = this.landmarkColors[i];
            this.ctx.fillStyle = colorToHex(color);
            
            const scaled = this.scalePoint(landmark);
            this.ctx.beginPath();
            this.ctx.arc(scaled.x, scaled.y, 6, 0, 2 * Math.PI); // Doubled size from 3 to 6
            this.ctx.fill();
        }
    }
    
    updateLayout() {
        if (!this.leftContainer) return;
        
        // Position Three.js container based on which hand is displayed
        if (this.displayHand === 'right') {
            // Three.js on left when showing right hand
            this.leftContainer.style.left = '0';
            this.leftContainer.style.right = 'auto';
        } else {
            // Three.js on right when showing left hand
            this.leftContainer.style.left = 'auto';
            this.leftContainer.style.right = '0';
        }
    }
    
    drawInfoPanel() {
        // Position info panel based on which hand is displayed
        const panelX = this.displayHand === 'right' ? this.width - 230 : 10;
        
        // Draw info panel (extended height for swap instruction)
        DP.drawInfoPanel(this.ctx, panelX, 10, 220, 110);
        
        // Position text based on panel position
        const textX = this.displayHand === 'right' ? this.width - 220 : 20;
        
        // Draw title
        DP.drawText(this.ctx, '3D VIEWER', textX, 25, 'bold 12px', DrawingStyles.colors.textTitle);
        
        // Draw instructions
        const side3D = this.displayHand === 'right' ? 'Left' : 'Right';
        const sideCamera = this.displayHand === 'right' ? 'Right' : 'Left';
        DP.drawText(this.ctx, `${side3D}: 3D visualization`, textX, 45, '11px');
        DP.drawText(this.ctx, `${sideCamera}: Camera view`, textX, 60, '11px');
        DP.drawText(this.ctx, 'Drag: rotate | Dbl-click: reset', textX, 75, '11px');
        DP.drawText(this.ctx, `Showing: ${this.displayHand} hand`, textX, 90, '11px');
        DP.drawText(this.ctx, 'Air tap here to swap hands', textX, 110, 'bold 11px');
    }
    
    scalePoint(point) {
        return {
            x: point.x * this.scale.x,
            y: point.y * this.scale.y
        };
    }
}

// Self-register this application
BaseApplication.register(ThreeDViewerApplication);
