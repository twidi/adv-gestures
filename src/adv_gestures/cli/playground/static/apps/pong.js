import { BaseApplication } from './_base.js';
import { DP, DrawingStyles } from '../drawing-primitives.js';

export class PongApplication extends BaseApplication {
    constructor(applicationManager) {
        super('pong', applicationManager);
        // FontAwesome racquet icon SVG path
        this.iconSvgPath = "M436.7 56.2C475 57.8 511.8 72 539.9 100.1L545.3 105.8C571.6 135.1 584 172.6 584.1 211L583.9 220.7C581.3 269.6 559.1 319.8 519.6 359.3C451.6 427.3 348.4 445.5 278.7 395.3L208.6 465.4C217.5 482.1 215.9 503 203.5 518.2L200.2 521.9L146.1 576C128.5 593.6 100.7 594.6 81.9 579.3L78.2 576L64.1 561.9C45.4 543.2 45.4 512.8 64.1 494L118.2 439.9L121.9 436.6C137 424.3 157.9 422.5 174.7 431.4L244.8 361.3C227 336.4 217.7 306.9 216.4 276.5L216.2 268.8C216.2 216.9 238.6 162.6 280.8 120.4C323 78.2 377.1 56 429 56L436.7 56.2zM98 528L112.1 542.1L166.1 488L152.1 473.9L98 528zM429 104C390.7 104 348.4 120.7 314.6 154.5C280.8 188.3 264 230.7 264.1 269L264.2 274.6C265.4 302.4 275.5 327.4 294.1 346C338.5 390.4 424 387.1 485.5 325.5C517.2 293.8 533.9 254.6 535.8 218.3L536 211.1C536 183 527 157.4 509.6 137.9L506 134.1C487.4 115.5 462.4 105.4 434.6 104.2L429 104z";
        this.showPointers = false;

        // Game state
        this.ball = {
            x: 0.5,  // Normalized position (0-1)
            y: 0.5,
            vx: 0.008,  // Velocity per frame
            vy: 0.004,
            size: 15,  // pixels
        };
        
        this.leftPaddle = {
            x: 0.25,  // Normalized X position (0-0.5 for left half)
            y: 0.5,  // Normalized Y position (0-1)
            height: 0.2,  // Normalized height
            width: 20,  // pixels
            speed: 0.005,  // Speed multiplier
        };
        
        this.rightPaddle = {
            x: 0.75,  // Normalized X position (0.5-1 for right half)
            y: 0.5,  // Normalized Y position (0-1)  
            height: 0.2,  // Normalized height
            width: 20,  // pixels
            speed: 0.005,  // Speed multiplier
        };
        
        this.scores = { left: 0, right: 0 };
        this.MAX_SCORE = 10;
        this.gameOver = false;
        this.winner = null; // 'left' or 'right'
        this.countdownStartTime = null;
        this.COUNTDOWN_DURATION = 5000; // 5 seconds in milliseconds
        
        // Hand detection parameters
        this.ANGLE_MARGIN = 20; // degrees margin for detecting upward pointing hands
        this.MIN_STRAIGHT_FINGERS = 3; // Minimum straight fingers (excluding thumb)
        
        // Game parameters  
        this.PADDLE_SPEED = 0.003;
        this.BALL_SPEED_INCREMENT = 0.0008;
        
        // Visual parameters
        this.NET_DASH_HEIGHT = 20;
        this.NET_GAP_HEIGHT = 10;
        
        // Game area margins (in pixels)
        this.MARGIN = 80; // Margin around the play area
        this.gameArea = {
            left: this.MARGIN,
            top: this.MARGIN,
            right: 0, // Will be set in resize()
            bottom: 0, // Will be set in resize()
            width: 0, // Will be set in resize()
            height: 0 // Will be set in resize()
        };
    }

    activate() {
        super.activate();
        // Initialize game area if not already done
        if (this.gameArea.width === 0) {
            this.resize(this.width, this.height);
        }
        // Reset everything to fresh state
        this.scores.left = 0;
        this.scores.right = 0;
        this.gameOver = false;
        this.winner = null;
        this.leftPaddle.x = 0.25;
        this.leftPaddle.y = 0.5;
        this.rightPaddle.x = 0.75;
        this.rightPaddle.y = 0.5;
        this.resetBall();
        this.startCountdown();
    }

    resize(width, height) {
        super.resize(width, height);
        // Update game area dimensions
        this.gameArea.left = this.MARGIN;
        this.gameArea.top = this.MARGIN;
        this.gameArea.right = width - this.MARGIN;
        this.gameArea.bottom = height - this.MARGIN;
        this.gameArea.width = this.gameArea.right - this.gameArea.left;
        this.gameArea.height = this.gameArea.bottom - this.gameArea.top;
    }

    resetBall() {
        this.ball.x = 0.5;
        this.ball.y = 0.5;
        // Random direction
        const angle = (Math.random() - 0.5) * Math.PI / 3; // ±30° from horizontal
        const speed = 0.008;
        this.ball.vx = Math.cos(angle) * speed * (Math.random() > 0.5 ? 1 : -1);
        this.ball.vy = Math.sin(angle) * speed;
    }

    resetGame() {
        this.scores.left = 0;
        this.scores.right = 0;
        this.gameOver = false;
        this.winner = null;
        this.resetBall();
        this.startCountdown();
    }

    startCountdown() {
        this.countdownStartTime = Date.now();
    }

    isInCountdown() {
        if (!this.countdownStartTime) return false;
        return Date.now() - this.countdownStartTime < this.COUNTDOWN_DURATION;
    }

    getCountdownRemaining() {
        if (!this.countdownStartTime) return 0;
        const elapsed = Date.now() - this.countdownStartTime;
        const remaining = Math.max(0, this.COUNTDOWN_DURATION - elapsed);
        return Math.ceil(remaining / 1000); // Return seconds remaining
    }

    getNormalizedPositionForVerticalHand(hand) {
        if (!hand || hand.main_direction_angle === undefined) { 
            return { x: NaN, y: NaN }; 
        }
        
        const angle = hand.main_direction_angle;
        
        // Check if pointing up (around 90°)
        if (Math.abs(angle - 90) > this.ANGLE_MARGIN) { 
            return { x: NaN, y: NaN }; 
        }
        
        // Check that at least 3 fingers (excluding thumb) are straight
        if (!hand.fingers) { 
            return { x: NaN, y: NaN }; 
        }
        
        const straightFingers = [
            hand.fingers.INDEX?.is_nearly_straight_or_straight,
            hand.fingers.MIDDLE?.is_nearly_straight_or_straight,
            hand.fingers.RING?.is_nearly_straight_or_straight,
            hand.fingers.PINKY?.is_nearly_straight_or_straight
        ].filter(isStraight => isStraight === true).length;
        
        if (straightFingers < this.MIN_STRAIGHT_FINGERS) { 
            return { x: NaN, y: NaN }; 
        }
        
        // Use palm centroid position for paddle control
        if (!hand.palm || !hand.palm.centroid) { 
            return { x: NaN, y: NaN }; 
        }

        const scaledCentroid = this.scalePoint(hand.palm.centroid);
        
        // Normalize positions (0-1)
        return {
            x: scaledCentroid.x / this.width,
            y: scaledCentroid.y / this.height
        };
    }

    update(handsData, gestures) {
        if (!super.update(handsData, gestures)) {
            return;
        }
        
        if (!this.handsData || !this.handsData.hands) return;

        // Check for CROSSED_FISTS to reset game
        if (this.isGestureJustAdded('CROSSED_FISTS', 'both')) {
            this.resetGame();
            return;
        }

        // Don't update paddles if game is over or in countdown
        if (this.gameOver || this.isInCountdown()) return;

        // Update paddles based on hand positions
        if (this.handsData.left) {
            const normalizedPos = this.getNormalizedPositionForVerticalHand(this.handsData.left);
            if (!isNaN(normalizedPos.x) && !isNaN(normalizedPos.y)) {
                // Convert screen position to game area position
                const screenX = normalizedPos.x * this.width;
                const screenY = normalizedPos.y * this.height;
                
                // Check if hand is within game area
                if (screenX >= this.gameArea.left && screenX <= this.gameArea.right &&
                    screenY >= this.gameArea.top && screenY <= this.gameArea.bottom) {
                    // Normalize to game area (0-1 within game area)
                    const gameX = (screenX - this.gameArea.left) / this.gameArea.width;
                    const gameY = (screenY - this.gameArea.top) / this.gameArea.height;
                    
                    // Direct paddle position - Y (clamped to keep paddle fully in game area)
                    this.leftPaddle.y = Math.max(this.leftPaddle.height / 2, 
                                        Math.min(1 - this.leftPaddle.height / 2, gameY));
                    
                    // Direct paddle position - X (limited to left half of game area: 0-0.5)
                    this.leftPaddle.x = Math.max(0.05, Math.min(0.45, gameX));
                }
            }
        }
        
        if (this.handsData.right) {
            const normalizedPos = this.getNormalizedPositionForVerticalHand(this.handsData.right);
            if (!isNaN(normalizedPos.x) && !isNaN(normalizedPos.y)) {
                // Convert screen position to game area position
                const screenX = normalizedPos.x * this.width;
                const screenY = normalizedPos.y * this.height;
                
                // Check if hand is within game area
                if (screenX >= this.gameArea.left && screenX <= this.gameArea.right &&
                    screenY >= this.gameArea.top && screenY <= this.gameArea.bottom) {
                    // Normalize to game area (0-1 within game area)
                    const gameX = (screenX - this.gameArea.left) / this.gameArea.width;
                    const gameY = (screenY - this.gameArea.top) / this.gameArea.height;
                    
                    // Direct paddle position - Y (clamped to keep paddle fully in game area)
                    this.rightPaddle.y = Math.max(this.rightPaddle.height / 2, 
                                         Math.min(1 - this.rightPaddle.height / 2, gameY));
                    
                    // Direct paddle position - X (limited to right half of game area: 0.5-1)
                    this.rightPaddle.x = Math.max(0.55, Math.min(0.95, gameX));
                }
            }
        }

        // Update ball physics only if game is not over and not in countdown
        if (!this.gameOver && !this.isInCountdown()) {
            this.updateBall();
        }
    }

    updateBall() {
        // Move ball
        this.ball.x += this.ball.vx;
        this.ball.y += this.ball.vy;

        // Ball collision with top and bottom of game area
        if (this.ball.y <= 0 || this.ball.y >= 1) {
            this.ball.vy = -this.ball.vy;
            this.ball.y = Math.max(0, Math.min(1, this.ball.y));
        }

        // Convert ball position to pixel coordinates within game area
        const ballPixelX = this.gameArea.left + this.ball.x * this.gameArea.width;
        const ballPixelY = this.gameArea.top + this.ball.y * this.gameArea.height;
        const ballRadius = this.ball.size / 2;

        // Left paddle collision
        if (this.ball.vx < 0) {
            const leftPaddlePixelX = this.gameArea.left + this.leftPaddle.x * this.gameArea.width;
            const paddleLeft = leftPaddlePixelX - this.leftPaddle.width / 2 - 5; // Add 5px tolerance
            const paddleRight = leftPaddlePixelX + this.leftPaddle.width / 2 + 5; // Add 5px tolerance
            
            // Check if ball overlaps with paddle in X axis
            if (ballPixelX - ballRadius <= paddleRight && ballPixelX + ballRadius >= paddleLeft) {
                const paddleTop = this.gameArea.top + (this.leftPaddle.y - this.leftPaddle.height / 2) * this.gameArea.height - 5; // Add tolerance
                const paddleBottom = this.gameArea.top + (this.leftPaddle.y + this.leftPaddle.height / 2) * this.gameArea.height + 5; // Add tolerance
                
                // Check Y overlap
                if (ballPixelY >= paddleTop && ballPixelY <= paddleBottom) {
                    // Position ball just outside paddle
                    this.ball.x = (leftPaddlePixelX + this.leftPaddle.width / 2 + ballRadius + 1 - this.gameArea.left) / this.gameArea.width;
                    this.ball.vx = Math.abs(this.ball.vx) + this.BALL_SPEED_INCREMENT;
                    // Add spin based on where ball hits paddle (adjust for tolerance)
                    const actualPaddleTop = this.gameArea.top + (this.leftPaddle.y - this.leftPaddle.height / 2) * this.gameArea.height;
                    const actualPaddleBottom = this.gameArea.top + (this.leftPaddle.y + this.leftPaddle.height / 2) * this.gameArea.height;
                    const clampedY = Math.max(actualPaddleTop, Math.min(actualPaddleBottom, ballPixelY));
                    const hitPosition = (clampedY - actualPaddleTop) / (actualPaddleBottom - actualPaddleTop) - 0.5;
                    this.ball.vy += hitPosition * 0.008;
                }
            }
        }

        // Right paddle collision
        if (this.ball.vx > 0) {
            const rightPaddlePixelX = this.gameArea.left + this.rightPaddle.x * this.gameArea.width;
            const paddleLeft = rightPaddlePixelX - this.rightPaddle.width / 2 - 5; // Add 5px tolerance
            const paddleRight = rightPaddlePixelX + this.rightPaddle.width / 2 + 5; // Add 5px tolerance
            
            // Check if ball overlaps with paddle in X axis
            if (ballPixelX + ballRadius >= paddleLeft && ballPixelX - ballRadius <= paddleRight) {
                const paddleTop = this.gameArea.top + (this.rightPaddle.y - this.rightPaddle.height / 2) * this.gameArea.height - 5; // Add tolerance
                const paddleBottom = this.gameArea.top + (this.rightPaddle.y + this.rightPaddle.height / 2) * this.gameArea.height + 5; // Add tolerance
                
                // Check Y overlap
                if (ballPixelY >= paddleTop && ballPixelY <= paddleBottom) {
                    // Position ball just outside paddle
                    this.ball.x = (rightPaddlePixelX - this.rightPaddle.width / 2 - ballRadius - 1 - this.gameArea.left) / this.gameArea.width;
                    this.ball.vx = -(Math.abs(this.ball.vx) + this.BALL_SPEED_INCREMENT);
                    // Add spin based on where ball hits paddle (adjust for tolerance)
                    const actualPaddleTop = this.gameArea.top + (this.rightPaddle.y - this.rightPaddle.height / 2) * this.gameArea.height;
                    const actualPaddleBottom = this.gameArea.top + (this.rightPaddle.y + this.rightPaddle.height / 2) * this.gameArea.height;
                    const clampedY = Math.max(actualPaddleTop, Math.min(actualPaddleBottom, ballPixelY));
                    const hitPosition = (clampedY - actualPaddleTop) / (actualPaddleBottom - actualPaddleTop) - 0.5;
                    this.ball.vy += hitPosition * 0.008;
                }
            }
        }

        // Score when ball goes off screen
        if (this.ball.x < -0.05) {
            this.scores.right++;
            if (this.scores.right >= this.MAX_SCORE) {
                this.gameOver = true;
                this.winner = 'right';
            } else {
                this.resetBall();
            }
        } else if (this.ball.x > 1.05) {
            this.scores.left++;
            if (this.scores.left >= this.MAX_SCORE) {
                this.gameOver = true;
                this.winner = 'left';
            } else {
                this.resetBall();
            }
        }

        // Limit ball vertical speed
        this.ball.vy = Math.max(-0.012, Math.min(0.012, this.ball.vy));
    }

    draw() {
        if (!this.ctx || !this.isActive) return;
        
        // Clear canvas
        DP.clearCanvas(this.ctx);
        
        // Draw victory/defeat backgrounds if game is over
        if (this.gameOver) {
            this.drawGameOverBackground();
        }
        
        // Draw game area border
        this.drawGameArea();
        
        // Draw game elements
        this.drawCourt();
        this.drawPaddles();
        if (!this.gameOver && !this.isInCountdown()) {
            this.drawBall();
        }
        this.drawScore();
        this.drawHandIndicators();
        
        // Draw countdown if active
        if (this.isInCountdown()) {
            this.drawCountdown();
        }
        
        // Draw game over text
        if (this.gameOver) {
            this.drawGameOverText();
        }
    }

    drawGameArea() {
        const ctx = this.ctx;
        
        // Draw game area border
        ctx.save();
        ctx.strokeStyle = DrawingStyles.colors.accent;
        ctx.lineWidth = 2;
        ctx.strokeRect(this.gameArea.left, this.gameArea.top, this.gameArea.width, this.gameArea.height);
        ctx.restore();
    }
    
    drawCourt() {
        const ctx = this.ctx;
        
        // Draw center net (only within game area)
        ctx.save();
        ctx.strokeStyle = DrawingStyles.colors.accent;
        ctx.lineWidth = 2;
        ctx.setLineDash([this.NET_DASH_HEIGHT, this.NET_GAP_HEIGHT]);
        ctx.beginPath();
        ctx.moveTo(this.gameArea.left + this.gameArea.width / 2, this.gameArea.top);
        ctx.lineTo(this.gameArea.left + this.gameArea.width / 2, this.gameArea.bottom);
        ctx.stroke();
        ctx.restore();
    }

    drawPaddles() {
        const ctx = this.ctx;
        
        ctx.fillStyle = DrawingStyles.colors.accent;
        
        // Left paddle (within game area)
        const leftPaddleX = this.gameArea.left + this.leftPaddle.x * this.gameArea.width - this.leftPaddle.width / 2;
        const leftPaddleY = this.gameArea.top + (this.leftPaddle.y - this.leftPaddle.height / 2) * this.gameArea.height;
        const leftPaddleHeight = this.leftPaddle.height * this.gameArea.height;
        DP.roundedRect(ctx, leftPaddleX, leftPaddleY, this.leftPaddle.width, leftPaddleHeight, 5);
        ctx.fill();
        
        // Right paddle (within game area)
        const rightPaddleX = this.gameArea.left + this.rightPaddle.x * this.gameArea.width - this.rightPaddle.width / 2;
        const rightPaddleY = this.gameArea.top + (this.rightPaddle.y - this.rightPaddle.height / 2) * this.gameArea.height;  
        const rightPaddleHeight = this.rightPaddle.height * this.gameArea.height;
        DP.roundedRect(ctx, rightPaddleX, rightPaddleY, this.rightPaddle.width, rightPaddleHeight, 5);
        ctx.fill();
    }

    drawBall() {
        const ctx = this.ctx;
        
        ctx.fillStyle = DrawingStyles.colors.accent;
        ctx.beginPath();
        ctx.arc(
            this.gameArea.left + this.ball.x * this.gameArea.width, 
            this.gameArea.top + this.ball.y * this.gameArea.height, 
            this.ball.size / 2, 
            0, 
            Math.PI * 2
        );
        ctx.fill();
    }

    drawScore() {
        const ctx = this.ctx;
        
        ctx.fillStyle = DrawingStyles.colors.accent;
        ctx.font = '48px monospace';
        ctx.textAlign = 'center';
        
        // Left score (in top margin, above game area)
        ctx.fillText(
            this.scores.left.toString(), 
            this.gameArea.left + this.gameArea.width / 4, 
            this.gameArea.top - 20
        );
        
        // Right score (in top margin, above game area)
        ctx.fillText(
            this.scores.right.toString(), 
            this.gameArea.left + (this.gameArea.width * 3) / 4, 
            this.gameArea.top - 20
        );
    }

    drawCountdown() {
        const ctx = this.ctx;
        const countdown = this.getCountdownRemaining();
        
        if (countdown <= 0) return;
        
        ctx.save();
        
        // Semi-transparent background
        ctx.fillStyle = 'rgba(0, 0, 0, 0.5)';
        ctx.fillRect(0, 0, this.width, this.height);
        
        // Countdown number
        ctx.font = '120px monospace';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillStyle = DrawingStyles.colors.accent;
        ctx.fillText(
            countdown.toString(),
            this.width / 2,
            this.height / 2
        );
        
        // "Get ready!" text
        ctx.font = '32px monospace';
        ctx.fillStyle = 'rgba(255, 255, 255, 0.8)';
        ctx.fillText(
            'Get ready!',
            this.width / 2,
            this.height / 2 + 100
        );
        
        ctx.restore();
    }

    drawGameOverBackground() {
        const ctx = this.ctx;
        
        // Only draw background for winner side with accent color
        if (this.winner === 'left') {
            // Left half winner background - accent color very transparent
            ctx.fillStyle = DrawingStyles.colors.accent + '15'; // Add low alpha hex (15 = ~8% opacity)
            ctx.fillRect(this.gameArea.left, this.gameArea.top, this.gameArea.width / 2, this.gameArea.height);
        } else if (this.winner === 'right') {
            // Right half winner background - accent color very transparent
            ctx.fillStyle = DrawingStyles.colors.accent + '15'; // Add low alpha hex (15 = ~8% opacity)
            ctx.fillRect(this.gameArea.left + this.gameArea.width / 2, this.gameArea.top, this.gameArea.width / 2, this.gameArea.height);
        }
    }

    drawGameOverText() {
        const ctx = this.ctx;
        
        ctx.save();
        
        const centerY = this.gameArea.top + this.gameArea.height / 2;
        
        // Only draw WINNER text with glow effect
        ctx.font = '64px monospace';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        
        // Determine winner position
        const winnerX = this.winner === 'left' 
            ? this.gameArea.left + this.gameArea.width / 4
            : this.gameArea.left + (this.gameArea.width * 3) / 4;
        
        // Draw glow effect
        ctx.shadowColor = DrawingStyles.colors.accent;
        ctx.shadowBlur = 20;
        ctx.shadowOffsetX = 0;
        ctx.shadowOffsetY = 0;
        
        // Draw WINNER text multiple times for stronger glow
        ctx.fillStyle = DrawingStyles.colors.accent;
        for (let i = 0; i < 3; i++) {
            ctx.fillText('WINNER!', winnerX, centerY);
        }
        
        // Reset shadow for other text
        ctx.shadowBlur = 0;
        
        // Instructions to restart
        ctx.font = '24px monospace';
        ctx.fillStyle = DrawingStyles.colors.accent;
        ctx.fillText(
            'Cross fists to play again',
            this.gameArea.left + this.gameArea.width / 2,
            centerY + 100
        );
        
        ctx.restore();
    }

    drawHandIndicators() {
        if (!this.handsData) return;
        
        const ctx = this.ctx;
        const indicatorSize = 30;
        const margin = 20;
        
        // Left hand indicator
        if (this.handsData.left) {
            const normalizedPos = this.getNormalizedPositionForVerticalHand(this.handsData.left);
            const isControlling = !isNaN(normalizedPos.x) && !isNaN(normalizedPos.y);
            
            ctx.save();
            ctx.fillStyle = isControlling ? DrawingStyles.colors.accent : 'rgba(255, 255, 255, 0.3)';
            ctx.beginPath();
            ctx.arc(margin + indicatorSize / 2, margin + indicatorSize / 2, indicatorSize / 2, 0, Math.PI * 2);
            ctx.fill();
            
            // Draw "L" in center
            ctx.fillStyle = isControlling ? '#000' : '#fff';
            ctx.font = '16px monospace';
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            ctx.fillText('L', margin + indicatorSize / 2, margin + indicatorSize / 2);
            ctx.restore();
        }
        
        // Right hand indicator  
        if (this.handsData.right) {
            const normalizedPos = this.getNormalizedPositionForVerticalHand(this.handsData.right);
            const isControlling = !isNaN(normalizedPos.x) && !isNaN(normalizedPos.y);
            
            ctx.save();
            ctx.fillStyle = isControlling ? DrawingStyles.colors.accent : 'rgba(255, 255, 255, 0.3)';
            ctx.beginPath();
            ctx.arc(this.width - margin - indicatorSize / 2, margin + indicatorSize / 2, indicatorSize / 2, 0, Math.PI * 2);
            ctx.fill();
            
            // Draw "R" in center
            ctx.fillStyle = isControlling ? '#000' : '#fff';
            ctx.font = '16px monospace';
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            ctx.fillText('R', this.width - margin - indicatorSize / 2, margin + indicatorSize / 2);
            ctx.restore();
        }
    }
}
