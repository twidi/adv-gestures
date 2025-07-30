# Gesture Recognition Playground — Product Requirements Document (PRD)
**Version:** v0.7
**Status:** Draft (agreed direction)
**Language:** English
**Owner:** Gesture SDK team

> **Design principles:** SSE‑only • zero server‑side business logic • client‑side overlays • pure JavaScript (no build, no TypeScript) • demo‑first

---
## 1. Purpose & Background
The Gesture Recognition Playground is a lightweight web environment to **preview raw outputs** from the hand/gesture recognition library and to **experiment interactively** with overlays and client‑side behavior. It is intended for **local demos** with a handful of simultaneous users and **no authentication**.

Stakeholder decisions:
- **Server**: Python (`aiohttp` + `aiortc` + `aiohttp-sse`). Its **only** job is to receive WebRTC video, run recognition, and emit **raw per-frame snapshots** via **SSE**. No server‑side derived events (`start/stop`) and no drawing.
- **Client**: **pure JavaScript** served as static files — **no bundler**, **no transpilation**, **no TypeScript**. The client renders the video, draws overlays, manages logs and UI, and performs coordinate conversion.
- **Transport**: WebRTC for video (client → server). **SSE only** for data (server → client). No WebSocket.
- **Overlays**: 100% client‑side.
- **Scale**: best effort, local demo usage; no SLOs.

---
## 2. Goals & Non‑Goals
### 2.1 Goals
- View, in real time, the **raw recognition data** alongside the live camera.
- Provide a **minimal, hackable UI** (collapsible logs, toggles, basic overlay drawing).
- Keep the server **stateless and thin**: streaming in, recognition, streaming out (SSE).
- Support **multi‑user demo** with an isolated recognizer per session UID.

### 2.2 Non‑Goals
- Authentication, roles, or user accounts.
- Server‑side gesture logic (e.g., `start/stop`), filtering/thresholds, or overlay drawing.
- Performance SLOs (e.g., latency guarantees).

---
## 3. Scope
### In scope
- Browser WebRTC capture → server decode → recognition → SSE **raw snapshot per recognized frame**.
- Static client UI: `<video>` + **canvas overlay**, collapsible logs panel (**closed by default**).
- Two persisted switches: **Mirror (ON by default)** and **Debug (OFF by default)**.
- **Data payloads come directly from the library** (see §7): the server emits `hands.to_dict()` per frame.

### Out of scope
- Authentication and hardened production deployment.
- Scaling beyond a handful of concurrent users.
- Plugin/app system (deferred).

---
## 4. System Overview
### 4.1 Data Flow (conceptual)
Client (browser)
- GetUserMedia (video) → WebRTC → **Server**
- EventSource (SSE) ← **Server** (per‑frame **snapshot**)

Server (Python)
- `aiortc` WebRTC receive → frame to recognition pipeline
- Recognition models → **`hands.to_dict()`** → emit via SSE
- No drawing, no derived events

---
## 5. Architecture & Components
### 5.1 Server (Python)
- **Frameworks:** `aiohttp` (HTTP/static), `aiortc` (WebRTC), `aiohttp-sse` (SSE).
- **Responsibilities (MUST):**
  - Accept WebRTC offers, receive video, decode frames.
  - Execute recognition and, for **each recognized frame**, serialize **exactly** `hands.to_dict()` to JSON and emit it over SSE (`event: snapshot`).
  - Optionally emit a single `event: connected` at stream open and `event: error` on failures.
  - Manage one recognizer instance **per UID**; clean up immediately when SSE connection closes.
- **Non‑Responsibilities (MUST NOT):**
  - Produce derived events (`start/stop`), thresholding, or any business logic.
  - Draw overlays or transform the outgoing JSON beyond `hands.to_dict()`.

### 5.2 Client (Pure JavaScript — no build)
- **Files:** `index.html`, `main.js` (optional `styles.css`). **No build step**; served as static files.
- **Responsibilities:**
  - Create a **UID** (UUID v4); read `mirror` and `debug` from `localStorage` (keys: `adv-gestures-playground-mirror`, `adv-gestures-playground-debug`, `adv-gestures-playground-camera`).
  - **Camera access:** Request getUserMedia; if denied or unavailable, display message: "Camera access is required for hand gesture recognition."
    - **Camera selection logic:**
      - If only one camera: use it directly.
      - If multiple cameras and one is stored in `localStorage`: use it directly but show a small camera switch button.
      - If multiple cameras and stored camera not found (or no stored preference): show camera selector.
    - **Camera selector UI:**
      - Dropdown with live preview when hovering/selecting items.
      - "Select" button to confirm choice.
      - Selected camera persisted to `localStorage`.
      - Changing camera requires page reload (new UID).
    - Apply resolution constraints based on config size (default 1280) as max width/height.
  - **Initial state:** Show "Connecting..." placeholder until WebRTC connection established.
  - **Connection sequence:**
    1. Camera access/selection
    2. SSE subscribe: `GET /sse?uid=...` (establishes recognizer on server)
    3. WebRTC handshake: `POST /webrtc/offer?uid=...` with JSON `{ sdp, mirror }`
  - SSE events: handle `connected`, **`snapshot` (payload = `hands.to_dict()` JSON)**, `error`.
    - If WebRTC negotiation fails, display specific error message via SSE error event.
  - **SSE reconnection:** If connection drops, show error state; require page reload to reconnect (new UID).
  - Render the video and **draw overlays** (transparent canvas).
  - **Debug ON** → draw **white dashed** bounding boxes using fields from the JSON.
  - **Coordinate conversion:** from backend pixel space (`stream_info.width/height`, supplied by the library) to displayed video size.
  - **Mirror switch:** default **ON**, persisted; sent in the initial offer (`mirror`) so the recognizer is configured accordingly. Upload stream is **not** mirrored; **display** is mirrored via CSS (`scaleX(-1)`). Switching **reloads** the page.
  - **Logs panel:** collapsible tray, **closed by default**.

---
## 6. Interfaces
### 6.1 HTTP Endpoints
- `GET /`
  Serves static client (`index.html`, `main.js`, `styles.css`).
- `POST /webrtc/offer?uid=<UID>`
  **Request (JSON):**
  ```json
  { "sdp": "<SDP string>", "mirror": true }
  ```
  **Response (JSON):**
  ```json
  { "sdp": "<answer SDP string>" }
  ```
- `GET /sse?uid=<UID>`
  **SSE events:**
  - `connected` — once after subscription opens (optional)
  - `snapshot` — **payload is exactly the JSON returned by `hands.to_dict()`**, with **no additional wrapping**
  - `error` — error payload

- `GET /healthz`
  Health check (200 OK).

### 6.2 CLI
- **Command:** `adv-gestures playground` (new CLI command implemented in `src/adv_gestures/cli/playground/`)
- **Options:** `--host` (default `127.0.0.1`), `--port` (default **9810**), `--open` (open page in browser).
- **Behavior:** serve static files and expose the endpoints above; graceful shutdown.
  - With `--open`: wait for server to respond on `/healthz` before opening browser.

---
## 7. Data Contract
> **Single source of truth:** **`hands.to_dict()`** (Python side).
> **Server MUST NOT wrap, rename, or transform fields**; the JSON emitted in `snapshot` events is **verbatim** the return value of `hands.to_dict()` for the current frame.

**Notes for the client:**
- Expect both **pixel** and **normalized** coordinates as provided by the library.
- Use `stream_info.width/height` from the payload for display scaling.
- Any enums/strings/field names are dictated by the library's `to_dict()` implementation.
- Performance metrics (FPS, latency) are included in the JSON payload but not displayed in the UI for MVP.

---
## 8. UX Requirements
- **Visual style:** Minimalist, modern sci-fi aesthetic. Dark theme with subtle neon accents.
- **Layout:** video occupies maximum available space while maintaining aspect ratio (with black bars for unused areas); transparent **canvas overlay** on top.
- **Control area (bottom-right corner):** Subtle, discrete controls including:
  - Connection indicator (WebRTC/SSE status)
  - Logs toggle button
  - Camera switch button (if multiple cameras available)
- **Logs panel:** collapsible/expandable; **closed by default**; rows show local timestamps and levels (`info|warn|error|debug`).
  - **Most recent logs at top**; keep maximum 100 entries.
  - Timestamp format: `HH:MM:SS.mm` (local time with centiseconds).
  - Log startup events: camera selected, SSE connected, WebRTC established.
  - Log gesture changes: when gestures are added/removed for `left`, `right`, or `both` hands, log format: `[timestamp] [hand] gesture added/removed: GESTURE_NAME (active: GESTURE1, GESTURE2, ...)`
  - Client maintains sets of active gestures per hand to detect changes between frames.
- **Mirror switch:** default **ON**, persisted (`localStorage`); affects **display** only; value sent at handshake as `{ mirror }`. Changing it **reloads** the page.
- **Debug switch:** default **OFF**, persisted; when ON, draw **white dashed** hand bounding boxes using fields from the JSON payload.
- **Keyboard shortcuts:**
  - `M` - Toggle mirror (triggers page reload)
  - `D` - Toggle debug mode
- **Coordinate conversion (client):**
  ```
  scaleX = displayedVideoWidth  / stream_info.width
  scaleY = displayedVideoHeight / stream_info.height
  x_display = x_backend * scaleX
  y_display = y_backend * scaleY
  ```
  For normalized values, multiply by displayed size directly. Mirroring is applied at the container level (video + canvas).

---
## 9. Non‑Functional Requirements
- **Scale:** a few concurrent users; **best effort** performance, no SLOs.
- **Browsers:** modern (ES Modules, WebRTC, SSE, MediaDevices).
- **Security/Privacy:** no authentication; browser permission flow handles camera consent.
- **CORS:** by default, same origin for app and API.

---
## 10. Testing & Acceptance Criteria
1. **WebRTC handshake**: `POST /webrtc/offer` returns a valid answer; server receives frames.
2. **SSE stream**: `GET /sse?uid` emits `connected` then `snapshot` at recognition cadence.
3. **Data contract**: the `snapshot` payload is **exactly** `hands.to_dict()` (no extra wrapper); field names match the library.
4. **Mirror**: default ON; visual mirroring applied; changing the switch + reload re‑initializes the session with correct server mirroring.
5. **Debug**: default OFF; when ON, dashed bounding boxes are visible.
6. **Scaling**: overlay remains aligned with video when the element is resized.
7. **Multi‑user demo**: at least two browsers/UIDs receive independent snapshots.

---
## 11. Deployment & Operations
- **CLI example:** `adv-gestures playground --host 127.0.0.1 --port 9810 --open`
- **Static client files:** served from `src/adv_gestures/cli/playground/static/`
  ```
  index.html
  main.js
  styles.css  # optional
  ```
- **Cache policy (CRITICAL):** Server MUST set HTTP headers to prevent caching of static files (`Cache-Control: no-cache, no-store, must-revalidate`). Development workflow requires immediate updates on reload.
- **Graceful shutdown:** per‑UID cleanup of recognizers and tracks.

---
## 12. Risks & Mitigations
- **Browser compatibility** → Support modern evergreen browsers only.
- **CPU/Throughput** → Client may skip frames if snapshots arrive faster than rendering capability (using `requestAnimationFrame` to maintain smooth display); server may optionally cap snapshot rate (non‑normative).
- **Coordinate mismatches** → Client must always use `stream_info.width/height` from the payload; mirroring remains visual only.

---
## 13. Milestones
1. **MVP Server**: `/webrtc/offer`, `/sse`, static serving, `/healthz`; `snapshot` emits `hands.to_dict()`.
2. **MVP Client**: video + canvas overlay, collapsible logs (default closed), mirror/debug switches (`localStorage`).
3. **Library parity**: ensure `hands.to_dict()` provides all required fields (pixels + normalized, stream info).
4. **Docs**: quick README (run, CLI options, SSE usage).
