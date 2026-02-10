// YOLO Manuscript Annotator - JavaScript
// All annotation logic runs in the browser

const CLASS_NAMES = {
    1: "text",
    2: "music",
    3: "staves"
};

const CLASS_COLORS = {
    1: "#5B8FA3",  // Muted blue-gray for text
    2: "#7A9B5F",  // Olive green for music
    3: "#B8860B"   // Bronze/gold for staves (Mothra's signature color)
};

class AnnotationTool {
    constructor() {
        this.canvas = document.getElementById('annotationCanvas');
        this.ctx = this.canvas.getContext('2d');
        this.canvasWrapper = document.getElementById('canvasWrapper');
        
        this.image = null;
        this.imageName = '';
        this.annotations = [];
        this.undoStack = [];
        this.currentClass = 1;
        
        this.isDrawing = false;
        this.isPanning = false;
        this.startPos = null;
        this.currentPos = null;
        this.panStart = null;
        
        // Zoom and view controls
        this.zoomLevel = 1.0;
        this.minZoom = 0.5;
        this.maxZoom = 5.0;
        this.zoomStep = 0.25;
        this.panOffset = { x: 0, y: 0 };
        
        // Display options
        this.showLabels = true;
        this.boxOpacity = 1.0;
        
        this.setupEventListeners();
        this.updateUI();
    }
    
    setupEventListeners() {
        // Image upload
        document.getElementById('imageInput').addEventListener('change', (e) => {
            this.loadImage(e.target.files[0]);
        });
        
        // Class selection buttons
        document.querySelectorAll('.class-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const classNum = parseInt(e.target.dataset.class);
                this.selectClass(classNum);
            });
        });
        
        // Action buttons
        document.getElementById('undoBtn').addEventListener('click', () => this.undo());
        document.getElementById('clearBtn').addEventListener('click', () => this.clearAll());
        document.getElementById('downloadJsonBtn').addEventListener('click', () => this.downloadJSON());
        document.getElementById('downloadYoloBtn').addEventListener('click', () => this.downloadYOLO());
        document.getElementById('downloadBothBtn').addEventListener('click', () => this.downloadBoth());
        
        // Zoom controls
        document.getElementById('zoomInBtn').addEventListener('click', () => this.zoom(1));
        document.getElementById('zoomOutBtn').addEventListener('click', () => this.zoom(-1));
        document.getElementById('resetZoomBtn').addEventListener('click', () => this.resetZoom());
        
        // Display options
        document.getElementById('showLabelsToggle').addEventListener('change', (e) => {
            this.showLabels = e.target.checked;
            this.render();
        });
        
        document.getElementById('boxOpacitySlider').addEventListener('input', (e) => {
            this.boxOpacity = parseFloat(e.target.value);
            document.getElementById('opacityValue').textContent = Math.round(this.boxOpacity * 100) + '%';
            this.render();
        });
        
        // Canvas mouse events
        this.canvas.addEventListener('mousedown', (e) => this.onMouseDown(e));
        this.canvas.addEventListener('mousemove', (e) => this.onMouseMove(e));
        this.canvas.addEventListener('mouseup', (e) => this.onMouseUp(e));
        this.canvas.addEventListener('mouseleave', () => this.onMouseLeave());
        this.canvas.addEventListener('wheel', (e) => this.onWheel(e), { passive: false });
        
        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => this.onKeyDown(e));
        
        // Prevent context menu on canvas
        this.canvas.addEventListener('contextmenu', (e) => e.preventDefault());
        
        // Update cursor initially
        this.updateCursor();
    }
    
    loadImage(file) {
        if (!file) return;
        
        this.imageName = file.name;
        document.getElementById('imageName').textContent = file.name;
        
        const reader = new FileReader();
        reader.onload = (e) => {
            const img = new Image();
            img.onload = () => {
                this.image = img;
                this.setupCanvas();
                this.render();
            };
            img.src = e.target.result;
        };
        reader.readAsDataURL(file);
    }
    
    setupCanvas() {
        // Set canvas size to match image (at current zoom level)
        const maxWidth = 1200;
        const maxHeight = 800;
        
        let width = this.image.width;
        let height = this.image.height;
        
        // Scale down if too large
        if (width > maxWidth || height > maxHeight) {
            const ratio = Math.min(maxWidth / width, maxHeight / height);
            width = width * ratio;
            height = height * ratio;
        }
        
        // Base canvas size (before zoom)
        this.baseWidth = width;
        this.baseHeight = height;
        
        // Apply zoom
        this.canvas.width = width * this.zoomLevel;
        this.canvas.height = height * this.zoomLevel;
        
        // Store scale factors for coordinate conversion
        this.scaleX = this.image.width / width;
        this.scaleY = this.image.height / height;
        
        // Update zoom level display
        document.getElementById('zoomLevel').textContent = Math.round(this.zoomLevel * 100) + '%';
    }
    
    selectClass(classNum) {
        this.currentClass = classNum;
        
        // Update button states
        document.querySelectorAll('.class-btn').forEach(btn => {
            btn.classList.remove('active');
            if (parseInt(btn.dataset.class) === classNum) {
                btn.classList.add('active');
            }
        });
        
        document.getElementById('currentClass').textContent = CLASS_NAMES[classNum];
        this.render();
    }
    
    getMousePos(e) {
        const rect = this.canvas.getBoundingClientRect();
        
        // Canvas is already zoomed (canvas.width = baseWidth * zoomLevel)
        // So we just get the direct pixel position on the canvas
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;
        
        return { x, y };
    }
    
    
    updateCursor() {
        if (!this.canvas) return;
        
        if (this.isPanning) {
            this.canvas.style.cursor = 'grabbing';
        } else if (this.zoomLevel > 1) {
            this.canvas.style.cursor = 'grab';
        } else {
            this.canvas.style.cursor = 'crosshair';
        }
    }
    
    onWheel(e) {
        if (!this.image) return;
        
        // Ctrl+wheel or just wheel to zoom
        if (e.ctrlKey || e.metaKey) {
            e.preventDefault();
            
            // Zoom in/out based on wheel direction
            const delta = e.deltaY < 0 ? 1 : -1;
            this.zoom(delta);
        }
    }
    
    onMouseDown(e) {
        if (!this.image) return;
        
        const pos = this.getMousePos(e);
        
        // Right click or Shift+click = pan mode (when zoomed)
        if (e.button === 2 || (e.button === 0 && e.shiftKey && this.zoomLevel > 1)) {
            this.isPanning = true;
            this.panStart = { x: e.clientX, y: e.clientY };
            this.updateCursor();
            e.preventDefault();
            return;
        }
        
        // Left click = draw
        this.startPos = pos;
        this.currentPos = pos;
        this.isDrawing = true;
    }
    
    onMouseMove(e) {
        const pos = this.getMousePos(e);
        
        // Update mouse position display
        document.getElementById('mousePos').textContent = 
            `${Math.round(pos.x)}, ${Math.round(pos.y)}`;
        
        // Handle panning
        if (this.isPanning && this.panStart) {
            const dx = e.clientX - this.panStart.x;
            const dy = e.clientY - this.panStart.y;
            
            this.canvasWrapper.scrollLeft -= dx;
            this.canvasWrapper.scrollTop -= dy;
            
            this.panStart = { x: e.clientX, y: e.clientY };
            return;
        }
        
        // Handle drawing
        if (this.isDrawing) {
            this.currentPos = pos;
            this.render();
        }
    }
    
    onMouseUp(e) {
        // End panning
        if (this.isPanning) {
            this.isPanning = false;
            this.panStart = null;
            this.updateCursor();
            return;
        }
        
        // End drawing
        if (!this.isDrawing || !this.image) return;
        
        const pos = this.getMousePos(e);
        this.currentPos = pos;
        
        // Create annotation
        this.createAnnotation();
        
        // Reset drawing state
        this.isDrawing = false;
        this.startPos = null;
        this.currentPos = null;
        
        this.render();
    }
    
    onMouseLeave() {
        if (this.isPanning) {
            this.isPanning = false;
            this.panStart = null;
            this.updateCursor();
        }
        
        if (this.isDrawing) {
            // Cancel current drawing
            this.isDrawing = false;
            this.startPos = null;
            this.currentPos = null;
            this.render();
        }
    }
    
    createAnnotation() {
        if (!this.startPos || !this.currentPos) return;
        
        // startPos/currentPos are in zoomed canvas coordinates
        // Convert to base canvas (divide by zoom), then to image coords (multiply by scale)
        const x1 = Math.min(this.startPos.x, this.currentPos.x) / this.zoomLevel * this.scaleX;
        const y1 = Math.min(this.startPos.y, this.currentPos.y) / this.zoomLevel * this.scaleY;
        const x2 = Math.max(this.startPos.x, this.currentPos.x) / this.zoomLevel * this.scaleX;
        const y2 = Math.max(this.startPos.y, this.currentPos.y) / this.zoomLevel * this.scaleY;
        
        const width = x2 - x1;
        const height = y2 - y1;
        
        // Validate minimum size (5 pixels in original image space)
        if (width < 5 || height < 5) {
            console.log('Box too small (min 5x5 pixels), ignoring');
            return;
        }
        
        const annotation = {
            class_id: this.currentClass,
            class_name: CLASS_NAMES[this.currentClass],
            bbox: [
                Math.round(x1),
                Math.round(y1),
                Math.round(x2),
                Math.round(y2)
            ],
            timestamp: new Date().toISOString()
        };
        
        this.annotations.push(annotation);
        this.undoStack.push({ action: 'add', data: annotation });
        
        console.log(`Added annotation #${this.annotations.length}: ${annotation.class_name}`);
        
        this.updateUI();
    }
    
    undo() {
        if (this.undoStack.length === 0) {
            console.log('Nothing to undo');
            return;
        }
        
        const lastAction = this.undoStack.pop();
        
        if (lastAction.action === 'add') {
            // Remove the last annotation
            const index = this.annotations.indexOf(lastAction.data);
            if (index > -1) {
                this.annotations.splice(index, 1);
                console.log('Undid annotation');
            }
        }
        
        this.updateUI();
        this.render();
    }
    
    deleteAnnotation(index) {
        if (index >= 0 && index < this.annotations.length) {
            const deleted = this.annotations.splice(index, 1)[0];
            console.log(`Deleted annotation: ${deleted.class_name}`);
            this.updateUI();
            this.render();
        }
    }
    
    clearAll() {
        if (this.annotations.length === 0) return;
        
        if (confirm(`Delete all ${this.annotations.length} annotations?`)) {
            this.annotations = [];
            this.undoStack = [];
            this.updateUI();
            this.render();
        }
    }
    
    onKeyDown(e) {
        // Class selection (1-3)
        if (e.key >= '1' && e.key <= '3') {
            this.selectClass(parseInt(e.key));
        }
        
        // Undo (u or Ctrl+Z)
        else if (e.key === 'u' || e.key === 'U' || (e.ctrlKey && e.key === 'z')) {
            e.preventDefault();
            this.undo();
        }
        
        // Cancel current drawing (ESC)
        else if (e.key === 'Escape') {
            if (this.isDrawing) {
                this.isDrawing = false;
                this.startPos = null;
                this.currentPos = null;
                this.render();
            }
        }
        
        // Zoom in (+/=)
        else if (e.key === '+' || e.key === '=') {
            e.preventDefault();
            this.zoom(1);
        }
        
        // Zoom out (-)
        else if (e.key === '-' || e.key === '_') {
            e.preventDefault();
            this.zoom(-1);
        }
        
        // Reset zoom (0)
        else if (e.key === '0') {
            e.preventDefault();
            this.resetZoom();
        }
        
        // Download JSON (j)
        else if (e.key === 'j' || e.key === 'J') {
            this.downloadJSON();
        }
        
        // Download YOLO (y)
        else if (e.key === 'y' || e.key === 'Y') {
            this.downloadYOLO();
        }
        
        // Download both as ZIP (z)
        else if ((e.key === 'z' || e.key === 'Z') && !e.ctrlKey) {
            this.downloadBoth();
        }
    }
    
    render() {
        if (!this.image) {
            this.ctx.fillStyle = '#000';
            this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
            this.ctx.fillStyle = '#666';
            this.ctx.font = '16px sans-serif';
            this.ctx.textAlign = 'center';
            this.ctx.fillText('Load an image to begin', this.canvas.width / 2, this.canvas.height / 2);
            return;
        }
        
        // Draw image (scaled by zoom)
        this.ctx.drawImage(this.image, 0, 0, this.canvas.width, this.canvas.height);
        
        // Draw existing annotations
        this.annotations.forEach((ann, idx) => {
            this.drawBox(ann.bbox, ann.class_id, `${ann.class_name} #${idx}`);
        });
        
        // Draw current selection
        if (this.isDrawing && this.startPos && this.currentPos) {
            const x1 = Math.min(this.startPos.x, this.currentPos.x);
            const y1 = Math.min(this.startPos.y, this.currentPos.y);
            const x2 = Math.max(this.startPos.x, this.currentPos.x);
            const y2 = Math.max(this.startPos.y, this.currentPos.y);
            
            // startPos/currentPos are in zoomed canvas pixel coordinates
            // Convert to base canvas coords (divide by zoom), then to image coords (multiply by scale)
            this.drawBox(
                [(x1 / this.zoomLevel) * this.scaleX, 
                 (y1 / this.zoomLevel) * this.scaleY, 
                 (x2 / this.zoomLevel) * this.scaleX, 
                 (y2 / this.zoomLevel) * this.scaleY],
                this.currentClass,
                'Drawing...',
                true
            );
        }
    }
    
    drawBox(bbox, classId, label, isDrawing = false) {
        // Convert to canvas coordinates (accounting for zoom)
        const x1 = (bbox[0] / this.scaleX) * this.zoomLevel;
        const y1 = (bbox[1] / this.scaleY) * this.zoomLevel;
        const x2 = (bbox[2] / this.scaleX) * this.zoomLevel;
        const y2 = (bbox[3] / this.scaleY) * this.zoomLevel;
        
        const color = CLASS_COLORS[classId];
        
        // Set global alpha for opacity control
        this.ctx.globalAlpha = isDrawing ? 1.0 : this.boxOpacity;
        
        // Draw rectangle
        this.ctx.strokeStyle = color;
        this.ctx.lineWidth = (isDrawing ? 3 : 2) * Math.max(0.5, Math.min(1.5, this.zoomLevel / 2));
        this.ctx.setLineDash(isDrawing ? [5, 5] : []);
        this.ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
        this.ctx.setLineDash([]);
        
        // Draw label (only if showLabels is true)
        if (label && this.showLabels) {
            // Clamp font size between 10-16px regardless of zoom
            const fontSize = Math.max(10, Math.min(16, 12 * this.zoomLevel));
            this.ctx.font = `${fontSize}px sans-serif`;
            const textMetrics = this.ctx.measureText(label);
            const textWidth = textMetrics.width;
            const textHeight = fontSize + 4;
            
            this.ctx.fillStyle = color;
            this.ctx.fillRect(x1, y1 - textHeight - 2, textWidth + 8, textHeight + 2);
            
            // Draw label text
            this.ctx.fillStyle = '#fff';
            this.ctx.fillText(label, x1 + 4, y1 - 6);
        }
        
        // Reset global alpha
        this.ctx.globalAlpha = 1.0;
    }
    
    updateUI() {
        // Update stats
        document.getElementById('totalCount').textContent = this.annotations.length;
        
        const counts = { 1: 0, 2: 0, 3: 0 };
        this.annotations.forEach(ann => {
            counts[ann.class_id]++;
        });
        
        document.getElementById('textCount').textContent = counts[1];
        document.getElementById('musicCount').textContent = counts[2];
        document.getElementById('stavesCount').textContent = counts[3];
        
        // Update annotations list
        const listEl = document.getElementById('annotationsList');
        if (this.annotations.length === 0) {
            listEl.innerHTML = '<div style="color: #666; text-align: center; padding: 20px;">No annotations yet. Draw boxes on the canvas!</div>';
        } else {
            listEl.innerHTML = this.annotations.map((ann, idx) => `
                <div class="annotation-item">
                    <span style="color: ${CLASS_COLORS[ann.class_id]}">
                        #${idx} - ${ann.class_name}
                    </span>
                    <span style="color: #888; font-size: 11px;">
                        [${ann.bbox[0]}, ${ann.bbox[1]}, ${ann.bbox[2]}, ${ann.bbox[3]}]
                        <button class="delete-btn" onclick="tool.deleteAnnotation(${idx})">Delete</button>
                    </span>
                </div>
            `).join('');
        }
    }
    
    
    zoom(direction) {
        if (!this.image) return;
        
        // direction: 1 = zoom in, -1 = zoom out
        const oldZoom = this.zoomLevel;
        this.zoomLevel += direction * this.zoomStep;
        
        // Clamp zoom level
        this.zoomLevel = Math.max(this.minZoom, Math.min(this.maxZoom, this.zoomLevel));
        
        if (this.zoomLevel === oldZoom) return; // No change
        
        // Update canvas size
        this.setupCanvas();
        
        // Update cursor
        this.updateCursor();
        
        this.render();
    }
    
    resetZoom() {
        if (!this.image) return;
        
        this.zoomLevel = 1.0;
        this.canvasWrapper.scrollLeft = 0;
        this.canvasWrapper.scrollTop = 0;
        
        this.setupCanvas();
        this.updateCursor();
        this.render();
    }
    
    downloadJSON() {
        if (this.annotations.length === 0) {
            alert('No annotations to download!');
            return;
        }
        
        if (!this.image) {
            alert('No image loaded!');
            return;
        }
        
        // Create session data (matches Python tool format)
        const sessionData = {
            image_path: this.imageName,
            image_width: this.image.width,
            image_height: this.image.height,
            timestamp: new Date().toISOString(),
            class_names: CLASS_NAMES,
            annotations: this.annotations
        };
        
        // Download JSON
        const blob = new Blob([JSON.stringify(sessionData, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `${this.imageName.replace(/\.[^/.]+$/, '')}_annotations.json`;
        a.click();
        URL.revokeObjectURL(url);
        
        console.log(`Downloaded JSON: ${a.download}`);
    }
    
    downloadYOLO() {
        if (this.annotations.length === 0) {
            alert('No annotations to download!');
            return;
        }
        
        if (!this.image) {
            alert('No image loaded!');
            return;
        }
        
        const yoloContent = this.exportYOLO();
        
        // Download YOLO .txt file
        const blob = new Blob([yoloContent], { type: 'text/plain' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `${this.imageName.replace(/\.[^/.]+$/, '')}.txt`;
        a.click();
        URL.revokeObjectURL(url);
        
        console.log(`Downloaded YOLO: ${a.download}`);
    }
    
    async downloadBoth() {
        if (this.annotations.length === 0) {
            alert('No annotations to download!');
            return;
        }
        
        if (!this.image) {
            alert('No image loaded!');
            return;
        }
        
        // Create session data for JSON
        const sessionData = {
            image_path: this.imageName,
            image_width: this.image.width,
            image_height: this.image.height,
            timestamp: new Date().toISOString(),
            class_names: CLASS_NAMES,
            annotations: this.annotations
        };
        
        const jsonContent = JSON.stringify(sessionData, null, 2);
        const yoloContent = this.exportYOLO();
        const baseName = this.imageName.replace(/\.[^/.]+$/, '');
        
        // Create a simple ZIP file using JSZip-like functionality
        // For simplicity, we'll use a workaround with data URLs in a single HTML file
        // Or download separately with a delay
        
        // Option 1: Download both with slight delay (simple, no dependencies)
        this.downloadFile(jsonContent, `${baseName}_annotations.json`, 'application/json');
        
        // Wait a moment before second download
        setTimeout(() => {
            this.downloadFile(yoloContent, `${baseName}.txt`, 'text/plain');
            
            // Also download classes.txt
            setTimeout(() => {
                const classesContent = Object.values(CLASS_NAMES).join('\n');
                this.downloadFile(classesContent, 'classes.txt', 'text/plain');
            }, 200);
        }, 200);
        
        console.log(`Downloaded both formats: JSON + YOLO + classes.txt`);
    }
    
    downloadFile(content, filename, mimeType) {
        const blob = new Blob([content], { type: mimeType });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = filename;
        a.click();
        URL.revokeObjectURL(url);
    }
    
    // Convert annotations to YOLO format
    exportYOLO() {
        if (this.annotations.length === 0 || !this.image) return null;
        
        const yoloLines = this.annotations.map(ann => {
            const [x1, y1, x2, y2] = ann.bbox;
            const classId = ann.class_id - 1; // YOLO uses 0-indexed classes
            
            // Convert to YOLO format (normalized center + width/height)
            const xCenter = (x1 + x2) / (2 * this.image.width);
            const yCenter = (y1 + y2) / (2 * this.image.height);
            const width = (x2 - x1) / this.image.width;
            const height = (y2 - y1) / this.image.height;
            
            return `${classId} ${xCenter.toFixed(6)} ${yCenter.toFixed(6)} ${width.toFixed(6)} ${height.toFixed(6)}`;
        });
        
        return yoloLines.join('\n');
    }
}

// Initialize the tool
const tool = new AnnotationTool();

// Make tool accessible globally for delete buttons
window.tool = tool;