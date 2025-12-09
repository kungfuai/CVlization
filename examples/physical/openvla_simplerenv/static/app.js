/**
 * OpenVLA SimplerEnv Demo - Frontend JavaScript
 *
 * Handles WebSocket communication and canvas rendering for the
 * robot manipulation task visualization.
 */

// DOM Elements
const canvas = document.getElementById('sim-canvas');
const ctx = canvas.getContext('2d');
const taskSelect = document.getElementById('task-select');
const maxStepsInput = document.getElementById('max-steps');
const useRandomCheckbox = document.getElementById('use-random');
const startBtn = document.getElementById('start-btn');
const stopBtn = document.getElementById('stop-btn');
const instructionEl = document.getElementById('instruction');
const stepCountEl = document.getElementById('step-count');
const statusEl = document.getElementById('status');
const taskDescEl = document.getElementById('task-desc');
const embodimentBadge = document.getElementById('embodiment-badge');
const freqBadge = document.getElementById('freq-badge');
const loadingOverlay = document.getElementById('loading-overlay');

// State
let ws = null;
let tasks = [];
let isRunning = false;

/**
 * Initialize the application
 */
async function init() {
    await loadTasks();
    setupEventListeners();
    drawPlaceholder();
}

/**
 * Load available tasks from the API
 */
async function loadTasks() {
    try {
        const response = await fetch('/api/tasks');
        const data = await response.json();
        tasks = data.tasks;

        // Populate task select
        taskSelect.innerHTML = '';
        tasks.forEach(task => {
            const option = document.createElement('option');
            option.value = task.id;
            option.textContent = task.id.replace(/_/g, ' ');
            taskSelect.appendChild(option);
        });

        // Update description for first task
        if (tasks.length > 0) {
            updateTaskDescription(tasks[0]);
        }
    } catch (error) {
        console.error('Failed to load tasks:', error);
        taskSelect.innerHTML = '<option value="">Failed to load tasks</option>';
    }
}

/**
 * Update task description panel
 */
function updateTaskDescription(task) {
    taskDescEl.textContent = task.description;

    embodimentBadge.textContent = task.embodiment;
    embodimentBadge.classList.remove('hidden');

    freqBadge.textContent = `${task.control_freq} Hz`;
    freqBadge.classList.remove('hidden');
}

/**
 * Set up event listeners
 */
function setupEventListeners() {
    taskSelect.addEventListener('change', () => {
        const task = tasks.find(t => t.id === taskSelect.value);
        if (task) {
            updateTaskDescription(task);
        }
    });

    startBtn.addEventListener('click', startEpisode);
    stopBtn.addEventListener('click', stopEpisode);
}

/**
 * Draw placeholder image on canvas
 */
function drawPlaceholder() {
    ctx.fillStyle = '#1a1a2e';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    ctx.fillStyle = '#4a4a6a';
    ctx.font = '18px system-ui';
    ctx.textAlign = 'center';
    ctx.fillText('Select a task and click Start', canvas.width / 2, canvas.height / 2);
}

/**
 * Start an episode
 */
function startEpisode() {
    if (isRunning) return;

    const taskId = taskSelect.value;
    if (!taskId) {
        setStatus('Please select a task', 'error');
        return;
    }

    isRunning = true;
    startBtn.disabled = true;
    stopBtn.disabled = false;
    loadingOverlay.classList.remove('hidden');
    setStatus('Connecting...', 'info');

    // Create WebSocket connection
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    ws = new WebSocket(`${protocol}//${window.location.host}/ws`);

    ws.onopen = () => {
        setStatus('Loading model...', 'info');

        // Send initialization message
        ws.send(JSON.stringify({
            task_id: taskId,
            max_steps: parseInt(maxStepsInput.value, 10),
            use_random: useRandomCheckbox.checked,
        }));
    };

    ws.onmessage = (event) => {
        const msg = JSON.parse(event.data);
        handleMessage(msg);
    };

    ws.onerror = (error) => {
        console.error('WebSocket error:', error);
        setStatus('Connection error', 'error');
        cleanup();
    };

    ws.onclose = () => {
        if (isRunning) {
            setStatus('Connection closed', 'info');
        }
        cleanup();
    };
}

/**
 * Stop the current episode
 */
function stopEpisode() {
    if (ws) {
        ws.close();
    }
    cleanup();
    setStatus('Stopped', 'info');
}

/**
 * Handle incoming WebSocket messages
 */
function handleMessage(msg) {
    switch (msg.type) {
        case 'init':
            loadingOverlay.classList.add('hidden');
            setStatus('Running...', 'running');
            break;

        case 'frame':
            displayFrame(msg);
            break;

        case 'done':
            const successText = msg.success ? 'Success!' : 'Episode ended';
            setStatus(`${successText} (${msg.steps} steps)`, msg.success ? 'success' : 'info');
            cleanup();
            break;

        case 'error':
            setStatus(`Error: ${msg.message}`, 'error');
            loadingOverlay.classList.add('hidden');
            cleanup();
            break;
    }
}

/**
 * Display a frame on the canvas
 */
function displayFrame(msg) {
    // Update info
    instructionEl.textContent = msg.instruction || '-';
    stepCountEl.textContent = msg.step;

    // Draw image
    const img = new Image();
    img.onload = () => {
        // Fit image to canvas while maintaining aspect ratio
        const scale = Math.min(
            canvas.width / img.width,
            canvas.height / img.height
        );
        const x = (canvas.width - img.width * scale) / 2;
        const y = (canvas.height - img.height * scale) / 2;

        ctx.fillStyle = '#1a1a2e';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        ctx.drawImage(img, x, y, img.width * scale, img.height * scale);
    };
    img.src = `data:image/jpeg;base64,${msg.image}`;
}

/**
 * Set status message with styling
 */
function setStatus(text, type = 'info') {
    statusEl.textContent = text;
    statusEl.className = 'value status-' + type;
}

/**
 * Clean up after episode ends
 */
function cleanup() {
    isRunning = false;
    startBtn.disabled = false;
    stopBtn.disabled = true;
    loadingOverlay.classList.add('hidden');
    ws = null;
}

// Initialize on load
document.addEventListener('DOMContentLoaded', init);
