/**
 * common.js - Shared JavaScript logic for Skin Disease Detector
 */

document.addEventListener('DOMContentLoaded', () => {
    // Initialize Lucide icons
    if (typeof lucide !== 'undefined') {
        lucide.createIcons();
    }
});

/**
 * Utility to show an error message in a specific element
 * @param {string} elementId 
 * @param {string} message 
 */
function showError(elementId, message) {
    const errorDiv = document.getElementById(elementId);
    if (errorDiv) {
        errorDiv.textContent = message;
        errorDiv.classList.remove('hidden');
    }
}

/**
 * Utility to show a success message in a specific element
 * @param {string} elementId 
 * @param {string} message 
 */
function showSuccess(elementId, message) {
    const successDiv = document.getElementById(elementId);
    if (successDiv) {
        successDiv.textContent = message;
        successDiv.classList.remove('hidden');
    }
}
