# Frontend Changes: Dark/Light Theme Toggle

## Overview
Added a theme toggle button that allows users to switch between dark and light themes with smooth transitions and persistent preference storage.

## Files Modified

### 1. `index.html`
- Added a theme toggle button with sun/moon SVG icons positioned in the top-right corner
- Button includes proper accessibility attributes (`aria-label`, `title`)
- Uses semantic HTML with keyboard navigation support

### 2. `style.css`
**New CSS Variables for Light Theme:**
- Added `[data-theme="light"]` selector with light-appropriate colors:
  - `--background: #f8fafc` (light gray background)
  - `--surface: #ffffff` (white surface)
  - `--text-primary: #1e293b` (dark text for contrast)
  - `--text-secondary: #64748b` (muted gray text)
  - `--border-color: #e2e8f0` (subtle borders)
  - `--assistant-message: #f1f5f9` (light message bubbles)
  - `--code-bg: rgba(0, 0, 0, 0.05)` (subtle code background)

**New Theme Toggle Button Styles:**
- Fixed position in top-right corner (`top: 1rem; right: 1rem`)
- Circular button design (44x44px) with border and shadow
- Hover, focus, and active states for visual feedback
- Icon switching logic (moon icon in dark mode, sun icon in light mode)

**Smooth Transitions:**
- Added CSS transitions to all themed elements for smooth color changes
- 0.3s ease transition on background-color, color, border-color, and box-shadow

### 3. `script.js`
**New Functions:**
- `initializeTheme()`: Loads saved theme preference from localStorage on page load
- `toggleTheme()`: Switches between dark/light themes and saves preference

**Event Listener:**
- Theme toggle button click triggers `toggleTheme()` function

## Features Implemented

1. **Toggle Button Design**
   - Circular button positioned in top-right corner
   - Sun icon displayed in light mode, moon icon in dark mode
   - Smooth scale animation on hover/click
   - Focus ring for keyboard navigation

2. **Light Theme Colors**
   - Light backgrounds with dark text for good contrast
   - Maintained visual hierarchy and design language
   - Proper accessibility standards met

3. **JavaScript Functionality**
   - Theme toggles on button click
   - Preference persisted in localStorage
   - Smooth transitions between themes

4. **Implementation Details**
   - Uses CSS custom properties (variables) for theme switching
   - `data-theme` attribute on `<html>` element controls theme
   - All existing elements work in both themes
   - Keyboard accessible (Tab + Enter/Space)

## Usage
Click the sun/moon button in the top-right corner to toggle between themes. Your preference is automatically saved and restored on page reload.
