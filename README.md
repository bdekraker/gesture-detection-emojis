
# Gesture Detection and Emoji Animation

This project implements a real-time gesture detection system using **MediaPipe** for hand and face tracking. It recognizes various hand gestures like thumbs up, heart, facepalm, OK hand, and victory hand, and responds by showing floating emojis on the screen.

The script also demonstrates how to draw emojis and text on frames captured by a webcam using **OpenCV** and **PIL (Pillow)**. Emojis like ❤️ (Heart), 🤦 (Facepalm), and 👍 (Thumbs Up) are displayed based on the detected gestures. Additionally, a floating animation effect is applied to these emojis, which move upward before disappearing after a short time.



https://github.com/user-attachments/assets/46d2fd71-3ae9-4ab6-a1a9-24f76a34c863



## Features

- **Gesture Recognition**: Detects several hand gestures, including:
  - 👍 **Thumbs Up**
  - ✌️ **Victory Hand**
  - 🤦 **Facepalm**
  - ❤️ **Heart Gesture** (with both hands)
  - 👌 **OK Hand**
- **Emoji and Text Rendering**: Displays emojis and corresponding text on-screen when a gesture is detected.
- **Floating Animation**: The detected emoji floats upwards for 2 seconds after a gesture is recognized.
- **Real-time Processing**: Uses the webcam for live input, making it interactive and suitable for video streams.

## How It Works

1. **Gesture Detection**: The system tracks hand and face landmarks using **MediaPipe**. Based on the position of landmarks, various gestures are detected.
2. **Emoji Rendering**: After a gesture is detected, an emoji and text are drawn on the screen at the location of the detected gesture. The emoji floats upwards with time, creating an animated effect.
3. **History Buffer**: A sliding window of recent gestures is maintained using a deque. This helps determine the most frequent gesture in the last few frames, ensuring that the emoji display is stable and doesn't flicker.

## Requirements

Install the necessary Python packages by running:

\`\`\`bash
pip install -r requirements.txt
\`\`\`

Here’s the content for \`requirements.txt\`:

\`\`\`txt
opencv-python
mediapipe
numpy
Pillow
\`\`\`

## Running the Project

To run the project, ensure your webcam is connected, then execute the following command:

\`\`\`bash
python gesture_detection.py
\`\`\`

The program will automatically open the webcam and start detecting gestures. When a recognized gesture is detected, it will display an emoji on the screen that floats upwards for a few seconds.

### Supported Gestures and Emojis

| Gesture          | Emoji | Description                                |
|------------------|-------|--------------------------------------------|
| Thumbs Up        | 👍    | Detected when thumb is pointing upward      |
| Victory Hand     | ✌️    | Detected with a "peace" sign gesture        |
| Facepalm         | 🤦    | Detected when hand covers the face          |
| Heart Gesture    | ❤️    | Detected using both hands to form a heart   |
| OK Hand          | 👌    | Detected when thumb and index finger touch  |

### Future Improvements

- **Expand Gesture Set**: Add new hand gestures like:
  - 👋 Waving hand gesture
  - ✍️ Writing gesture
  - 🖖 Vulcan salute gesture
- **Custom Emoji**: Add support for custom emojis that users can define.
- **Sound Effects**: Play a sound when a gesture is detected to make the interaction more engaging.
- **Multi-language Support**: Translate floating text to different languages.
- **Gesture-based Controls**: Use hand gestures to control elements in the interface, such as pausing/resuming video, changing filters, etc.
- **Face Expressions**: Add recognition of face expressions (e.g., smile, surprise) and trigger specific emojis or actions.

### Troubleshooting

1. **Emoji Display Issue**:
   - Ensure the path to the emoji font file (Segoe UI Emoji) is correct. The current path in the script is for Windows:
     \`\`\`
     C:/Windows/Fonts/seguiemj.ttf
     \`\`\`
   - For other operating systems, replace this path with the correct font path (e.g., \`/Library/Fonts/Apple Color Emoji.ttc\` for macOS).

2. **Webcam Issues**:
   - Ensure your webcam is properly connected.
   - If using an external camera, modify the \`cv2.VideoCapture(0)\` to the appropriate camera index.

3. **Slow Performance**:
   - Reduce the frame size to improve performance. Modify this section in the code to lower the screen height and width:
     \`\`\`python
     screen_height, screen_width = 480, 640  # Adjust to smaller size for performance
     \`\`\`

4. **MediaPipe Hands Not Detecting**:
   - Ensure the environment is properly set up with the required packages. Run \`pip list\` to check if \`mediapipe\` is installed.

### License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
