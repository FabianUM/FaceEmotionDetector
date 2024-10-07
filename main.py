from controllers.FaceEmotionVideoBar import FaceEmotionDetector
from pages.gui import create_gui

def start_detection():
    detector = FaceEmotionDetector()
    detector.detect_emotions()

def exit_program(window):
    window.quit()

if __name__ == "__main__":
    create_gui(start_detection, exit_program)