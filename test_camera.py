"""
Test Camera Script - ตรวจสอบว่ากล้องใช้งานได้หรือไม่
"""
import cv2
import time

print("🔍 Testing Camera Availability")
print("=" * 60)

# Test different camera indices and backends
backends = [
    (cv2.CAP_DSHOW, "DirectShow (Windows)"),
    (cv2.CAP_MSMF, "Media Foundation (Windows)"),
    (cv2.CAP_ANY, "Auto detect")
]

found_camera = False

for camera_index in range(3):
    print(f"\n📹 Testing Camera Index: {camera_index}")
    print("-" * 60)
    
    for backend, backend_name in backends:
        try:
            print(f"  Trying {backend_name}...", end=" ")
            cap = cv2.VideoCapture(camera_index, backend)
            
            if cap.isOpened():
                # Try to read a frame
                ret, frame = cap.read()
                
                if ret and frame is not None:
                    h, w = frame.shape[:2]
                    print(f"✅ SUCCESS!")
                    print(f"    Resolution: {w}x{h}")
                    print(f"    FPS: {cap.get(cv2.CAP_PROP_FPS)}")
                    
                    # Show frame for 2 seconds
                    cv2.imshow(f"Camera {camera_index} - {backend_name}", frame)
                    cv2.waitKey(2000)
                    cv2.destroyAllWindows()
                    
                    found_camera = True
                else:
                    print("❌ Cannot read frame")
                
                cap.release()
            else:
                print("❌ Cannot open")
        except Exception as e:
            print(f"❌ Error: {str(e)}")
        
        time.sleep(0.2)

print("\n" + "=" * 60)
if found_camera:
    print("✅ At least one camera is working!")
    print("\n💡 Use the working camera index and backend in camera_simple.py")
else:
    print("❌ No working camera found!")
    print("\n💡 Troubleshooting:")
    print("  1. Check if camera is physically connected")
    print("  2. Check Windows Device Manager (Win+X → Device Manager)")
    print("  3. Close other apps using camera:")
    print("     - Zoom, Teams, Skype, OBS, etc.")
    print("  4. Check camera privacy settings:")
    print("     - Settings → Privacy → Camera")
    print("  5. Try unplugging and replugging USB camera")
    print("  6. Restart computer")

print("=" * 60)
input("\nPress Enter to exit...")
