import cv2
import threading
import time
from datetime import datetime
from typing import List, Dict, Optional
import numpy as np
import subprocess
import os
import pyaudio
import wave
import queue


class AudioRecorder:
    def __init__(self, device_index=None):
        self.CHUNK = 1024
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 2
        self.RATE = 44100
        self.p = pyaudio.PyAudio()
        self.frames = []
        self.is_recording = False
        self.device_index = device_index
        self.start_time = None
        self.audio_queue = queue.Queue()
        
    def start_recording(self):
        self.is_recording = True
        self.frames = []
        self.start_time = time.perf_counter()
        
        self.stream = self.p.open(
            format=self.FORMAT,
            channels=self.CHANNELS,
            rate=self.RATE,
            input=True,
            input_device_index=self.device_index,
            frames_per_buffer=self.CHUNK,
            stream_callback=self._audio_callback
        )
        
        self.stream.start_stream()
        print(f"Audio recording started at {self.start_time}")
        
        # Start processing thread
        self.processing_thread = threading.Thread(target=self._process_audio_queue)
        self.processing_thread.start()
        
    def _audio_callback(self, in_data, frame_count, time_info, status):
        """Non-blocking audio callback for lower latency."""
        if self.is_recording:
            self.audio_queue.put(in_data)
        return (None, pyaudio.paContinue)
    
    def _process_audio_queue(self):
        """Process audio data from queue in separate thread."""
        while self.is_recording or not self.audio_queue.empty():
            try:
                data = self.audio_queue.get(timeout=0.1)
                self.frames.append(data)
            except queue.Empty:
                continue
                
    def stop_recording(self):
        self.is_recording = False
        
        if hasattr(self, 'stream'):
            self.stream.stop_stream()
            self.stream.close()
            
        if hasattr(self, 'processing_thread'):
            self.processing_thread.join()
        
        end_time = time.perf_counter()
        duration = end_time - self.start_time if self.start_time else 0
        print(f"Audio recording stopped. Duration: {duration:.2f}s")
        
    def save_recording(self, filename):
        wf = wave.open(filename, 'wb')
        wf.setnchannels(self.CHANNELS)
        wf.setsampwidth(self.p.get_sample_size(self.FORMAT))
        wf.setframerate(self.RATE)
        wf.writeframes(b''.join(self.frames))
        wf.close()
        print(f"Audio saved to {filename}")
        
    def close(self):
        self.p.terminate()


class CameraRecorder:
    def __init__(self, camera_index: int = 0):
        """
        Initialize the CameraRecorder.
        
        Args:
            camera_index: Index of the camera to use
        """
        self.camera_index = camera_index
        self.cap = None
        self.is_running = False
        self.is_recording = False
        self.video_writer = None
        self.thread = None
        self.noise_reduction_enabled = True
        self.noise_reduction_strength = 10  # Adjustable strength (1-50)
        self.frame_buffer = []
        self.recording_thread = None
        self.temp_filename = None
        self.final_filename = None
        self.audio_recorder = None
        self.temp_audio_filename = None
        self.record_audio = False
        self.video_start_time = None
        self.audio_sync_offset = 0.0  # Audio delay compensation in seconds
        self.frame_timestamps = []
        self.actual_frame_count = 0
        

    def start(self, 
              show_video: bool = False, 
              record_video: bool = False, 
              output_filename: Optional[str] = None, 
              width: int = 1920, 
              height: int = 1080, 
              fps: int = 60,
              output_format: str = 'mov', 
              enable_noise_reduction: bool = False,
              bitrate: str = '5M', 
              quality: int = 23,
              record_audio: bool = True,
              audio_device_index: Optional[int] = None,
              audio_sync_offset: float = -0.1):
        """
        Start camera capture, display, and/or recording.
        
        Args:
            show_video: Whether to display the video
            record_video: Whether to record the video
            output_filename: Custom filename for recording (optional)
            width: Desired width (default 1920 for 1080p)
            height: Desired height (default 1080 for 1080p)
            fps: Desired FPS (default 30)
            output_format: Output format ('mov' or 'mp4', default 'mov')
            enable_noise_reduction: Whether to apply noise reduction (default True)
            bitrate: Target bitrate (e.g., '5M' for 5 Mbps, '8M' for 8 Mbps)
            quality: H.264 quality setting (0-51, lower = better quality, 23 = default)
            record_audio: Whether to record audio along with video (default True)
            audio_device_index: Audio device index to use (None for default)
            audio_sync_offset: Audio sync adjustment in seconds (positive = delay audio, negative = advance audio)
        """
        if self.is_running:
            print("Camera is already running")
            return
            
        self.noise_reduction_enabled = enable_noise_reduction
        self.target_bitrate = bitrate
        self.quality = quality
        self.record_audio = record_audio and record_video
        self.audio_sync_offset = audio_sync_offset
        self.frame_timestamps = []
        self.actual_frame_count = 0
            
        # Open camera
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            print(f"Cannot open camera {self.camera_index}")
            return
        
        # IMPORTANT: Set these properties BEFORE reading any frames
        # Set video codec to MJPEG for better performance at high resolutions
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        
        # Set desired resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        
        # Verify actual resolution
        self.actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
        
        print(f"Requested: {width}x{height} @ {fps}fps")
        print(f"Actual: {self.actual_width}x{self.actual_height} @ {self.actual_fps}fps")
        
        # Setup recording if requested
        if record_video:
            if output_filename is None:
                output_filename = f"recording_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{output_format}"
            else:
                output_filename = output_filename + f".{output_format}"

            if os.path.exists(output_filename):
                response = input(f"File {output_filename} already exists. Overwrite? (y/n): ")
                if response.lower() != 'y':
                    raise FileExistsError(f"File {output_filename} already exists. Choose a different name or delete it.")

            self.final_filename = output_filename
            
            # Setup audio recording if requested - BEFORE video setup
            if self.record_audio:
                try:
                    self.audio_recorder = AudioRecorder(device_index=audio_device_index)
                    self.temp_audio_filename = f"temp_audio_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
                    print(f"Audio recording enabled (device index: {audio_device_index or 'default'})")
                    print(f"Audio sync offset: {audio_sync_offset}s")
                except Exception as e:
                    print(f"Failed to initialize audio recording: {e}")
                    print("Continuing with video only...")
                    self.record_audio = False
                    self.audio_recorder = None
            
            # Method 1: Direct OpenCV recording (limited compression control)
            if not self._check_ffmpeg_available():
                print("FFmpeg not found. Using OpenCV recording with limited compression.")
                # Use H.264 codec with OpenCV
                fourcc = cv2.VideoWriter_fourcc(*'MJPG')
                self.video_writer = cv2.VideoWriter(output_filename, fourcc, 
                                                    self.actual_fps, (self.actual_width, self.actual_height))
                
            else:
                # Method 2: Use temporary file and compress with FFmpeg
                print("Using FFmpeg for advanced compression")
                self.temp_filename = f"temp_{datetime.now().strftime('%Y%m%d_%H%M%S')}.avi"
                
                # Record to temporary file with lossless/high-quality codec
                fourcc = cv2.VideoWriter_fourcc(*'MJPG')  # High quality temporary
                self.video_writer = cv2.VideoWriter(self.temp_filename, fourcc, 
                                                    self.actual_fps, (self.actual_width, self.actual_height))
            
            self.is_recording = True
            print(f"Recording to: {output_filename}")
            print(f"Target bitrate: {bitrate}")
            print(f"Quality setting: {quality}")
            print(f"Noise reduction: {'Enabled' if self.noise_reduction_enabled else 'Disabled'}")
            print(f"Audio recording: {'Enabled' if self.record_audio else 'Disabled'}")
        
        # Start capture thread
        self.is_running = True
        self.thread = threading.Thread(target=self._capture_loop, args=(show_video,))
        self.thread.daemon = True
        self.thread.start()
        
        # Wait a moment for video to stabilize, then start audio recording
        if self.is_recording and self.record_audio and self.audio_recorder:
            self.audio_recorder.start_recording()
            self.video_start_time = time.perf_counter()
    
    def _check_ffmpeg_available(self):
        """Check if FFmpeg is available on the system."""
        try:
            subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False
    
    def _compress_video_with_ffmpeg(self):
        """Compress the temporary video file using FFmpeg with iPhone-like settings."""
        if not self.temp_filename or not os.path.exists(self.temp_filename):
            print("No temporary file to compress")
            return
        
        print(f"\nCompressing video with FFmpeg...")
        print(f"Input: {self.temp_filename}")
        print(f"Output: {self.final_filename}")
        
        # Calculate actual video duration and FPS from timestamps
        if self.frame_timestamps and len(self.frame_timestamps) > 1:
            actual_duration = self.frame_timestamps[-1] - self.frame_timestamps[0]
            actual_fps_measured = len(self.frame_timestamps) / actual_duration
            print(f"Actual video duration: {actual_duration:.2f}s")
            print(f"Actual FPS (measured): {actual_fps_measured:.1f}")
            print(f"Total frames: {len(self.frame_timestamps)}")
        
        # Check if we have audio to merge
        if self.record_audio and self.temp_audio_filename and os.path.exists(self.temp_audio_filename):
            print(f"Audio input: {self.temp_audio_filename}")
            
            # FFmpeg command with audio merging and sync adjustment
            cmd = [
                'ffmpeg',
                '-i', self.temp_filename,  # Video input
                '-itsoffset', str(self.audio_sync_offset),  # Audio offset
                '-i', self.temp_audio_filename,  # Audio input
                '-c:v', 'libx264',  # H.264 codec
                '-preset', 'slow',  # Better compression
                '-crf', str(self.quality),  # Constant Rate Factor
                '-b:v', self.target_bitrate,  # Target bitrate
                '-maxrate', self.target_bitrate,  # Maximum bitrate
                '-bufsize', str(int(float(self.target_bitrate[:-1]) * 2)) + 'M',  # Buffer size
                '-c:a', 'aac',  # Audio codec
                '-b:a', '192k',  # Audio bitrate
                '-movflags', '+faststart',  # Optimize for streaming
                '-pix_fmt', 'yuv420p',  # Pixel format for compatibility
                '-async', '1',  # Audio sync method
                '-vsync', 'cfr',  # Constant frame rate
                '-r', str(self.actual_fps),  # Force output frame rate
                '-shortest',  # Stop encoding when shortest stream ends
                '-y',  # Overwrite output file
                self.final_filename
            ]
        else:
            # FFmpeg command without audio (original behavior)
            cmd = [
                'ffmpeg',
                '-i', self.temp_filename,  # Input file
                '-c:v', 'libx264',  # H.264 codec
                '-preset', 'slow',  # Better compression
                '-crf', str(self.quality),  # Constant Rate Factor
                '-b:v', self.target_bitrate,  # Target bitrate
                '-maxrate', self.target_bitrate,  # Maximum bitrate
                '-bufsize', str(int(float(self.target_bitrate[:-1]) * 2)) + 'M',  # Buffer size
                '-movflags', '+faststart',  # Optimize for streaming
                '-pix_fmt', 'yuv420p',  # Pixel format for compatibility
                '-r', str(self.actual_fps),  # Force output frame rate
                '-vsync', 'cfr',  # Constant frame rate
                '-y',  # Overwrite output file
                self.final_filename
            ]
        
        try:
            # Run FFmpeg
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                # Get file sizes for comparison
                temp_size = os.path.getsize(self.temp_filename) / (1024 * 1024)  # MB
                final_size = os.path.getsize(self.final_filename) / (1024 * 1024)  # MB
                compression_ratio = (1 - final_size / temp_size) * 100
                
                print(f"\nCompression successful!")
                print(f"Original video size: {temp_size:.1f} MB")
                print(f"Compressed size: {final_size:.1f} MB")
                print(f"Compression ratio: {compression_ratio:.1f}%")
                
                # Delete temporary files
                os.remove(self.temp_filename)
                if self.temp_audio_filename and os.path.exists(self.temp_audio_filename):
                    os.remove(self.temp_audio_filename)
                    print("Temporary audio file deleted")
            else:
                print(f"FFmpeg error: {result.stderr}")
                
        except Exception as e:
            print(f"Error during compression: {e}")
    
    def set_noise_reduction_strength(self, strength: int):
        """
        Set the noise reduction strength.
        
        Args:
            strength: Strength value (1-50, higher = more smoothing)
        """
        self.noise_reduction_strength = max(1, min(50, strength))
        print(f"Noise reduction strength set to: {self.noise_reduction_strength}")
    
    def toggle_noise_reduction(self):
        """Toggle noise reduction on/off."""
        self.noise_reduction_enabled = not self.noise_reduction_enabled
        print(f"Noise reduction: {'Enabled' if self.noise_reduction_enabled else 'Disabled'}")
    
    def reduce_noise(self, frame: np.ndarray) -> np.ndarray:
        """
        Apply noise reduction to a frame.
        
        Args:
            frame: Input frame
            
        Returns:
            Denoised frame
        """
        if not self.noise_reduction_enabled:
            return frame
        
        # Use bilateral filter for good balance of speed and quality
        denoised = cv2.bilateralFilter(frame, 9, self.noise_reduction_strength * 5, self.noise_reduction_strength * 5)
        
        return denoised
        

    def stop(self):
        """Stop camera capture and recording."""
        self.is_running = False
        if self.thread:
            self.thread.join()
            
        # Stop audio recording first if active
        if self.record_audio and self.audio_recorder:
            self.audio_recorder.stop_recording()
            if self.temp_audio_filename:
                self.audio_recorder.save_recording(self.temp_audio_filename)
            self.audio_recorder.close()
            
        if self.video_writer:
            self.video_writer.release()
            
        if self.cap:
            self.cap.release()
            
        cv2.destroyAllWindows()
        
        # Compress video if using FFmpeg method
        if self.temp_filename and self._check_ffmpeg_available():
            self._compress_video_with_ffmpeg()
        
        print("Camera stopped")
        

    def _capture_loop(self, show_video: bool):
        """Main capture loop running in separate thread."""
        current_fps = self.actual_fps
        frame_count = 0
        start_time = time.perf_counter()
        
        while self.is_running:
            ret, frame = self.cap.read()
            self.last_frame = frame
            if not ret:
                print("Failed to read frame")
                break
            
            current_time = time.perf_counter()
            frame_count += 1
            
            # Apply noise reduction
            processed_frame = self.reduce_noise(frame)
            
            # Display video if requested
            if show_video:
                display_frame = processed_frame.copy()
                
                # Calculate actual FPS
                if frame_count % self.actual_fps == 0:
                    elapsed = current_time - start_time
                    current_fps = frame_count / elapsed
                cv2.putText(display_frame, f"FPS: {current_fps:.1f}", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Show noise reduction status
                if self.noise_reduction_enabled:
                    cv2.putText(display_frame, f"NR: ON (Strength: {self.noise_reduction_strength})", 
                               (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                else:
                    cv2.putText(display_frame, "NR: OFF", (10, 60), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # Show recording info
                if self.is_recording:
                    rec_text = f"REC - Bitrate: {self.target_bitrate}"
                    if self.record_audio:
                        rec_text += f" [Audio ON, Sync: {self.audio_sync_offset}s]"
                    cv2.putText(display_frame, rec_text, 
                               (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                cv2.imshow(f'Camera {self.camera_index}', display_frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    self.is_running = False
                    break
                elif key == ord('n'):
                    self.toggle_noise_reduction()
                elif key == ord('+') or key == ord('='):
                    self.set_noise_reduction_strength(self.noise_reduction_strength + 5)
                elif key == ord('-'):
                    self.set_noise_reduction_strength(self.noise_reduction_strength - 5)
                    
            # Record video if writer is setup
            if self.is_recording and self.video_writer:
                self.video_writer.write(processed_frame)
                # Store timestamp for this frame
                if self.video_start_time:
                    self.frame_timestamps.append(current_time - self.video_start_time)
                self.actual_frame_count += 1


def get_camera_capabilities(camera_index: int = 0) -> Dict:
    """
    Get detailed camera capabilities by testing different resolutions.
    
    Args:
        camera_index: Camera index to test
        
    Returns:
        Dictionary with camera capabilities
    """
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        return None
    
    # Set to MJPEG for better high-res support
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    
    # Test common resolutions
    resolutions_to_test = [
        (3840, 2160, "4K"),
        (2560, 1440, "1440p"),
        (1920, 1080, "1080p"),
        (1280, 720, "720p"),
        (640, 480, "480p")
    ]
    
    supported_resolutions = []
    
    for width, height, name in resolutions_to_test:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        
        actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        if actual_width == width and actual_height == height:
            supported_resolutions.append({
                'width': width,
                'height': height,
                'name': name,
                'fps': fps
            })
    
    # Get backend info
    backend = cap.getBackendName()
    
    cap.release()
    
    return {
        'index': camera_index,
        'backend': backend,
        'supported_resolutions': supported_resolutions
    }


def print_camera_details():
    """Print detailed information about all cameras."""
    print("\nDetailed Camera Information:")
    print("=" * 60)
    
    for i in range(10):  # Check first 10 camera indices
        info = get_camera_capabilities(i)
        if info:
            print(f"\nCamera {i} (Backend: {info['backend']}):")
            print("-" * 40)
            for res in info['supported_resolutions']:
                print(f"  {res['name']}: {res['width']}x{res['height']} @ {res['fps']}fps")


def list_audio_devices():
    """List all available audio input devices."""
    p = pyaudio.PyAudio()
    print("\nAvailable Audio Devices:")
    print("=" * 60)
    
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        if info['maxInputChannels'] > 0:  # Only input devices
            print(f"Device {i}: {info['name']}")
            print(f"  Channels: {info['maxInputChannels']}")
            print(f"  Sample Rate: {info['defaultSampleRate']}")
            print("-" * 40)
    
    p.terminate()


def diagnose_camera(camera_index=0):
    cap = cv2.VideoCapture(camera_index)
    
    # Try different formats
    formats = [
        ('MJPG', cv2.VideoWriter_fourcc(*'MJPG')),
        ('MP4V', cv2.VideoWriter_fourcc(*'MP4V')),
        ('H264', cv2.VideoWriter_fourcc(*'H264')),
        ('YUYV', cv2.VideoWriter_fourcc(*'YUYV')),
        ('XVID', cv2.VideoWriter_fourcc(*'XVID')),
        ('AVC1', cv2.VideoWriter_fourcc(*'AVC1'))
    ]
    
    for format_name, fourcc in formats:
        print(f"\nTesting {format_name} format:")
        
        for requested_fps in [60, 30, 15]:
            cap.set(cv2.CAP_PROP_FOURCC, fourcc)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
            cap.set(cv2.CAP_PROP_FPS, requested_fps)
            time.sleep(0.5)
            
            actual_fps = cap.get(cv2.CAP_PROP_FPS)
            actual_buffer = cap.get(cv2.CAP_PROP_BUFFERSIZE)
            print(f"Camera reports: {actual_fps} FPS, buffer size: {actual_buffer}")
            
            # Measure actual FPS
            start = time.perf_counter()
            frames = 0
            while frames < 4*requested_fps:
                ret, _ = cap.read()
                if ret:
                    frames += 1
                else:
                    print("  Frame read failed")
            elapsed = time.perf_counter() - start
            actual_fps = frames / elapsed
            
            print(f"  Requested {requested_fps}fps → Actual {actual_fps:.1f}fps")
    
    cap.release()


# Example usage
if __name__ == "__main__":
    # Show available cameras
    print_camera_details()
    
    # Show available audio devices
    list_audio_devices()
    
    # Use the camera
    recorder = CameraRecorder(camera_index=6)
    
    # Start displaying and recording with iPhone-like compression
    recorder.start(
        show_video=True, 
        record_video=False,
        output_format='mov',
        enable_noise_reduction=False,
        bitrate='6M',  # 4 Mbps like iPhone (adjust between 2M-8M)
        quality=23,     # H.264 quality (18-28 range, 23 is balanced)
        record_audio=True,  # Enable audio recording
        audio_device_index=None,  # Use default audio device, or specify device index
    )
    
    print("\nControls:")
    print("  'q' - Quit")
    print("  'n' - Toggle noise reduction")
    print("  '+' - Increase noise reduction strength")
    print("  '-' - Decrease noise reduction strength")
    
    try:
        # Keep main thread alive
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping...")
        
    # Stop recording
    recorder.stop()

    # diagnose_camera(camera_index=6)