import sys
import os
import time
import subprocess
from pathlib import Path
import shutil
import threading
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import librosa
import librosa.display
from scipy import signal
import soundfile as sf

from PyQt5.QtWidgets import (QApplication, QMainWindow, QFileDialog, QPushButton, 
                             QVBoxLayout, QHBoxLayout, QWidget, QLabel, QProgressBar, 
                             QListWidget, QMessageBox, QFrame, QSplitter, 
                             QListWidgetItem, QAbstractItemView, QLineEdit, QFormLayout,
                             QTabWidget, QScrollArea, QGridLayout, QGroupBox)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QPropertyAnimation, QEasingCurve, QSize, QTimer, QObject
from PyQt5.QtGui import QIcon, QColor, QPalette, QFont, QDragEnterEvent, QDropEvent, QPixmap

# Check for FFmpeg and set the path
FFMPEG_PATH = 'ffmpeg'  # Default to system path

# First check if FFmpeg is in the local directory
local_ffmpeg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ffmpeg', 'bin', 'ffmpeg.exe')
if os.path.exists(local_ffmpeg_path):
    FFMPEG_PATH = local_ffmpeg_path
    print(f"Using local FFmpeg: {FFMPEG_PATH}")
else:
    # Check if FFmpeg is in the system path
    try:
        result = subprocess.run(['ffmpeg', '-version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode == 0:
            print("Using system FFmpeg")
        else:
            print("Warning: FFmpeg found but returned an error code.")
            print("FFmpeg is required for audio conversion.")
    except FileNotFoundError:
        print("FFmpeg not found in system PATH.")
        print("Checking for FFmpeg in local directory...")
        
        # Check common installation paths
        possible_paths = [
            os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ffmpeg.exe'),
            os.path.join(os.path.dirname(os.path.abspath(__file__)), 'bin', 'ffmpeg.exe'),
        ]
        
        ffmpeg_found = False
        for path in possible_paths:
            if os.path.exists(path):
                FFMPEG_PATH = path
                print(f"Found FFmpeg: {FFMPEG_PATH}")
                ffmpeg_found = True
                break
        
        if not ffmpeg_found:
            print("Error: FFmpeg not found. Please ensure the FFmpeg executable is in one of these locations:")
            print("1. In your system PATH")
            print("2. In the 'ffmpeg/bin' directory next to this script")
            print("3. In the same directory as this script")
            sys.exit(1)


class AudioAnalyzer:
    """Class for analyzing audio files and comparing original with converted files"""
    
    @staticmethod
    def analyze_file(file_path):
        """Load and analyze a single audio file"""
        try:
            # Load audio file with librosa
            y, sr = librosa.load(file_path, sr=None)
            
            # Extract features
            duration = librosa.get_duration(y=y, sr=sr)
            rms = librosa.feature.rms(y=y)[0].mean()
            spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0].mean()
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0].mean()
            zero_crossing_rate = librosa.feature.zero_crossing_rate(y=y)[0].mean()
            
            # Get spectrum
            spec = np.abs(librosa.stft(y))
            
            return {
                'waveform': y,
                'sample_rate': sr,
                'duration': duration,
                'rms': rms,
                'spectral_centroid': spectral_centroid,
                'spectral_rolloff': spectral_rolloff,
                'zero_crossing_rate': zero_crossing_rate,
                'spectrum': spec,
                'max_amplitude': np.max(np.abs(y)),
                'mean_amplitude': np.mean(np.abs(y))
            }
            
        except Exception as e:
            print(f"Error analyzing {file_path}: {e}")
            return None
    
    @staticmethod
    def compare_files(original_path, converted_path):
        """Compare original and converted audio files"""
        original = AudioAnalyzer.analyze_file(original_path)
        converted = AudioAnalyzer.analyze_file(converted_path)
        
        if not original or not converted:
            return None
        
        # Calculate differences and similarity metrics
        volume_diff_pct = ((converted['rms'] - original['rms']) / original['rms']) * 100 if original['rms'] > 0 else 0
        
        # Resample if needed to compare waveforms directly
        if original['sample_rate'] != converted['sample_rate']:
            original_waveform = librosa.resample(original['waveform'], 
                                                 orig_sr=original['sample_rate'],
                                                 target_sr=converted['sample_rate'])
        else:
            original_waveform = original['waveform']
            
        # Ensure same length for correlation
        min_length = min(len(original_waveform), len(converted['waveform']))
        original_waveform = original_waveform[:min_length]
        converted_waveform = converted['waveform'][:min_length]
        
        # Calculate cross-correlation to measure similarity
        correlation = np.corrcoef(original_waveform, converted_waveform)[0, 1]
        
        # Calculate signal-to-noise ratio (SNR)
        if min_length > 0:
            diff = original_waveform - converted_waveform
            if np.sum(original_waveform**2) > 0:
                snr = 10 * np.log10(np.sum(original_waveform**2) / np.sum(diff**2)) if np.sum(diff**2) > 0 else float('inf')
            else:
                snr = 0
        else:
            snr = 0

        # Spectral comparison
        if original['spectrum'].shape[1] > 0 and converted['spectrum'].shape[1] > 0:
            min_spec_length = min(original['spectrum'].shape[1], converted['spectrum'].shape[1])
            spec_correlation = np.corrcoef(
                np.mean(original['spectrum'][:, :min_spec_length], axis=0),
                np.mean(converted['spectrum'][:, :min_spec_length], axis=0)
            )[0, 1]
        else:
            spec_correlation = 0
            
        return {
            'original': original,
            'converted': converted,
            'volume_diff_pct': volume_diff_pct,
            'correlation': correlation,
            'spectral_correlation': spec_correlation,
            'snr': snr,
            'quality_score': (correlation + spec_correlation + min(1, snr/30)) / 3 * 100 if correlation is not None and spec_correlation is not None else 0
        }


class AudioProcessor(QThread):
    progress_signal = pyqtSignal(str, int)  # file, progress percentage
    complete_signal = pyqtSignal(str, bool, object)  # file, success, analysis_results

    def __init__(self, file_path, output_dir):
        super().__init__()
        self.file_path = file_path
        self.output_dir = output_dir
        self.canceled = False
        self.process = None

    def run(self):
        try:
            filename = os.path.basename(self.file_path)
            self.progress_signal.emit(filename, 10)
            
            # Prepare output file path (preserve directory structure)
            rel_path = os.path.splitext(filename)[0] + ".wav"
            output_path = os.path.join(self.output_dir, rel_path)
            
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Set up ffmpeg command for conversion to 44.1kHz, 16-bit
            ffmpeg_cmd = [
                FFMPEG_PATH,  # Use the detected FFmpeg path
                '-i', self.file_path,
                '-ar', '44100',     # Sample rate: 44.1kHz
                '-sample_fmt', 's16',  # Sample format: 16-bit signed integer
                '-ac', '2',         # Audio channels: stereo
                '-y',               # Overwrite output file if it exists
                output_path
            ]
            
            self.progress_signal.emit(filename, 30)
            
            if self.canceled:
                return
            
            # Start conversion process
            self.process = subprocess.Popen(
                ffmpeg_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True
            )
            
            # Monitor progress
            while self.process.poll() is None:
                if self.canceled:
                    self.process.terminate()
                    return
                
                self.progress_signal.emit(filename, 50)
                time.sleep(0.1)
                
            self.progress_signal.emit(filename, 70)
            
            # Check if process completed successfully
            if self.process.returncode == 0:
                # Analyze audio files and compare
                self.progress_signal.emit(filename, 80)
                try:
                    analysis_results = AudioAnalyzer.compare_files(self.file_path, output_path)
                    self.progress_signal.emit(filename, 90)
                except Exception as e:
                    print(f"Error during analysis: {e}")
                    analysis_results = None
                
                self.progress_signal.emit(filename, 100)
                self.complete_signal.emit(filename, True, analysis_results)
            else:
                error = self.process.stderr.read() if self.process else "Unknown error"
                print(f"Error processing {self.file_path}: {error}")
                self.complete_signal.emit(filename, False, None)
                
        except Exception as e:
            print(f"Error processing {self.file_path}: {e}")
            self.complete_signal.emit(filename, False, None)
    
    def cancel(self):
        self.canceled = True
        if self.process and self.process.poll() is None:
            self.process.terminate()


class ProcessingManager(QObject):
    all_complete_signal = pyqtSignal(dict)  # analysis_results_dict
    
    def __init__(self, max_workers=None):
        super().__init__()
        self.max_workers = max_workers or os.cpu_count()
        self.active_jobs = {}
        self.pending_files = []
        self.output_dir = ""
        self.analysis_results = {}
        self.successful_conversions = 0
        self.failed_conversions = 0
    
    def add_files(self, files, output_dir):
        self.pending_files.extend(files)
        self.output_dir = output_dir
        self.analysis_results = {}
        self.successful_conversions = 0
        self.failed_conversions = 0
        
    def start_processing(self):
        self._process_next_batch()
    
    def _process_next_batch(self):
        # Start as many jobs as we have capacity for
        while len(self.active_jobs) < self.max_workers and self.pending_files:
            file_path = self.pending_files.pop(0)
            processor = AudioProcessor(file_path, self.output_dir)
            filename = os.path.basename(file_path)
            
            processor.progress_signal.connect(self.update_progress)
            processor.complete_signal.connect(self.job_complete)
            
            self.active_jobs[filename] = processor
            processor.start()
            
        # If no jobs are active and no files are pending, we're done
        if not self.active_jobs and not self.pending_files:
            self.all_complete_signal.emit(self.analysis_results)
    
    def update_progress(self, filename, progress):
        # This method will be connected to the UI
        pass
    
    def job_complete(self, filename, success, analysis_results):
        if success:
            self.successful_conversions += 1
            if analysis_results:
                self.analysis_results[filename] = analysis_results
        else:
            self.failed_conversions += 1
            
        if filename in self.active_jobs:
            self.active_jobs[filename].quit()
            self.active_jobs[filename].wait()
            del self.active_jobs[filename]
            
            # Process next batch
            self._process_next_batch()
    
    def cancel_all(self):
        for job in self.active_jobs.values():
            job.cancel()
        self.pending_files.clear()
        

class MatplotlibCanvas(FigureCanvas):
    """Matplotlib canvas for plotting audio visualizations"""
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi, tight_layout=True)
        self.axes = self.fig.add_subplot(111)
        
        FigureCanvas.__init__(self, self.fig)
        self.setParent(parent)
        
        # Fix: Use QSizePolicy directly instead of QApplication.sizePolicy()
        from PyQt5.QtWidgets import QSizePolicy
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.updateGeometry()
        
    def clear(self):
        self.axes.clear()
        self.draw()

class AudioAnalysisDisplay(QWidget):
    """Widget to display audio analysis results"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(400)
        
        # Create layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Create tabs for different visualizations
        self.tabs = QTabWidget()
        self.tabs.setStyleSheet("""
            QTabWidget::pane {
                border: 1px solid #dde1e7;
                background-color: white;
                border-radius: 6px;
            }
            QTabBar::tab {
                background-color: #f0f2f5;
                border: 1px solid #dde1e7;
                border-bottom-color: #dde1e7;
                border-top-left-radius: 6px;
                border-top-right-radius: 6px;
                min-width: 160px;  /* Increased width for tab text */
                height: 36px;      /* Fixed height */
                padding: 4px 8px;  /* Adjusted padding */
                color: #4a5568;
                font-size: 13px;
            }
            QTabBar::tab:selected {
                background-color: white;
                border-bottom-color: white;
                color: #3e64ff;
                font-weight: bold;
            }
            QTabBar::tab:hover {
                background-color: #e6e9ed;
            }
        """)
        
        # Add tab for waveform
        self.waveform_widget = QWidget()
        waveform_layout = QVBoxLayout(self.waveform_widget)
        self.waveform_canvas = MatplotlibCanvas(self.waveform_widget, width=8, height=3)
        waveform_layout.addWidget(self.waveform_canvas)
        self.tabs.addTab(self.waveform_widget, "Waveform Comparison")
        
        # Add tab for spectrum
        self.spectrum_widget = QWidget()
        spectrum_layout = QVBoxLayout(self.spectrum_widget)
        self.spectrum_canvas = MatplotlibCanvas(self.spectrum_widget, width=8, height=3)
        spectrum_layout.addWidget(self.spectrum_canvas)
        self.tabs.addTab(self.spectrum_widget, "Spectrum Analysis")
        
        # Add tab for metrics
        self.metrics_widget = QWidget()
        metrics_layout = QVBoxLayout(self.metrics_widget)
        self.metrics_canvas = MatplotlibCanvas(self.metrics_widget, width=8, height=3)
        metrics_layout.addWidget(self.metrics_canvas)
        self.tabs.addTab(self.metrics_widget, "Quality Metrics")
        
        # Quality summary section
        self.quality_group = QGroupBox("Conversion Quality Summary")
        self.quality_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 1px solid #dde1e7;
                border-radius: 6px;
                margin-top: 1.5ex;
                padding-top: 1.5ex;
                background-color: #f8f9fa;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top center;
                padding: 0 8px;
                color: #3e64ff;
                font-size: 14px;
            }
        """)
        quality_layout = QGridLayout(self.quality_group)
        quality_layout.setContentsMargins(15, 15, 15, 15)
        quality_layout.setSpacing(10)
        
        # Labels for metrics in a grid
        labels = [
            ("Overall Quality:", "100%", "quality_score"),
            ("Volume Difference:", "0%", "volume_diff"),
            ("Waveform Similarity:", "100%", "correlation"),
            ("Spectral Similarity:", "100%", "spectral_corr"),
            ("Signal-to-Noise Ratio:", "Infinity", "snr")
        ]
        
        self.quality_labels = {}
        for i, (label_text, default_value, key) in enumerate(labels):
            label = QLabel(label_text)
            label.setStyleSheet("font-weight: bold; color: #4a5568; font-size: 13px;")
            value_label = QLabel(default_value)
            value_label.setStyleSheet("color: #2d3748; font-size: 13px;")
            quality_layout.addWidget(label, i, 0)
            quality_layout.addWidget(value_label, i, 1)
            self.quality_labels[key] = value_label
            
        # Add sections to main layout
        layout.addWidget(self.tabs, 7)
        layout.addWidget(self.quality_group, 3)
        
        # Initialize empty
        self.clear()
        
    def clear(self):
        """Clear all plots and reset metrics"""
        self.waveform_canvas.clear()
        self.spectrum_canvas.clear()
        self.metrics_canvas.clear()
        
        # Reset metrics
        self.quality_labels["quality_score"].setText("N/A")
        self.quality_labels["volume_diff"].setText("N/A")
        self.quality_labels["correlation"].setText("N/A")
        self.quality_labels["spectral_corr"].setText("N/A")
        self.quality_labels["snr"].setText("N/A")
        
    def update_analysis(self, analysis_results):
        """Update the display with new analysis results"""
        if not analysis_results:
            self.clear()
            return
            
        # Clear previous plots
        self.waveform_canvas.axes.clear()
        self.spectrum_canvas.axes.clear()
        self.metrics_canvas.axes.clear()
        
        # Get data
        original = analysis_results['original']
        converted = analysis_results['converted']
        
        # Plot waveforms
        ax = self.waveform_canvas.axes
        time_orig = np.arange(len(original['waveform'])) / original['sample_rate']
        time_conv = np.arange(len(converted['waveform'])) / converted['sample_rate']
        
        ax.plot(time_orig, original['waveform'], alpha=0.7, label='Original', color='#3e64ff')
        ax.plot(time_conv, converted['waveform'], alpha=0.7, label='Converted', color='#38b2ac')
        ax.set_title('Waveform Comparison', fontsize=12, fontweight='bold', color='#2d3748')
        ax.set_xlabel('Time (s)', fontsize=10, color='#4a5568')
        ax.set_ylabel('Amplitude', fontsize=10, color='#4a5568')
        ax.legend(frameon=True, fancybox=True, shadow=True)
        ax.grid(True, alpha=0.2)
        # Set background color
        ax.set_facecolor('#f8f9fa')
        self.waveform_canvas.draw()
        
        # Plot spectrums
        ax = self.spectrum_canvas.axes
        
        # Get log-scaled spectrum for better visualization
        orig_spec_db = librosa.amplitude_to_db(original['spectrum'], ref=np.max)
        conv_spec_db = librosa.amplitude_to_db(converted['spectrum'], ref=np.max)
        
        # Use colormap for spectrogram
        img1 = librosa.display.specshow(orig_spec_db, x_axis='time', y_axis='log', 
                                       sr=original['sample_rate'], ax=ax, alpha=0.8,
                                       cmap='viridis')
        ax.set_title('Audio Spectrum Visualization', fontsize=12, fontweight='bold', color='#2d3748')
        ax.set_xlabel('Time (s)', fontsize=10, color='#4a5568')
        ax.set_ylabel('Frequency (Hz)', fontsize=10, color='#4a5568')
        # Set background color
        ax.set_facecolor('#f8f9fa')
        self.spectrum_canvas.fig.colorbar(img1, ax=ax, format="%+2.f dB")
        self.spectrum_canvas.draw()
        
        # Plot quality metrics
        ax = self.metrics_canvas.axes
        metrics = [
            ('Quality', analysis_results['quality_score']),
            ('Volume', 100 - abs(analysis_results['volume_diff_pct'])),
            ('Waveform', analysis_results['correlation'] * 100),
            ('Spectrum', analysis_results['spectral_correlation'] * 100),
            ('SNR', min(100, analysis_results['snr'] * 3.33) if analysis_results['snr'] < 30 else 100)
        ]
        
        labels = [m[0] for m in metrics]
        values = [max(0, min(100, m[1])) for m in metrics]  # Clamp between 0 and 100
        
        # Set colors based on values
        colors = []
        for value in values:
            if value >= 95:
                colors.append('#38b2ac')  # Teal for excellent
            elif value >= 85:
                colors.append('#4299e1')  # Blue for good
            elif value >= 75:
                colors.append('#ecc94b')  # Yellow for acceptable
            elif value >= 60:
                colors.append('#ed8936')  # Orange for concerning
            else:
                colors.append('#e53e3e')  # Red for poor
        
        bars = ax.bar(labels, values, color=colors, width=0.6)
        ax.set_ylim(0, 110)
        ax.set_title('Quality Metrics (%)', fontsize=12, fontweight='bold', color='#2d3748')
        ax.grid(True, axis='y', alpha=0.2)
        
        # Set background color
        ax.set_facecolor('#f8f9fa')
        
        # Add value labels on top of bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 2,
                   f"{value:.1f}%", ha='center', va='bottom', fontsize=9, 
                   color='#4a5568', fontweight='bold')
            
        self.metrics_canvas.draw()
        
        # Update metrics text
        self.quality_labels["quality_score"].setText(f"{analysis_results['quality_score']:.1f}%")
        self.quality_labels["volume_diff"].setText(f"{analysis_results['volume_diff_pct']:.2f}%")
        self.quality_labels["correlation"].setText(f"{analysis_results['correlation'] * 100:.1f}%")
        self.quality_labels["spectral_corr"].setText(f"{analysis_results['spectral_correlation'] * 100:.1f}%")
        self.quality_labels["snr"].setText(f"{analysis_results['snr']:.1f} dB")
        
        # Color-code the metrics
        self._set_metric_color(self.quality_labels["quality_score"], analysis_results['quality_score'])
        self._set_metric_color(self.quality_labels["volume_diff"], 
                              100 - abs(analysis_results['volume_diff_pct']), 
                              invert=True)
        self._set_metric_color(self.quality_labels["correlation"], 
                              analysis_results['correlation'] * 100)
        self._set_metric_color(self.quality_labels["spectral_corr"], 
                              analysis_results['spectral_correlation'] * 100)
        self._set_metric_color(self.quality_labels["snr"], 
                              min(100, analysis_results['snr'] * 3.33) if analysis_results['snr'] < 30 else 100)
    
    def _set_metric_color(self, label, value, invert=False):
        """Set color of metric label based on value"""
        display_val = abs(value) if invert else value
        
        if display_val >= 95:
            label.setStyleSheet("color: #38b2ac; font-weight: bold; font-size: 13px;")  # Teal for excellent
        elif display_val >= 85:
            label.setStyleSheet("color: #4299e1; font-weight: bold; font-size: 13px;")  # Blue for good
        elif display_val >= 75:
            label.setStyleSheet("color: #ecc94b; font-weight: bold; font-size: 13px;")  # Yellow for acceptable
        elif display_val >= 60:
            label.setStyleSheet("color: #ed8936; font-weight: bold; font-size: 13px;")  # Orange for concerning
        else:
            label.setStyleSheet("color: #e53e3e; font-weight: bold; font-size: 13px;")  # Red for poor


class LoadingOverlay(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.setAttribute(Qt.WA_TranslucentBackground)
        
        # Create layout
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)
        
        # Add loading indicator container
        loading_container = QWidget()
        loading_container.setStyleSheet("""
            background-color: rgba(45, 55, 72, 0.85);
            border-radius: 15px;
            border: 1px solid rgba(255, 255, 255, 0.2);
        """)
        loading_container.setFixedSize(320, 160)
        
        container_layout = QVBoxLayout(loading_container)
        
        # Add loading indicator (in this case, a simple label)
        self.loading_label = QLabel("Processing...")
        self.loading_label.setAlignment(Qt.AlignCenter)
        self.loading_label.setStyleSheet("""
            color: #FFFFFF;
            font-size: 20px;
            font-weight: bold;
            padding: 20px;
        """)
        
        # Add smaller text beneath
        self.sub_label = QLabel("Converting audio files to CD quality")
        self.sub_label.setAlignment(Qt.AlignCenter)
        self.sub_label.setStyleSheet("""
            color: rgba(255, 255, 255, 0.8);
            font-size: 14px;
        """)
        
        container_layout.addWidget(self.loading_label)
        container_layout.addWidget(self.sub_label)
        layout.addWidget(loading_container)
        layout.setAlignment(Qt.AlignCenter)
        
        # Animation timer
        self.dots = 0
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._animate_dots)
        self.timer.start(500)  # Update every 500ms
        
        self.hide()
    
    def _animate_dots(self):
        self.dots = (self.dots + 1) % 4
        self.loading_label.setText(f"Processing{'.' * self.dots}")
    
    def showEvent(self, event):
        super().showEvent(event)
        self.timer.start()
    
    def hideEvent(self, event):
        super().hideEvent(event)
        self.timer.stop()


class FileListWidget(QListWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setDragDropMode(QAbstractItemView.DragDrop)
        self.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.setAlternatingRowColors(True)
        
        # Set uniform item height to prevent jumping
        self.setStyleSheet("""
            QListWidget {
                background-color: white;
                border: 1px solid #dde1e7;
                border-radius: 8px;
                padding: 8px;
            }
            QListWidget::item {
                padding: 10px;
                margin: 3px 0px;
                border-bottom: 1px solid #e9ecef;
                min-height: 28px;
                font-size: 13px;
            }
            QListWidget::item:selected {
                background-color: #3e64ff;
                color: white;
                border-radius: 6px;
            }
            QListWidget::item:alternate {
                background-color: #f8f9fa;
            }
            QListWidget::item:hover {
                background-color: #edf2f7;
                border-radius: 6px;
            }
        """)
    
    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            super().dragEnterEvent(event)
    
    def dropEvent(self, event: QDropEvent):
        if event.mimeData().hasUrls():
            event.setDropAction(Qt.CopyAction)
            event.accept()
            
            file_paths = []
            for url in event.mimeData().urls():
                if url.isLocalFile():
                    file_path = url.toLocalFile()
                    # Check if it's an audio file based on extension
                    if os.path.isfile(file_path) and file_path.lower().endswith(('.mp3', '.wav', '.ogg', '.flac', '.aac', '.m4a')):
                        file_paths.append(file_path)
            
            if file_paths:
                self.addFiles(file_paths)
        else:
            super().dropEvent(event)
    
    def addFiles(self, file_paths):
        for file_path in file_paths:
            filename = os.path.basename(file_path)
            item = QListWidgetItem(filename)
            item.setData(Qt.UserRole, file_path)
            self.addItem(item)


class AnimatedButton(QPushButton):
    def __init__(self, text, parent=None):
        super().__init__(text, parent)
        self.setStyleSheet("""
            QPushButton {
                background-color: #3e64ff;
                color: white;
                border: none;
                border-radius: 8px;
                padding: 10px 20px;
                font-weight: bold;
                min-width: 120px;
                font-size: 13px;
                transition: background-color 0.3s;
            }
            QPushButton:hover {
                background-color: #2d4cdf;
            }
            QPushButton:pressed {
                background-color: #1a37c7;
            }
            QPushButton:disabled {
                background-color: #d1d5db;
                color: #9ca3af;
            }
        """)
        
        # Set fixed size for consistency
        self.setFixedHeight(42)


class ProgressListItem(QWidget):
    def __init__(self, filename, parent=None):
        super().__init__(parent)
        
        # Set fixed height to prevent jumping
        self.setFixedHeight(60)
        
        layout = QHBoxLayout()
        layout.setContentsMargins(8, 8, 8, 8)
        self.setLayout(layout)
        
        self.filename_label = QLabel(filename)
        self.filename_label.setFixedWidth(250)
        self.filename_label.setStyleSheet("font-weight: normal; color: #2d3748; font-size: 13px;")
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #dde1e7;
                border-radius: 6px;
                background-color: #f8f9fa;
                text-align: center;
                padding: 2px;
                height: 24px;
                font-size: 12px;
            }
            QProgressBar::chunk {
                background-color: #3e64ff;
                border-radius: 5px;
            }
        """)
        
        self.status_label = QLabel("Pending")
        self.status_label.setFixedWidth(100)
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("""
            padding: 4px 10px;
            background-color: #f8f9fa;
            border: 1px solid #dde1e7;
            border-radius: 6px;
            color: #6c757d;
            font-size: 13px;
        """)
        
        layout.addWidget(self.filename_label)
        layout.addWidget(self.progress_bar, 1)
        layout.addWidget(self.status_label)
    
    def update_progress(self, progress):
        self.progress_bar.setValue(progress)
        if progress < 100:
            self.status_label.setText(f"{progress}%")
            if progress > 0:
                self.status_label.setStyleSheet("""
                    padding: 4px 10px;
                    background-color: #e6f6ff;
                    border: 1px solid #b3e0ff;
                    border-radius: 6px;
                    color: #0366d6;
                    font-size: 13px;
                """)
        else:
            self.status_label.setText("Complete")
            self.status_label.setStyleSheet("""
                padding: 4px 10px;
                background-color: #d9f8ef;
                border: 1px solid #b3e7db;
                border-radius: 6px;
                color: #107565;
                font-weight: bold;
                font-size: 13px;
            """)
    
    def set_error(self):
        self.status_label.setText("Error")
        self.status_label.setStyleSheet("""
            padding: 4px 10px;
            background-color: #fee2e2;
            border: 1px solid #fca5a5;
            border-radius: 6px;
            color: #b91c1c;
            font-weight: bold;
            font-size: 13px;
        """)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Track Ready Pro - Audio Converter")
        
        # Fix window size and completely disable resizing
        self.setFixedSize(1150, 780)
        self.setWindowFlags(Qt.Window | Qt.MSWindowsFixedSizeDialogHint | Qt.WindowCloseButtonHint | Qt.WindowMinimizeButtonHint)
        
        # Set application style
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f8f9fa;
            }
            QLabel {
                font-size: 14px;
                color: #2d3748;
            }
            QTabWidget::pane {
                border: 1px solid #dde1e7;
                border-radius: 6px;
                background-color: white;
            }
            QTabBar::tab {
                background-color: #f0f2f5;
                border: 1px solid #dde1e7;
                border-bottom-color: #dde1e7;
                border-top-left-radius: 6px;
                border-top-right-radius: 6px;
                min-width: 120px;
                padding: 10px 16px;
                color: #4a5568;
                font-size: 13px;
            }
            QTabBar::tab:selected {
                background-color: white;
                border-bottom-color: white;
                color: #3e64ff;
                font-weight: bold;
            }
            QTabBar::tab:hover {
                background-color: #e6e9ed;
            }
            QScrollBar:vertical {
                border: none;
                background: #f8f9fa;
                width: 10px;
                margin: 0px;
            }
            QScrollBar::handle:vertical {
                background: #c1c9d8;
                min-height: 20px;
                border-radius: 5px;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                border: none;
                background: none;
                height: 0px;
            }
            QScrollBar:horizontal {
                border: none;
                background: #f8f9fa;
                height: 10px;
                margin: 0px;
            }
            QScrollBar::handle:horizontal {
                background: #c1c9d8;
                min-width: 20px;
                border-radius: 5px;
            }
            QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
                border: none;
                background: none;
                width: 0px;
            }
        """)
        
        # Initialize variables
        self.file_list = []
        self.output_directory = ""
        self.processing_manager = ProcessingManager()
        self.processing_manager.update_progress = self.update_file_progress
        self.progress_items = {}
        self.current_analysis_file = None
        
        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(25, 25, 25, 25)
        main_layout.setSpacing(20)
        
        # Header with gradient background
        header_widget = QWidget()
        header_widget.setStyleSheet("""
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #2c3e50, stop:1 #3e64ff);
            border-radius: 10px;
            margin: 0px;
            padding: 15px;
        """)
        header_layout = QVBoxLayout(header_widget)
        header_layout.setContentsMargins(20, 22, 20, 22)
        
        header_label = QLabel("Track Ready Pro")
        header_label.setStyleSheet("font-size: 30px; font-weight: bold; color: white; margin-bottom: 5px;")
        header_label.setAlignment(Qt.AlignCenter)
        header_layout.addWidget(header_label)
        
        subtitle_label = QLabel("Audio Converter with Quality Analysis")
        subtitle_label.setStyleSheet("font-size: 16px; color: rgba(255, 255, 255, 0.85);")
        subtitle_label.setAlignment(Qt.AlignCenter)
        header_layout.addWidget(subtitle_label)
        
        main_layout.addWidget(header_widget)
        
        # Main tabs for workflow
        self.main_tabs = QTabWidget()
        
        # Convert tab - with file selection and conversion
        convert_tab = QWidget()
        convert_layout = QVBoxLayout(convert_tab)
        convert_layout.setContentsMargins(15, 20, 15, 15)
        
        # Splitter for file list and progress
        splitter = QSplitter(Qt.Horizontal)
        
        # Left side - File selection
        file_widget = QWidget()
        file_layout = QVBoxLayout(file_widget)
        file_layout.setContentsMargins(0, 0, 0, 0)
        
        # File header section with icon
        file_header_layout = QHBoxLayout()
        file_header = QLabel("Input Files")
        file_header.setStyleSheet("font-weight: bold; font-size: 16px; color: #2d3748;")
        
        # Info section
        info_label = QLabel("Drag & drop audio files here or use the buttons below")
        info_label.setStyleSheet("color: #718096; font-size: 13px; font-style: italic;")
        
        file_layout.addWidget(file_header)
        file_layout.addWidget(info_label)
        file_layout.addSpacing(5)
        
        # File list
        self.file_list_widget = FileListWidget()
        file_layout.addWidget(self.file_list_widget, 1)
        
        # Buttons for file handling
        file_buttons_layout = QHBoxLayout()
        file_buttons_layout.setSpacing(15)
        
        self.add_files_btn = AnimatedButton("Add Files")
        self.add_files_btn.clicked.connect(self.add_files)
        file_buttons_layout.addWidget(self.add_files_btn)
        
        self.clear_files_btn = AnimatedButton("Clear List")
        self.clear_files_btn.clicked.connect(self.clear_files)
        self.clear_files_btn.setStyleSheet("""
            QPushButton {
                background-color: #e2e8f0;
                color: #4a5568;
                border: none;
                border-radius: 8px;
                padding: 10px 20px;
                font-weight: bold;
                min-width: 120px;
                font-size: 13px;
            }
            QPushButton:hover {
                background-color: #cbd5e1;
            }
            QPushButton:pressed {
                background-color: #94a3b8;
            }
            QPushButton:disabled {
                background-color: #f1f5f9;
                color: #9ca3af;
            }
        """)
        file_buttons_layout.addWidget(self.clear_files_btn)
        
        file_layout.addLayout(file_buttons_layout)
        file_layout.addSpacing(10)
        
        # Output directory selection
        output_dir_container = QGroupBox("Output Settings")
        output_dir_container.setStyleSheet("""
            QGroupBox {
                font-size: 15px;
                font-weight: bold;
                color: #2d3748;
                border: 1px solid #dde1e7;
                border-radius: 8px;
                margin-top: 1ex;
                padding: 12px;
                background-color: #f8fafc;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
        """)
        
        # Use a vertical layout instead of a form layout
        output_box_layout = QVBoxLayout(output_dir_container)
        output_box_layout.setContentsMargins(15, 25, 15, 15)
        output_box_layout.setSpacing(12)
    
        # Create a separate horizontal layout for the directory selector
        dir_select_layout = QHBoxLayout()
        dir_select_layout.setSpacing(15)
    
        # Add a label
        dir_label = QLabel("Output Directory:")
        dir_label.setStyleSheet("font-weight: bold; color: #4a5568; font-size: 13px;")
    
        # Configure the line edit
        self.output_dir_edit = QLineEdit()
        self.output_dir_edit.setReadOnly(True)
        self.output_dir_edit.setPlaceholderText("Select output directory...")
        self.output_dir_edit.setFixedHeight(42)  # Match button height
        self.output_dir_edit.setStyleSheet("""
            QLineEdit {
                padding: 8px 12px;
                border: 1px solid #dde1e7;
                border-radius: 6px;
                background-color: white;
                font-size: 13px;
            }
        """)
    
        # Configure the browse button
        output_dir_btn = AnimatedButton("Browse...")
        output_dir_btn.clicked.connect(self.select_output_directory)
        output_dir_btn.setFixedWidth(120)
    
        # Add widgets to horizontal layout
        dir_select_layout.addWidget(dir_label)
        dir_select_layout.addWidget(self.output_dir_edit, 1)  # 1 = stretch factor
        dir_select_layout.addWidget(output_dir_btn)
    
        # Add horizontal layout to main container
        output_box_layout.addLayout(dir_select_layout)
        
        file_layout.addWidget(output_dir_container)
        
        # Right side - Progress
        progress_widget = QWidget()
        progress_layout = QVBoxLayout(progress_widget)
        progress_layout.setContentsMargins(0, 0, 0, 0)
        
        progress_header = QLabel("Conversion Progress")
        progress_header.setStyleSheet("font-weight: bold; font-size: 16px; color: #2d3748;")
        progress_layout.addWidget(progress_header)
        
        progress_info = QLabel("Select files for detailed analysis after conversion")
        progress_info.setStyleSheet("color: #718096; font-size: 13px; font-style: italic;")
        progress_layout.addWidget(progress_info)
        progress_layout.addSpacing(5)
        
        # Progress list
        self.progress_list = QListWidget()
        self.progress_list.setStyleSheet("""
            QListWidget {
                background-color: #ffffff;
                border: 1px solid #dde1e7;
                border-radius: 8px;
            }
        """)
        self.progress_list.itemClicked.connect(self.show_file_analysis)
        progress_layout.addWidget(self.progress_list, 1)
        
        # Add widgets to splitter
        splitter.addWidget(file_widget)
        splitter.addWidget(progress_widget)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)
        
        convert_layout.addWidget(splitter, 1)
        
        # Convert button with enhanced styling
        convert_container = QWidget()
        convert_layout_btn = QVBoxLayout(convert_container)
        convert_layout_btn.setContentsMargins(10, 15, 10, 5)
        
        self.convert_btn = QPushButton("Convert to CD Quality")
        self.convert_btn.clicked.connect(self.start_conversion)
        self.convert_btn.setMinimumHeight(50)
        self.convert_btn.setEnabled(False)
        self.convert_btn.setStyleSheet("""
            QPushButton {
                background-color: #38b2ac;
                color: white;
                border: none;
                border-radius: 8px;
                padding: 12px 24px;
                font-weight: bold;
                font-size: 16px;
                min-width: 200px;
            }
            QPushButton:hover {
                background-color: #319795;
            }
            QPushButton:pressed {
                background-color: #2c7a7b;
            }
            QPushButton:disabled {
                background-color: #cbd5e0;
                color: #718096;
            }
        """)
        
        convert_layout_btn.addWidget(self.convert_btn)
        convert_layout_btn.setAlignment(Qt.AlignCenter)
        convert_layout.addWidget(convert_container)
        
        # Analysis tab - to view detailed analysis of converted files
        analysis_tab = QWidget()
        analysis_layout = QVBoxLayout(analysis_tab)
        analysis_layout.setContentsMargins(15, 20, 15, 15)
        
        # File selector section
        file_selector_widget = QWidget()
        file_selector_layout = QVBoxLayout(file_selector_widget)
        file_selector_layout.setContentsMargins(0, 0, 0, 15)
        
        file_selector_label = QLabel("Select converted file to analyze:")
        file_selector_label.setStyleSheet("font-weight: bold; font-size: 16px; color: #2d3748; padding-bottom: 5px;")
        file_selector_layout.addWidget(file_selector_label)
        
        self.file_selector_combo = QListWidget()
        self.file_selector_combo.setMaximumHeight(150)
        self.file_selector_combo.setAlternatingRowColors(True)
        self.file_selector_combo.setStyleSheet("""
            QListWidget {
                background-color: white;
                border: 1px solid #dde1e7;
                border-radius: 8px;
                padding: 8px;
            }
            QListWidget::item {
                padding: 8px;
                margin: 2px 0px;
                border-bottom: 1px solid #e9ecef;
                min-height: 24px;
                font-size: 13px;
            }
            QListWidget::item:selected {
                background-color: #3e64ff;
                color: white;
                border-radius: 6px;
            }
            QListWidget::item:alternate {
                background-color: #f8f9fa;
            }
            QListWidget::item:hover {
                background-color: #edf2f7;
                border-radius: 6px;
            }
        """)
        self.file_selector_combo.itemClicked.connect(self.show_file_analysis_from_tab)
        file_selector_layout.addWidget(self.file_selector_combo)
        
        analysis_layout.addWidget(file_selector_widget)
        
        # Analysis display
        self.analysis_display = AudioAnalysisDisplay()
        scroll_area = QScrollArea()
        scroll_area.setWidget(self.analysis_display)
        scroll_area.setWidgetResizable(True)
        scroll_area.setStyleSheet("""
            QScrollArea {
                border: none;
                background-color: white;
                border-radius: 8px;
            }
        """)
        analysis_layout.addWidget(scroll_area)
        
        # Add tabs to main tabs
        self.main_tabs.addTab(convert_tab, "Convert")
        self.main_tabs.addTab(analysis_tab, "Analysis")
        
        main_layout.addWidget(self.main_tabs)
        
        # Create and style the status bar
        self.statusBar().setStyleSheet("""
            QStatusBar {
                background-color: #f8f9fa;
                color: #4a5568;
                border-top: 1px solid #dde1e7;
                padding: 8px;
                font-size: 13px;
            }
        """)
        self.statusBar().showMessage("Ready to convert audio files")
        
        # Create and hide the loading overlay
        self.loading_overlay = LoadingOverlay(self)
        self.loading_overlay.hide()
        self.loading_overlay.setGeometry(self.rect())
        
        # Connect signals
        self.processing_manager.all_complete_signal.connect(self.conversion_complete)
        
        # Credit label with better styling
        credit_label = QLabel("Created with ♪ by LOLITEMAULTES | Enhanced for optimal CD audio quality")
        credit_label.setStyleSheet("""
            color: #718096; 
            font-size: 12px;
            padding: 6px;
            background-color: rgba(248, 249, 250, 0.7);
            border-radius: 4px;
        """)
        credit_label.setAlignment(Qt.AlignRight)
        main_layout.addWidget(credit_label)
    
    def add_files(self):
        file_dialog = QFileDialog()
        file_dialog.setFileMode(QFileDialog.ExistingFiles)
        file_dialog.setNameFilter("Audio Files (*.mp3 *.wav *.ogg *.flac *.aac *.m4a)")
        
        if file_dialog.exec_():
            file_paths = file_dialog.selectedFiles()
            self.file_list_widget.addFiles(file_paths)
            self.update_convert_button()
            self.statusBar().showMessage(f"Added {len(file_paths)} file(s)")
    
    def clear_files(self):
        self.file_list_widget.clear()
        self.update_convert_button()
        self.statusBar().showMessage("File list cleared")
    
    def select_output_directory(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if directory:
            self.output_directory = directory
            self.output_dir_edit.setText(directory)
            self.update_convert_button()
            self.statusBar().showMessage(f"Output directory set: {directory}")
    
    def update_convert_button(self):
        # Make sure we're using a boolean value
        can_convert = (self.file_list_widget.count() > 0) and bool(self.output_directory)
        self.convert_btn.setEnabled(can_convert)
        
        # Update button text based on state
        if can_convert:
            file_count = self.file_list_widget.count()
            self.convert_btn.setText(f"Convert {file_count} File{'s' if file_count > 1 else ''}")
        else:
            self.convert_btn.setText("Convert to CD Quality")
    
    def start_conversion(self):
        # Disable UI during conversion
        self.convert_btn.setEnabled(False)
        self.add_files_btn.setEnabled(False)
        self.clear_files_btn.setEnabled(False)
        
        # Show loading indicator
        self.loading_overlay.show()
        self.statusBar().showMessage("Converting audio files...")
        
        # Clear progress list
        self.progress_list.clear()
        self.progress_items = {}
        
        # Clear file selector for analysis tab
        self.file_selector_combo.clear()
        
        # Collect all files
        file_paths = []
        for i in range(self.file_list_widget.count()):
            item = self.file_list_widget.item(i)
            file_path = item.data(Qt.UserRole)
            file_paths.append(file_path)
            
            # Add to progress list
            filename = os.path.basename(file_path)
            list_item = QListWidgetItem()
            self.progress_list.addItem(list_item)
            
            progress_widget = ProgressListItem(filename)
            list_item.setSizeHint(progress_widget.sizeHint())
            self.progress_list.setItemWidget(list_item, progress_widget)
            
            self.progress_items[filename] = progress_widget
        
        # Start processing
        self.processing_manager.add_files(file_paths, self.output_directory)
        self.processing_manager.start_processing()
    
    def update_file_progress(self, filename, progress):
        if filename in self.progress_items:
            self.progress_items[filename].update_progress(progress)
    
    def show_file_analysis(self, item):
        # Get filename from the selected item
        filename = None
        for i in range(self.progress_list.count()):
            if self.progress_list.item(i) == item:
                # Get the filename from ProgressListItem
                progress_item = self.progress_list.itemWidget(item)
                if progress_item:
                    filename = progress_item.filename_label.text()
                break
                
        if filename and filename in self.processing_manager.analysis_results:
            self.current_analysis_file = filename
            analysis_results = self.processing_manager.analysis_results[filename]
            
            # Update analysis display
            self.analysis_display.update_analysis(analysis_results)
            
            # Switch to analysis tab
            self.main_tabs.setCurrentIndex(1)
            
            # Highlight the selected file in the file selector
            for i in range(self.file_selector_combo.count()):
                if self.file_selector_combo.item(i).text() == filename:
                    self.file_selector_combo.setCurrentItem(self.file_selector_combo.item(i))
                    break
            
            self.statusBar().showMessage(f"Analyzing: {filename}")
    
    def show_file_analysis_from_tab(self, item):
        filename = item.text()
        if filename in self.processing_manager.analysis_results:
            self.current_analysis_file = filename
            analysis_results = self.processing_manager.analysis_results[filename]
            
            # Update analysis display
            self.analysis_display.update_analysis(analysis_results)
            self.statusBar().showMessage(f"Analyzing: {filename}")
    
    def conversion_complete(self, analysis_results):
        # Hide loading overlay
        self.loading_overlay.hide()
        
        # Re-enable UI
        self.convert_btn.setEnabled(True)
        self.add_files_btn.setEnabled(True)
        self.clear_files_btn.setEnabled(True)
        
        # Update file selector for analysis tab
        self.file_selector_combo.clear()
        for filename in analysis_results.keys():
            self.file_selector_combo.addItem(QListWidgetItem(filename))
        
        # Calculate overall statistics
        total_files = self.processing_manager.successful_conversions + self.processing_manager.failed_conversions
        successful = self.processing_manager.successful_conversions
        
        # Calculate average quality score
        avg_quality = 0
        problem_files = []
        
        for filename, results in analysis_results.items():
            quality_score = results.get('quality_score', 0)
            avg_quality += quality_score
            
            # Track files with potential issues
            if quality_score < 90:
                problem_level = "minor"
                if quality_score < 80:
                    problem_level = "significant" 
                if quality_score < 70:
                    problem_level = "severe"
                    
                problem_files.append((filename, quality_score, problem_level))
        
        if analysis_results:
            avg_quality /= len(analysis_results)
        
        self.statusBar().showMessage(f"Conversion complete! Average quality: {avg_quality:.1f}%")
        
        # Create detailed message for the popup
        message = f"<h3>Conversion Results</h3>"
        message += f"<p><b>Files Processed:</b> {total_files}<br>"
        message += f"<b>Successfully Converted:</b> {successful}<br>"
        message += f"<b>Failed:</b> {self.processing_manager.failed_conversions}</p>"
        
        if analysis_results:
            # Set color based on average quality
            quality_color = "#38b2ac"  # Default teal (excellent)
            if avg_quality < 95:
                quality_color = "#4299e1"  # Blue (good)
            if avg_quality < 85:
                quality_color = "#ecc94b"  # Yellow (acceptable)
            if avg_quality < 75:
                quality_color = "#ed8936"  # Orange (concerning)
            if avg_quality < 65:
                quality_color = "#e53e3e"  # Red (poor)
                
            message += f"<p><b>Average Quality Score:</b> <span style='color:{quality_color};font-weight:bold;'>{avg_quality:.1f}%</span></p>"
            
            if problem_files:
                message += "<p><b>Files with Potential Issues:</b></p><ul>"
                for filename, score, level in problem_files:
                    # Set color based on problem level
                    level_color = "#4299e1"  # Blue for minor
                    if level == "significant":
                        level_color = "#ecc94b"  # Yellow for significant
                    if level == "severe":
                        level_color = "#e53e3e"  # Red for severe
                        
                    message += f"<li>{filename}: <span style='color:{level_color};font-weight:bold;'>{score:.1f}% - {level} concerns</span></li>"
                message += "</ul>"
                
                message += "<p>Click on any file in the progress list to view detailed analysis.</p>"
            else:
                message += "<p style='color:#38b2ac;font-weight:bold;'>All conversions completed with excellent quality!</p>"
        
        message += f"<p>Files saved to: <span style='font-family:monospace;background-color:#f0f2f5;padding:2px 5px;border-radius:3px;'>{self.output_directory}</span></p>"
        
        # Create a styled message box
        msg_box = QMessageBox(self)
        msg_box.setWindowTitle("Conversion Complete")
        msg_box.setText(message)
        msg_box.setIcon(QMessageBox.Information)
        msg_box.setStyleSheet("""
            QMessageBox {
                background-color: white;
                font-size: 14px;
            }
            QLabel {
                min-width: 500px;
            }
            QPushButton {
                background-color: #3e64ff;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 8px 16px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #2d4cdf;
            }
            QPushButton:pressed {
                background-color: #1a37c7;
            }
        """)
        
        # Add an "Open Folder" button
        open_folder_button = msg_box.addButton("Open Output Folder", QMessageBox.ActionRole)
        ok_button = msg_box.addButton(QMessageBox.Ok)
        ok_button.setDefault(True)
        
        result = msg_box.exec_()
        
        # Handle button clicks
        if msg_box.clickedButton() == open_folder_button:
            # Open the output folder in file explorer
            try:
                if sys.platform == 'win32':
                    os.startfile(self.output_directory)
                elif sys.platform == 'darwin':  # macOS
                    subprocess.call(['open', self.output_directory])
                else:  # Linux
                    subprocess.call(['xdg-open', self.output_directory])
            except Exception as e:
                print(f"Error opening folder: {e}")
    
# Add this class to allow directly running the conversion without GUI
class CommandLineConverter:
    def __init__(self):
        self.total_files = 0
        self.processed_files = 0
    
    def convert_files(self, input_paths, output_dir):
        """Convert audio files from input paths to output directory"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Collect all files
        files_to_process = []
        for path in input_paths:
            if os.path.isdir(path):
                for root, _, files in os.walk(path):
                    for file in files:
                        if file.lower().endswith(('.mp3', '.wav', '.ogg', '.flac', '.aac', '.m4a')):
                            files_to_process.append(os.path.join(root, file))
            elif os.path.isfile(path) and path.lower().endswith(('.mp3', '.wav', '.ogg', '.flac', '.aac', '.m4a')):
                files_to_process.append(path)
        
        self.total_files = len(files_to_process)
        print(f"\n=== Track Ready Pro CLI ===")
        print(f"Found {self.total_files} audio files to process")
        print(f"Converting to 44.1kHz, 16-bit CD quality audio...")
        print(f"Output directory: {output_dir}\n")
        
        # Create threads for processing
        threads = []
        max_threads = os.cpu_count() or 4
        
        # Process files in batches
        for i in range(0, len(files_to_process), max_threads):
            batch = files_to_process[i:i+max_threads]
            current_threads = []
            
            for file_path in batch:
                thread = threading.Thread(
                    target=self._convert_file,
                    args=(file_path, output_dir)
                )
                current_threads.append(thread)
                thread.start()
            
            # Wait for all threads in batch to complete
            for thread in current_threads:
                thread.join()
            
            threads.extend(current_threads)
        
        print(f"\n=== Conversion Complete ===")
        print(f"All {self.total_files} files have been converted to CD quality!")
        print(f"Output directory: {output_dir}")
        return True
    
    def _convert_file(self, file_path, output_dir):
        """Convert a single audio file using FFmpeg"""
        try:
            # Create output file path (preserve relative directory structure)
            rel_path = os.path.basename(file_path)
            output_path = os.path.join(output_dir, os.path.splitext(rel_path)[0] + ".wav")
            
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Build FFmpeg command
            ffmpeg_cmd = [
                FFMPEG_PATH,  # Use the detected FFmpeg path
                '-i', file_path,
                '-ar', '44100',     # Sample rate: 44.1kHz
                '-sample_fmt', 's16',  # Sample format: 16-bit
                '-ac', '2',         # Audio channels: stereo
                '-y',               # Overwrite output file if it exists
                output_path
            ]
            
            # Run FFmpeg
            process = subprocess.run(
                ffmpeg_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True
            )
            
            # Check result
            if process.returncode == 0:
                self.processed_files += 1
                
                # Analyze the conversion
                try:
                    analysis = AudioAnalyzer.compare_files(file_path, output_path)
                    quality_score = analysis['quality_score']
                    volume_diff = analysis['volume_diff_pct']
                    
                    quality_text = "Excellent"
                    quality_symbol = "✓"
                    if quality_score < 95:
                        quality_text = "Good"
                        quality_symbol = "✓"
                    if quality_score < 85:
                        quality_text = "Acceptable"
                        quality_symbol = "⚠"
                    if quality_score < 75:
                        quality_text = "Mediocre"
                        quality_symbol = "⚠"
                    if quality_score < 65:
                        quality_text = "Poor"
                        quality_symbol = "✗"
                    
                    print(f"[{quality_symbol}] ({self.processed_files}/{self.total_files}): {os.path.basename(file_path)} | Quality: {quality_score:.1f}% ({quality_text})")
                    
                    # Show warnings if issues detected
                    if abs(volume_diff) > 3:
                        print(f"    ⚠️ Warning: Volume difference of {volume_diff:.1f}% detected")
                    if analysis['correlation'] < 0.9:
                        print(f"    ⚠️ Warning: Possible waveform distortion detected (similarity: {analysis['correlation']*100:.1f}%)")
                    
                except Exception as e:
                    print(f"[!] ({self.processed_files}/{self.total_files}): {os.path.basename(file_path)} | Analysis failed: {str(e)}")
                
            else:
                print(f"[✗] Error converting {file_path}: {process.stderr}")
                
        except Exception as e:
            print(f"[✗] Failed to convert {file_path}: {str(e)}")


if __name__ == "__main__":
    # Check if running in command line mode
    if len(sys.argv) > 1 and sys.argv[1] == "--cli":
        if len(sys.argv) < 4:
            print("Usage: python track-ready-pro.py --cli <input_folder_or_files> <output_folder>")
            print("Example: python track-ready-pro.py --cli ./music output_folder")
            sys.exit(1)
            
        input_paths = sys.argv[2:-1]
        output_dir = sys.argv[-1]
        
        converter = CommandLineConverter()
        converter.convert_files(input_paths, output_dir)
        
    else:
        # Print environment information
        print(f"Track Ready Pro - Audio Converter v1.0.0")
        print(f"Python version: {sys.version}")
        print(f"Running script from: {os.path.abspath(__file__)}")
        
        # Create and show GUI application
        app = QApplication(sys.argv)
        app.setStyle("Fusion")  # Use Fusion style for a modern look
        
        # Apply palette for nicer colors
        palette = QPalette()
        palette.setColor(QPalette.Window, QColor(248, 250, 252))
        palette.setColor(QPalette.WindowText, QColor(45, 55, 72))
        palette.setColor(QPalette.Base, QColor(255, 255, 255))
        palette.setColor(QPalette.AlternateBase, QColor(248, 249, 250))
        palette.setColor(QPalette.ToolTipBase, QColor(255, 255, 255))
        palette.setColor(QPalette.ToolTipText, QColor(45, 55, 72))
        palette.setColor(QPalette.Text, QColor(45, 55, 72))
        palette.setColor(QPalette.Button, QColor(240, 242, 245))
        palette.setColor(QPalette.ButtonText, QColor(45, 55, 72))
        palette.setColor(QPalette.Link, QColor(62, 100, 255))
        palette.setColor(QPalette.Highlight, QColor(62, 100, 255))
        palette.setColor(QPalette.HighlightedText, QColor(255, 255, 255))
        app.setPalette(palette)
        
        # Set application font
        font = QFont("Segoe UI", 9)  # Use system font for better compatibility
        app.setFont(font)
        
        window = MainWindow()
        window.show()
        
        sys.exit(app.exec_())