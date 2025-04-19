![trackreadypro](https://github.com/user-attachments/assets/af82adbf-f9b7-49e4-a464-c898a90de166)
---

Professional audio converter and analysis tool for preparing music tracks for digital distribution and CD-quality release.

## Features

- Convert audio files to high-quality format (44.1kHz, 16-bit WAV)
- Analyze audio quality with detailed visualizations
- Multi-threaded batch processing for efficient workflow
- Track waveform and spectral analysis
- Comprehensive conversion quality metrics
- Modern, user-friendly interface

![image](https://github.com/user-attachments/assets/3df55a39-60a0-44c1-a2cd-215d22ba7637)

## Getting Started

### Prerequisites

- Python 3.6 or higher
- PyQt5
- Librosa
- Matplotlib
- NumPy
- SciPy
- FFmpeg

### Installation

1. Clone this repository:
   ```
   git clone https://github.com/lolitemaultes/track-ready-pro.git
   ```

2. Install required packages:
   ```
   pip install -r requirements.txt
   ```

3. Make sure FFmpeg is installed on your system or place it in the application directory.

## Usage

Run the application:

```
python track-ready-pro.py
```

For command-line batch processing:

```
python track-ready-pro.py --cli <input_folder_or_files> <output_folder>
```

### Using the Application

1. **Add audio files** using the "Add Files" button or drag and drop
2. **Select an output directory** by clicking "Browse..."
3. **Convert files** by clicking the "Convert to Release Quality" button
4. **Analyze results** by selecting a file from the conversion list
5. Review quality metrics and visualizations in the Analysis tab

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Original code by LOLITEMAULTES
- Enhanced for professional audio production and digital distribution
