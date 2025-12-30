from flask import Flask, request, jsonify, send_file, Response
from flask_cors import CORS
import yt_dlp
import os
import json
import re
import threading
import queue
from datetime import datetime, timedelta
from pathlib import Path
import mimetypes
import hashlib
import time
import sys
import logging
from urllib.parse import urlparse, parse_qs
from concurrent.futures import ThreadPoolExecutor
import signal
import atexit
import shutil

# --- লগিং কনফিগারেশন (শুধুমাত্র টার্মিনালে দেখাবে, ফাইল তৈরি করবে না) ---
logging.basicConfig(
    level=logging.INFO, # এখন টার্মিনালে তথ্য দেখা যাবে
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout) # এটি শুধু কনসোলে দেখাবে
        # logging.FileHandler('youtube_downloader.log') <--- এই লাইনটি আমরা সরিয়ে দিয়েছি
    ]
)
logger = logging.getLogger(__name__)

# --- Flask অ্যাপ ডিফাইন করা (এই অংশটি আপনার কোড থেকে মিসিং ছিল) ---
app = Flask(__name__, static_folder='static', static_url_path='/static')
CORS(app, resources={r"/*": {"origins": "*"}})

# --- Configuration ---
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB
app.config['DOWNLOAD_FOLDER'] = Path('temp_downloads')
app.config['MAX_CONCURRENT_DOWNLOADS'] = 3
app.config['CACHE_DURATION'] = 300  # 5 minutes cache
app.config['CHUNK_SIZE'] = 8192  # 8KB chunks for streaming
app.config['DELETE_AFTER_DOWNLOAD'] = True  # Auto delete after sending to user

# ফোল্ডার তৈরি নিশ্চিত করা
app.config['DOWNLOAD_FOLDER'].mkdir(exist_ok=True)

# Global state
download_tasks = {}
download_queue = queue.Queue()
download_history = []
executor = ThreadPoolExecutor(max_workers=app.config['MAX_CONCURRENT_DOWNLOADS'])


class YouTubeValidator:
    """Enhanced YouTube URL validator"""
    
    YOUTUBE_DOMAINS = [
        'youtube.com',
        'www.youtube.com',
        'm.youtube.com',
        'youtu.be',
        'www.youtu.be',
        'youtube-nocookie.com',
        'music.youtube.com',
    ]
    
    @staticmethod
    def normalize_url(url: str) -> str:
        """Normalize YouTube URL to standard format"""
        try:
            url = url.strip()
            
            # Remove extra spaces and newlines
            url = ' '.join(url.split())
            
            # Check if it's a mobile URL
            url = url.replace('m.youtube.com', 'www.youtube.com')
            
            # Add https if missing
            if not url.startswith(('http://', 'https://')):
                url = 'https://' + url
                
            return url
        except Exception as e:
            logger.error(f"URL normalization error: {e}")
            return url
    
    @staticmethod
    def is_valid_youtube_url(url: str) -> bool:
        """Check if URL is a valid YouTube URL with comprehensive patterns"""
        try:
            url = YouTubeValidator.normalize_url(url)
            parsed = urlparse(url)
            
            # Check domain
            domain = parsed.netloc.lower()
            is_youtube_domain = any(
                yt_domain in domain for yt_domain in YouTubeValidator.YOUTUBE_DOMAINS
            )
            
            if not is_youtube_domain:
                return False
            
            # Check path patterns
            path = parsed.path.lower()
            
            # Video patterns
            video_patterns = [
                r'^/watch$',          # /watch?v=VIDEO_ID
                r'^/v/',              # /v/VIDEO_ID
                r'^/embed/',          # /embed/VIDEO_ID
                r'^/shorts/',         # /shorts/VIDEO_ID
                r'^/live/',           # /live/VIDEO_ID
            ]
            
            # Playlist patterns
            playlist_patterns = [
                r'^/playlist$',       # /playlist?list=PLAYLIST_ID
                r'^/watch$',          # /watch?v=VIDEO_ID&list=PLAYLIST_ID
            ]
            
            # Check for video patterns in path
            is_video_path = any(re.match(pattern, path) for pattern in video_patterns)
            is_playlist_path = any(re.match(pattern, path) for pattern in playlist_patterns)
            
            # Check query parameters
            query = parse_qs(parsed.query)
            
            # Video ID patterns in query
            has_video_id = 'v' in query and len(query['v'][0]) >= 11
            
            # Short URL pattern (youtu.be)
            if domain in ['youtu.be', 'www.youtu.be']:
                video_id = path.strip('/')
                return len(video_id) >= 11
            
            # Playlist ID pattern
            has_playlist_id = 'list' in query
            
            # Return True if it's a valid video or playlist
            return (
                (is_video_path and has_video_id) or
                (is_playlist_path and has_playlist_id) or
                has_video_id or
                has_playlist_id or
                '/shorts/' in path or
                '/embed/' in path
            )
            
        except Exception as e:
            logger.error(f"URL validation error for '{url}': {e}")
            return False
    
    @staticmethod
    def extract_video_id(url: str) -> str:
        """Extract video ID from URL"""
        try:
            url = YouTubeValidator.normalize_url(url)
            parsed = urlparse(url)
            
            # Short URL format (youtu.be)
            if parsed.netloc in ['youtu.be', 'www.youtu.be']:
                return parsed.path.strip('/')
            
            # Standard YouTube URL
            query = parse_qs(parsed.query)
            if 'v' in query:
                return query['v'][0]
            
            # Embed URL
            if '/embed/' in parsed.path:
                return parsed.path.split('/embed/')[1].split('?')[0]
            
            return ''
        except Exception:
            return ''


class ProgressHook:
    """Handle yt-dlp progress with ANSI escape code removal"""
    
    @staticmethod
    def remove_ansi_codes(text):
        """Remove ANSI escape codes from text"""
        if not text:
            return text
        ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
        return ansi_escape.sub('', text)
    
    @staticmethod
    def clean_progress_data(d):
        """Clean progress data from yt-dlp"""
        cleaned = {}
        
        # Handle percent
        percent_str = d.get('_percent_str', '0%')
        percent_str = ProgressHook.remove_ansi_codes(percent_str)
        try:
            cleaned['percent'] = float(percent_str.strip('%').strip())
        except:
            cleaned['percent'] = 0.0
        
        # Handle other fields
        fields_to_clean = ['_speed_str', '_eta_str', '_total_bytes_str', 'downloaded_bytes_str']
        for field in fields_to_clean:
            value = d.get(field, '')
            if value:
                cleaned[field[1:] if field.startswith('_') else field] = ProgressHook.remove_ansi_codes(str(value))
            else:
                cleaned[field[1:] if field.startswith('_') else field] = ''
        
        # Add status
        cleaned['status'] = d.get('status', '')
        
        return cleaned


class SmartDownloadManager:
    """Manages downloads with smart cleanup"""
    
    def __init__(self, download_folder='temp_downloads', max_age_minutes=10):
        self.download_dir = Path(download_folder)
        self.max_age_minutes = max_age_minutes  # Reduced to 10 minutes for temp files
        self.download_dir.mkdir(exist_ok=True)
        
        logger.info(f"Download manager initialized. Download folder: {self.download_dir}")

    def get_unique_filename(self, title: str, extension: str, task_id: str = None) -> Path:
        """Generate unique filename with task ID"""
        # Clean the title for filename
        safe_title = re.sub(r'[^\w\s-]', '', title)[:50]
        safe_title = re.sub(r'[-\s]+', '_', safe_title)
        
        # Add task ID and timestamp for uniqueness
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        task_suffix = f"_{task_id}" if task_id else ""
        filename = f"{timestamp}{task_suffix}_{safe_title}.{extension}"
        
        return self.download_dir / filename
    
    def cleanup_old_files(self, max_age_minutes=None):
        """Remove files older than specified minutes"""
        try:
            age = max_age_minutes or self.max_age_minutes
            cutoff_time = time.time() - (age * 60)
            deleted_count = 0
            
            for file_path in self.download_dir.glob('*'):
                if file_path.is_file():
                    if file_path.stat().st_mtime < cutoff_time:
                        try:
                            file_path.unlink()
                            deleted_count += 1
                            logger.info(f"Cleaned up old file: {file_path.name}")
                        except Exception as e:
                            logger.error(f"Failed to delete {file_path}: {e}")
            
            if deleted_count > 0:
                logger.info(f"Cleanup completed: Deleted {deleted_count} old files")
                
        except Exception as e:
            logger.error(f"Cleanup error: {e}")
    
    def cleanup_file(self, filepath: str):
        """Clean up a specific file"""
        try:
            path = Path(filepath)
            if path.exists():
                path.unlink()
                logger.info(f"Cleaned up file: {path.name}")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to delete {filepath}: {e}")
            return False
    
    def get_folder_stats(self) -> dict:
        """Get statistics about download folder"""
        try:
            files = list(self.download_dir.glob('*'))
            files = [f for f in files if f.is_file()]
            total_size = sum(f.stat().st_size for f in files)
            
            return {
                'file_count': len(files),
                'total_size': total_size,
                'total_size_human': self._format_size(total_size),
                'download_folder': str(self.download_dir),
            }
        except Exception as e:
            logger.error(f"Failed to get folder stats: {e}")
            return {}
    
    def _format_size(self, size_bytes):
        """Format bytes to human readable format"""
        if size_bytes == 0:
            return "0 B"
        units = ['B', 'KB', 'MB', 'GB', 'TB']
        i = 0
        while size_bytes >= 1024 and i < len(units)-1:
            size_bytes /= 1024.0
            i += 1
        return f"{size_bytes:.2f} {units[i]}"


class YouTubeDownloader:
    """Main downloader class with all resolutions support"""
    
    def __init__(self):
        self.ydl_opts_base = {
            'quiet': True,
            'no_warnings': True,
            'extract_flat': False,
            'socket_timeout': 30,
            'noprogress': False,
            'retries': 10,
            'fragment_retries': 10,
            'skip_unavailable_fragments': True,
            'keep_fragments': False,
            'continuedl': True,
            'nocheckcertificate': True,
            'ignoreerrors': False,
            'logtostderr': False,
            'consoletitle': False,
            'no_color': True,
        }
        
        self.download_manager = SmartDownloadManager(app.config['DOWNLOAD_FOLDER'])
        self.validator = YouTubeValidator()
        logger.info("YouTubeDownloader initialized successfully")
    
    def get_video_info(self, url: str) -> dict:
        """Get video information"""
        try:
            # Normalize URL first
            url = self.validator.normalize_url(url)
            
            ydl_opts = self.ydl_opts_base.copy()
            ydl_opts.update({
                'extract_flat': True,
                'force_generic_extractor': False,
            })
            
            logger.info(f"Fetching info for URL: {url}")
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
                
                if not info:
                    raise Exception("Failed to extract video information")
                
                # Process video info
                result = self._process_video_info(info, url)
                result['success'] = True
                
                return result
                
        except yt_dlp.utils.DownloadError as e:
            error_msg = str(e)
            if "Private video" in error_msg:
                return {'success': False, 'error': 'This video is private'}
            elif "Video unavailable" in error_msg:
                return {'success': False, 'error': 'Video is unavailable'}
            elif "age restricted" in error_msg.lower():
                return {'success': False, 'error': 'Age-restricted video. Cannot download.'}
            else:
                return {'success': False, 'error': f'YouTube error: {error_msg}'}
                
        except Exception as e:
            logger.error(f"Error getting video info: {e}", exc_info=True)
            return {'success': False, 'error': f'Failed to fetch video info: {str(e)}'}
    
    def _process_video_info(self, info: dict, url: str) -> dict:
        """Process and format video information"""
        result = {
            'title': info.get('title', 'Unknown Title'),
            'duration': info.get('duration', 0),
            'uploader': info.get('uploader', 'Unknown'),
            'channel_url': info.get('channel_url', ''),
            'view_count': info.get('view_count', 0),
            'like_count': info.get('like_count', 0),
            'thumbnail': info.get('thumbnail', ''),
            'description': self._truncate_description(info.get('description', '')),
            'upload_date': self._format_date(info.get('upload_date', '')),
            'age_limit': info.get('age_limit', 0),
            'webpage_url': info.get('webpage_url', url),
            'available_formats': [],
            'subtitles': [],
            'is_live': info.get('is_live', False),
            'video_id': info.get('id', ''),
            'url': url,
        }
        
        # Process formats and get available resolutions
        formats = info.get('formats', [])
        available_resolutions = self._get_available_resolutions(formats)
        result['available_formats'] = available_resolutions
        
        # Get best audio format
        audio_formats = self._get_audio_formats(formats)
        if audio_formats:
            result['available_formats'].append({
                'id': 'mp3',
                'name': 'MP3 Audio',
                'resolution': 'Audio',
                'extension': 'mp3',
                'quality': '192kbps'
            })
        
        return result
    
    def _get_available_resolutions(self, formats):
        """Get available resolutions from video formats"""
        resolutions = []
        resolution_map = {
            144: '144p',
            240: '240p',
            360: '360p',
            480: '480p',
            720: '720p HD',
            1080: '1080p Full HD',
            1440: '1440p 2K',
            2160: '2160p 4K'
        }
        
        seen_heights = set()
        
        for fmt in formats:
            if fmt.get('vcodec') != 'none':  # Video format
                height = fmt.get('height')
                if height and height not in seen_heights:
                    seen_heights.add(height)
                    if height in resolution_map:
                        resolutions.append({
                            'id': str(height),
                            'name': resolution_map[height],
                            'resolution': f"{height}p",
                            'height': height,
                            'extension': fmt.get('ext', 'mp4'),
                            'filesize': fmt.get('filesize', 0),
                            'filesize_human': self._format_size(fmt.get('filesize', 0))
                        })
        
        # Add best quality option
        if resolutions:
            max_height = max([r['height'] for r in resolutions])
            resolutions.append({
                'id': 'best',
                'name': 'Best Quality',
                'resolution': f'Best ({max_height}p)',
                'height': max_height,
                'extension': 'mp4',
                'filesize': 0,
                'filesize_human': 'Variable'
            })
        
        # Sort by height
        resolutions.sort(key=lambda x: x['height'])
        return resolutions
    
    def _get_audio_formats(self, formats):
        """Get audio formats"""
        audio_formats = []
        for fmt in formats:
            if fmt.get('acodec') != 'none' and fmt.get('vcodec') == 'none':
                audio_formats.append(fmt)
        return audio_formats
    
    def download_video(self, url: str, format_id: str, task_id: str) -> dict:
        """Download video with specified format"""
        try:
            url = self.validator.normalize_url(url)
            ydl_opts = self._prepare_download_options(format_id, task_id)
            
            progress_data = {
                'status': 'preparing',
                'percent': 0,
                'speed': '0 B/s',
                'eta': 'N/A',
                'total_size': '0 B',
                'downloaded': '0 B',
                'filename': '',
                'task_id': task_id,
            }
            
            def progress_hook(d):
                if d['status'] == 'downloading':
                    cleaned = ProgressHook.clean_progress_data(d)
                    progress_data['status'] = 'downloading'
                    progress_data['percent'] = cleaned['percent']
                    progress_data['speed'] = cleaned.get('speed_str', '0 B/s')
                    progress_data['eta'] = cleaned.get('eta_str', 'N/A')
                    progress_data['total_size'] = cleaned.get('total_bytes_str', '0 B')
                    progress_data['downloaded'] = cleaned.get('downloaded_bytes_str', '0 B')
                    
                    if task_id in download_tasks:
                        download_tasks[task_id]['progress'] = progress_data.copy()
                        
                elif d['status'] == 'finished':
                    progress_data['status'] = 'processing'
                    progress_data['percent'] = 100
                    if task_id in download_tasks:
                        download_tasks[task_id]['progress'] = progress_data.copy()
            
            ydl_opts['progress_hooks'] = [progress_hook]
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                filename = ydl.prepare_filename(info)
                
                # MP3 বা মার্জড ফাইলের ক্ষেত্রে এক্সটেনশন চেক
                if format_id == 'mp3' and not filename.endswith('.mp3'):
                    filename = filename.rsplit('.', 1)[0] + '.mp3'
                elif not os.path.exists(filename):
                    # যদি মার্জ হওয়ার পর .mp4 হয়ে যায়
                    potential_file = filename.rsplit('.', 1)[0] + '.mp4'
                    if os.path.exists(potential_file):
                        filename = potential_file

                if os.path.exists(filename):
                    file_size = os.path.getsize(filename)
                    return {
                        'success': True,
                        'filename': filename,
                        'file_size': file_size,
                        'file_size_human': self._format_size(file_size),
                        'title': info.get('title', 'Unknown'),
                        'duration': info.get('duration', 0),
                        'thumbnail': info.get('thumbnail', ''),
                        'task_id': task_id,
                    }
                else:
                    raise Exception("Downloaded file not found")
                
        except Exception as e:
            logger.error(f"Download error for task {task_id}: {e}", exc_info=True)
            return {'success': False, 'error': str(e), 'task_id': task_id}
    
    def _prepare_download_options(self, format_id: str, task_id: str) -> dict:
        """কুকিজ সহ ডাউনলোড অপশন সেট করা"""
        ydl_opts = self.ydl_opts_base.copy()
        
        # কুকিজ ফাইল চেক করা
        cookie_file = 'youtube_cookies.txt'
        if os.path.exists(cookie_file):
            ydl_opts['cookiefile'] = cookie_file
            logger.info("ইউটিউব অথেন্টিকেশনের জন্য কুকিজ ব্যবহার করা হচ্ছে।")
            
        output_dir = str(self.download_manager.download_dir)
        ydl_opts['outtmpl'] = os.path.join(output_dir, f'%(title)s_%(id)s_{format_id}.%(ext)s')
        
        if format_id == 'mp3':
            ydl_opts['format'] = 'bestaudio/best'
            ydl_opts['postprocessors'] = [{'key': 'FFmpegExtractAudio','preferredcodec': 'mp3','preferredquality': '192'}]
        else:
            # AV1 ব্লক করে MP4 এবং নির্দিষ্ট রেজোলিউশন নিশ্চিত করা
            ydl_opts['format'] = f"bestvideo[height<={format_id}][vcodec!=av01][ext=mp4]+bestaudio[ext=m4a]/best[height<={format_id}][vcodec!=av01][ext=mp4]/best"
            ydl_opts.update({
                'merge_output_format': 'mp4',
                'postprocessors': [{'key': 'FFmpegVideoConvertor','preferedformat': 'mp4'}],
            })
        
        return ydl_opts
    
    def _truncate_description(self, description: str, max_length: int = 500) -> str:
        """Truncate description if too long"""
        if not description:
            return ''
        if len(description) <= max_length:
            return description
        return description[:max_length] + '...'
    
    def _format_date(self, date_str: str) -> str:
        """Format upload date"""
        if not date_str or len(date_str) != 8:
            return date_str
        try:
            year, month, day = date_str[:4], date_str[4:6], date_str[6:8]
            return f"{year}-{month}-{day}"
        except:
            return date_str
    
    def _format_size(self, size_bytes):
        """Format bytes to human readable format"""
        if size_bytes == 0:
            return "0 B"
        units = ['B', 'KB', 'MB', 'GB', 'TB']
        i = 0
        while size_bytes >= 1024 and i < len(units)-1:
            size_bytes /= 1024.0
            i += 1
        return f"{size_bytes:.2f} {units[i]}"
    
    def _save_history(self):
       pass

# Initialize components
downloader = YouTubeDownloader()
validator = YouTubeValidator()


# API Routes
@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'success': True,
        'status': 'running',
        'timestamp': datetime.now().isoformat(),
        'version': '1.1.0',
        'features': ['144p-1080p video', 'MP3 audio', 'auto-cleanup'],
    })


@app.route('/api/validate', methods=['POST'])
def validate_url():
    """Validate YouTube URL"""
    try:
        data = request.json
        url = data.get('url', '').strip()
        
        if not url:
            return jsonify({'success': False, 'error': 'URL is required'}), 400
        
        # Normalize and validate
        normalized_url = validator.normalize_url(url)
        is_valid = validator.is_valid_youtube_url(normalized_url)
        
        if not is_valid:
            return jsonify({
                'success': False,
                'error': 'Invalid YouTube URL. Please check the URL and try again.',
                'suggestions': [
                    'Make sure it\'s a valid YouTube video URL',
                    'Try using the full URL: https://www.youtube.com/watch?v=...',
                    'For short URLs, use: https://youtu.be/...',
                    'Check if the video is publicly available'
                ]
            }), 400
        
        return jsonify({
            'success': True,
            'normalized_url': normalized_url,
            'video_id': validator.extract_video_id(normalized_url),
        })
        
    except Exception as e:
        logger.error(f"Validation error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/info', methods=['POST'])
def get_video_info():
    """Get video information"""
    try:
        data = request.json
        url = data.get('url', '').strip()
        
        if not url:
            return jsonify({'success': False, 'error': 'URL is required'}), 400
        
        # Validate URL first
        normalized_url = validator.normalize_url(url)
        if not validator.is_valid_youtube_url(normalized_url):
            return jsonify({'success': False, 'error': 'Invalid YouTube URL'}), 400
        
        # Get video info
        info = downloader.get_video_info(normalized_url)
        
        return jsonify(info)
        
    except Exception as e:
        logger.error(f"Video info error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/download', methods=['POST'])
def start_download():
    """Start a download task"""
    try:
        data = request.json
        url = data.get('url', '').strip()
        format_id = data.get('format', 'best')
        
        if not url:
            return jsonify({'success': False, 'error': 'URL is required'}), 400
        
        # Validate URL
        normalized_url = validator.normalize_url(url)
        if not validator.is_valid_youtube_url(normalized_url):
            return jsonify({'success': False, 'error': 'Invalid YouTube URL'}), 400
        
        # Generate task ID
        task_id = hashlib.md5(f"{normalized_url}_{format_id}_{datetime.now().timestamp()}".encode()).hexdigest()[:12]
        
        # Create task entry
        download_tasks[task_id] = {
            'url': normalized_url,
            'format': format_id,
            'status': 'queued',
            'progress': {
                'percent': 0,
                'status': 'queued',
                'speed': '0 B/s',
                'eta': 'N/A',
            },
            'created_at': datetime.now().isoformat(),
            'task_id': task_id,
        }
        
        # Start download in background
        def download_task():
            try:
                download_tasks[task_id]['status'] = 'downloading'
                result = downloader.download_video(normalized_url, format_id, task_id)
                
                if result['success']:
                    download_tasks[task_id]['status'] = 'completed'
                    download_tasks[task_id]['result'] = result
                    download_tasks[task_id]['progress']['status'] = 'completed'
                    download_tasks[task_id]['progress']['percent'] = 100
                else:
                    download_tasks[task_id]['status'] = 'failed'
                    download_tasks[task_id]['error'] = result.get('error', 'Unknown error')
                    download_tasks[task_id]['progress']['status'] = 'failed'
                    
            except Exception as e:
                download_tasks[task_id]['status'] = 'failed'
                download_tasks[task_id]['error'] = str(e)
                download_tasks[task_id]['progress']['status'] = 'failed'
                logger.error(f"Download task error: {e}", exc_info=True)
        
        # Submit to thread pool
        executor.submit(download_task)
        
        return jsonify({
            'success': True,
            'task_id': task_id,
            'message': 'Download started successfully',
            'normalized_url': normalized_url,
            'format': format_id,
        })
        
    except Exception as e:
        logger.error(f"Download start error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/progress/<task_id>', methods=['GET'])
def get_progress(task_id):
    """Get download progress"""
    if task_id not in download_tasks:
        return jsonify({'success': False, 'error': 'Task not found'}), 404
    
    task = download_tasks[task_id]
    return jsonify({
        'success': True,
        'task_id': task_id,
        'status': task['status'],
        'progress': task.get('progress', {}),
        'result': task.get('result'),
        'error': task.get('error'),
        'created_at': task.get('created_at'),
    })


@app.route('/api/downloads', methods=['GET'])
def list_downloads():
    """List all downloads"""
    active_tasks = {k: v for k, v in download_tasks.items() 
                   if v['status'] in ['queued', 'downloading']}
    completed_tasks = {k: v for k, v in download_tasks.items() 
                      if v['status'] == 'completed'}
    failed_tasks = {k: v for k, v in download_tasks.items() 
                   if v['status'] == 'failed'}
    
    return jsonify({
        'success': True,
        'active_tasks': list(active_tasks.values()),
        'completed_tasks': list(completed_tasks.values()),
        'failed_tasks': list(failed_tasks.values()),
        'active_count': len(active_tasks),
        'completed_count': len(completed_tasks),
        'failed_count': len(failed_tasks),
    })


@app.route('/api/history', methods=['GET'])
def get_history():
    """Get download history"""
    return jsonify({
        'success': True,
        'history': download_history[-30:],  # Last 30 downloads
        'total_count': len(download_history),
    })


@app.route('/api/download-file/<task_id>', methods=['GET'])
def download_file(task_id):
    """Download the completed file"""
    if task_id not in download_tasks:
        return jsonify({'success': False, 'error': 'Task not found'}), 404
    
    task = download_tasks[task_id]
    result = task.get('result')
    
    if not result or 'filename' not in result:
        return jsonify({'success': False, 'error': 'ফাইল তথ্য পাওয়া যায়নি'}), 404
    
    # সঠিক ফাইল পাথ যেটা yt-dlp সেভ করেছে
    filename = result['filename']
    
    if not os.path.exists(filename):
        return jsonify({'success': False, 'error': 'ফাইলটি সার্ভারে নেই। নতুন করে ডাউনলোড করুন।'}), 404

    # ডাউনলোডের জন্য নাম তৈরি
    title = result.get('title', 'video')
    _, ext = os.path.splitext(filename)
    safe_filename = f"{title}{ext}"
    
    # Delayed Delete (৫ মিনিট পর ডিলিট হবে)
    if app.config.get('DELETE_AFTER_DOWNLOAD', True):
        threading.Thread(target=lambda: (time.sleep(300), os.remove(filename) if os.path.exists(filename) else None), daemon=True).start()

    import urllib.parse
    encoded_name = urllib.parse.quote(safe_filename)
    
    response = send_file(filename, as_attachment=True, download_name=safe_filename, conditional=True)
    response.headers["Content-Disposition"] = f"attachment; filename*=UTF-8''{encoded_name}"
    return response


@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get download statistics"""
    try:
        folder_stats = downloader.download_manager.get_folder_stats()
        
        # Cleanup old files
        downloader.download_manager.cleanup_old_files()
        
        return jsonify({
            'success': True,
            'folder_stats': folder_stats,
            'total_tasks': len(download_tasks),
            'active_tasks': len([t for t in download_tasks.values() 
                                if t['status'] in ['queued', 'downloading']]),
            'history_count': len(download_history),
            'server_uptime': datetime.now().isoformat(),
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/cleanup', methods=['POST'])
def cleanup_files():
    """Clean up old files"""
    try:
        data = request.json
        max_age_minutes = data.get('max_age_minutes', 10)
        
        # Run cleanup
        downloader.download_manager.cleanup_old_files(max_age_minutes)
        
        # Get updated stats
        folder_stats = downloader.download_manager.get_folder_stats()
        
        return jsonify({
            'success': True,
            'message': f'Cleanup completed (files older than {max_age_minutes} minutes)',
            'folder_stats': folder_stats,
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/formats', methods=['GET'])
def get_formats():
    """Get available formats"""
    formats = [
        {'id': 'best', 'name': 'Best Quality', 'description': 'Best available quality'},
        {'id': '144', 'name': '144p', 'description': '256x144 resolution'},
        {'id': '240', 'name': '240p', 'description': '426x240 resolution'},
        {'id': '360', 'name': '360p', 'description': '640x360 resolution'},
        {'id': '480', 'name': '480p', 'description': '854x480 resolution'},
        {'id': '720', 'name': '720p HD', 'description': '1280x720 resolution'},
        {'id': '1080', 'name': '1080p Full HD', 'description': '1920x1080 resolution'},
        {'id': 'mp3', 'name': 'MP3 Audio', 'description': 'Audio Only (MP3 format)'},
    ]
    
    return jsonify({
        'success': True,
        'formats': formats,
    })


@app.route('/api/clear-history', methods=['POST'])
def clear_history():
    """Clear download history"""
    try:
        global download_history
        download_history.clear()
        
        # Also clear history file
        history_file = Path('download_history.json')
        if history_file.exists():
            history_file.unlink()
        
        return jsonify({
            'success': True,
            'message': 'History cleared successfully'
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


# Serve frontend - IMPORTANT: This route should be LAST
@app.route('/')
def serve_index():
    """Serve the main HTML page"""
    try:
        return send_file('static/index.html')
    except:
        return jsonify({
            'success': False, 
            'error': 'Frontend not found. Please check if static/index.html exists.',
            'api_endpoints': [
                '/api/health',
                '/api/validate',
                '/api/info',
                '/api/download',
                '/api/formats',
            ]
        }), 404


# Error handlers
@app.errorhandler(404)
def not_found_error(error):
    return jsonify({'success': False, 'error': 'Resource not found'}), 404


@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {error}")
    return jsonify({'success': False, 'error': 'Internal server error'}), 500


# Cleanup on exit
def cleanup():
    """Cleanup function on exit"""
    logger.info("Shutting down...")
    executor.shutdown(wait=False)
    downloader.download_manager.cleanup_old_files(1)  # Cleanup files older than 1 minute
    logger.info("Cleanup completed.")


# Register cleanup
atexit.register(cleanup)
signal.signal(signal.SIGTERM, lambda sig, frame: cleanup())
signal.signal(signal.SIGINT, lambda sig, frame: cleanup())


if __name__ == '__main__':
    # Load history if exists
    # Run the server
    logger.info("Starting YouTube Downloader server...")
    logger.info(f"Download folder: {app.config['DOWNLOAD_FOLDER']}")
    logger.info(f"Max concurrent downloads: {app.config['MAX_CONCURRENT_DOWNLOADS']}")
    logger.info(f"Auto delete after download: {app.config['DELETE_AFTER_DOWNLOAD']}")
    
    import os
    port = int(os.environ.get("PORT", 5000))
    
    app.run(
        host='0.0.0.0',
        port=port,  # এখানে ভেরিয়েবলটি ব্যবহার করুন
        debug=False,
        threaded=True,
        use_reloader=False
    )
