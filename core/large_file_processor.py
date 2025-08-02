# core/large_file_processor.py
"""
Large File PDF Processor - Optimized for 4GB+ files
Handles memory management, chunked processing, and progress tracking
"""

import fitz  # PyMuPDF
import os
import io
import gc
import psutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Tuple, Optional, Iterator, Callable
from dataclasses import dataclass
import logging
from pathlib import Path

from core.pdf_processor import PDFProcessor, ProcessedImage
from utils.logger import get_logger

logger = get_logger(__name__)

@dataclass
class ProcessingProgress:
    """Track processing progress for large files"""
    total_pages: int
    processed_pages: int
    total_images: int
    processed_images: int
    current_phase: str
    start_time: float
    estimated_completion: Optional[float] = None
    memory_usage_mb: float = 0
    
    @property
    def progress_percentage(self) -> float:
        if self.total_pages == 0:
            return 0.0
        return (self.processed_pages / self.total_pages) * 100

@dataclass
class MemoryStats:
    """Memory usage statistics"""
    total_mb: float
    available_mb: float
    used_mb: float
    process_mb: float
    threshold_mb: float
    is_critical: bool

class MemoryMonitor:
    """Monitor and manage memory usage during processing"""
    
    def __init__(self, max_memory_mb: int = 2048):
        self.max_memory_mb = max_memory_mb
        self.critical_threshold = max_memory_mb * 0.9
        self.warning_threshold = max_memory_mb * 0.7
        self._monitoring = False
        self._monitor_thread = None
        
    def start_monitoring(self):
        """Start memory monitoring in background thread"""
        self._monitoring = True
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        
    def stop_monitoring(self):
        """Stop memory monitoring"""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=1)
            
    def get_memory_stats(self) -> MemoryStats:
        """Get current memory statistics"""
        memory = psutil.virtual_memory()
        process = psutil.Process()
        process_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        return MemoryStats(
            total_mb=memory.total / 1024 / 1024,
            available_mb=memory.available / 1024 / 1024,
            used_mb=memory.used / 1024 / 1024,
            process_mb=process_memory,
            threshold_mb=self.max_memory_mb,
            is_critical=process_memory > self.critical_threshold
        )
        
    def should_pause_processing(self) -> bool:
        """Check if processing should be paused due to memory pressure"""
        stats = self.get_memory_stats()
        return stats.is_critical
        
    def _monitor_loop(self):
        """Background monitoring loop"""
        while self._monitoring:
            stats = self.get_memory_stats()
            if stats.is_critical:
                logger.warning(f"Critical memory usage: {stats.process_mb:.1f}MB / {self.max_memory_mb}MB")
                # Force garbage collection
                gc.collect()
            elif stats.process_mb > self.warning_threshold:
                logger.info(f"High memory usage: {stats.process_mb:.1f}MB / {self.max_memory_mb}MB")
            
            time.sleep(5)  # Check every 5 seconds

class TextHighlighter:
    """Extract and highlight specific words in PDF text"""
    
    def __init__(self, risk_keywords: List[str]):
        self.risk_keywords = [kw.lower() for kw in risk_keywords]
        
    def extract_text_with_positions(self, page) -> Dict:
        """Extract text with word positions for highlighting"""
        text_dict = page.get_text("dict")
        words_data = []
        full_text = ""
        
        for block in text_dict["blocks"]:
            if "lines" in block:
                for line in block["lines"]:
                    for span in line["spans"]:
                        text = span["text"]
                        bbox = span["bbox"]  # x0, y0, x1, y1
                        font_size = span["size"]
                        
                        # Split into words and track positions
                        words = text.split()
                        word_start_x = bbox[0]
                        
                        for word in words:
                            word_data = {
                                "word": word,
                                "bbox": [word_start_x, bbox[1], word_start_x + len(word) * font_size * 0.6, bbox[3]],
                                "font_size": font_size,
                                "is_highlighted": word.lower() in self.risk_keywords,
                                "risk_level": "high" if word.lower() in self.risk_keywords else "none"
                            }
                            words_data.append(word_data)
                            full_text += word + " "
                            word_start_x += len(word) * font_size * 0.6 + 5  # Approximate spacing
        
        return {
            "full_text": full_text.strip(),
            "words": words_data,
            "highlighted_count": len([w for w in words_data if w["is_highlighted"]])
        }

class LargeFilePDFProcessor:
    """Enhanced PDF processor optimized for large files (4GB+)"""
    
    def __init__(self, 
                 max_memory_mb: int = 2048, 
                 chunk_size: int = 10,
                 max_workers: int = 2):
        self.base_processor = PDFProcessor()
        self.max_memory_mb = max_memory_mb
        self.chunk_size = chunk_size
        self.max_workers = max_workers
        self.memory_monitor = MemoryMonitor(max_memory_mb)
        self.progress_callback: Optional[Callable[[ProcessingProgress], None]] = None
        
        # Risk keywords for highlighting
        self.risk_keywords = [
            "alcohol", "beer", "wine", "vodka", "whiskey", "gambling", "casino", 
            "poker", "betting", "adult", "explicit", "nsfw", "naked", "nude",
            "inappropriate", "dating", "romance", "intimate", "sexy"
        ]
        
        self.text_highlighter = TextHighlighter(self.risk_keywords)
        logger.info(f"Large File PDF Processor initialized - Max Memory: {max_memory_mb}MB")
        
    def set_progress_callback(self, callback: Callable[[ProcessingProgress], None]):
        """Set callback for progress updates"""
        self.progress_callback = callback
        
    def process_large_pdf(self, pdf_path: str, 
                         output_dir: str = None) -> Tuple[List[ProcessedImage], List[Dict], ProcessingProgress]:
        """Process large PDF with memory management and progress tracking"""
        
        start_time = time.time()
        
        # Validate file
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
            
        file_size_mb = os.path.getsize(pdf_path) / 1024 / 1024
        logger.info(f"Processing large PDF: {Path(pdf_path).name} ({file_size_mb:.1f} MB)")
        
        # Start memory monitoring
        self.memory_monitor.start_monitoring()
        
        try:
            # Get PDF info
            pdf_doc = fitz.open(pdf_path)
            total_pages = pdf_doc.page_count
            pdf_doc.close()
            
            # Initialize progress tracking
            progress = ProcessingProgress(
                total_pages=total_pages,
                processed_pages=0,
                total_images=0,
                processed_images=0,
                current_phase="Initializing",
                start_time=start_time
            )
            
            logger.info(f"PDF has {total_pages} pages, processing in chunks of {self.chunk_size}")
            
            # Process in chunks
            all_processed_images = []
            all_text_data = []
            
            for chunk_start in range(0, total_pages, self.chunk_size):
                chunk_end = min(chunk_start + self.chunk_size, total_pages)
                
                progress.current_phase = f"Processing pages {chunk_start + 1}-{chunk_end}"
                self._update_progress(progress)
                
                # Process chunk
                chunk_images, chunk_text = self._process_chunk(
                    pdf_path, chunk_start, chunk_end, progress
                )
                
                all_processed_images.extend(chunk_images)
                all_text_data.extend(chunk_text)
                
                # Update progress
                progress.processed_pages = chunk_end
                progress.total_images = len(all_processed_images)
                progress.processed_images = len(all_processed_images)
                
                # Memory management
                if self.memory_monitor.should_pause_processing():
                    logger.warning("Pausing for memory cleanup...")
                    gc.collect()
                    time.sleep(2)
                
                self._update_progress(progress)
            
            # Final progress update
            progress.current_phase = "Completed"
            progress.estimated_completion = time.time()
            self._update_progress(progress)
            
            logger.info(f"Large PDF processing completed: {len(all_processed_images)} images, {len(all_text_data)} text blocks")
            
            return all_processed_images, all_text_data, progress
            
        finally:
            self.memory_monitor.stop_monitoring()
            
    def _process_chunk(self, pdf_path: str, start_page: int, end_page: int, 
                      progress: ProcessingProgress) -> Tuple[List[ProcessedImage], List[Dict]]:
        """Process a chunk of pages"""
        
        processed_images = []
        text_data = []
        
        try:
            pdf_doc = fitz.open(pdf_path)
            
            for page_num in range(start_page, end_page):
                if page_num >= pdf_doc.page_count:
                    break
                    
                page = pdf_doc[page_num]
                
                # Extract and process images
                image_list = page.get_images()
                for img_index, img in enumerate(image_list):
                    try:
                        # Use base processor for image extraction
                        extracted_img = self.base_processor._extract_single_image(
                            pdf_doc, img, page_num, img_index
                        )
                        
                        if extracted_img:
                            # Generate caption
                            caption, confidence = self.base_processor.florence_model.generate_caption(
                                extracted_img.image
                            )
                            
                            # Convert to ProcessedImage
                            processed_img = ProcessedImage(
                                page_number=extracted_img.page_number,
                                image_index=extracted_img.image_index,
                                size=extracted_img.size,
                                format=extracted_img.format,
                                caption=caption,
                                confidence=confidence,
                                image_base64=self.base_processor._image_to_base64(extracted_img.image),
                                image_hash=extracted_img.image_hash,
                                analysis_metadata={
                                    "processing_timestamp": time.time(),
                                    "model_used": "Florence-2-base",
                                    "chunk_processed": f"{start_page}-{end_page}"
                                }
                            )
                            
                            processed_images.append(processed_img)
                            
                    except Exception as e:
                        logger.warning(f"Failed to process image {img_index} on page {page_num + 1}: {e}")
                
                # Extract text with highlighting
                try:
                    text_with_positions = self.text_highlighter.extract_text_with_positions(page)
                    text_with_positions["page_number"] = page_num + 1
                    text_data.append(text_with_positions)
                    
                except Exception as e:
                    logger.warning(f"Failed to extract text from page {page_num + 1}: {e}")
                
                # Memory check
                if self.memory_monitor.should_pause_processing():
                    logger.warning("Memory pressure detected, cleaning up...")
                    gc.collect()
            
            pdf_doc.close()
            
        except Exception as e:
            logger.error(f"Failed to process chunk {start_page}-{end_page}: {e}")
            
        return processed_images, text_data
        
    def _update_progress(self, progress: ProcessingProgress):
        """Update progress and call callback if set"""
        progress.memory_usage_mb = self.memory_monitor.get_memory_stats().process_mb
        
        if progress.processed_pages > 0 and progress.total_pages > 0:
            elapsed = time.time() - progress.start_time
            pages_per_second = progress.processed_pages / max(elapsed, 1)
            remaining_pages = progress.total_pages - progress.processed_pages
            progress.estimated_completion = time.time() + (remaining_pages / max(pages_per_second, 0.1))
        
        if self.progress_callback:
            self.progress_callback(progress)
            
        logger.info(f"Progress: {progress.progress_percentage:.1f}% - {progress.current_phase}")

    def get_file_info(self, pdf_path: str) -> Dict:
        """Get comprehensive file information"""
        try:
            file_size = os.path.getsize(pdf_path)
            file_size_mb = file_size / 1024 / 1024
            
            pdf_doc = fitz.open(pdf_path)
            page_count = pdf_doc.page_count
            metadata = pdf_doc.metadata
            pdf_doc.close()
            
            # Estimate processing requirements
            estimated_memory_mb = min(file_size_mb * 0.5, 1024)  # Conservative estimate
            estimated_time_minutes = page_count * 0.5  # ~30 seconds per page
            
            return {
                "file_path": pdf_path,
                "file_name": Path(pdf_path).name,
                "file_size_mb": round(file_size_mb, 2),
                "file_size_gb": round(file_size_mb / 1024, 2),
                "page_count": page_count,
                "metadata": metadata,
                "processing_estimates": {
                    "memory_required_mb": round(estimated_memory_mb),
                    "estimated_time_minutes": round(estimated_time_minutes),
                    "recommended_chunk_size": max(5, min(20, 100 // max(1, int(file_size_mb / 100))))
                },
                "is_large_file": file_size_mb > 100,
                "requires_chunking": file_size_mb > 500
            }
            
        except Exception as e:
            logger.error(f"Failed to get file info: {e}")
            return {"error": str(e)}

# Example usage
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python large_file_processor.py <pdf_path>")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    
    def progress_callback(progress: ProcessingProgress):
        print(f"Progress: {progress.progress_percentage:.1f}% - {progress.current_phase}")
        print(f"Memory usage: {progress.memory_usage_mb:.1f}MB")
    
    # Test large file processing
    processor = LargeFilePDFProcessor(max_memory_mb=1024, chunk_size=5)
    processor.set_progress_callback(progress_callback)
    
    # Get file info first
    file_info = processor.get_file_info(pdf_path)
    print(f"File info: {file_info}")
    
    # Process file
    images, text_data, final_progress = processor.process_large_pdf(pdf_path)
    
    print(f"\nProcessing completed:")
    print(f"Images processed: {len(images)}")
    print(f"Text blocks extracted: {len(text_data)}")
    print(f"Total time: {final_progress.estimated_completion - final_progress.start_time:.1f}s")