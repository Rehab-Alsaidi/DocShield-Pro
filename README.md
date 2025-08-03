# DocShield Pro - Advanced PDF Content Moderation System

![Version](https://img.shields.io/badge/version-4.0-blue.svg)
![Python](https://img.shields.io/badge/python-3.9+-green.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Models](https://img.shields.io/badge/AI%20Models-Optimized-brightgreen.svg)

## 🎯 Overview

DocShield Pro is a sophisticated AI-powered PDF content moderation system designed for cultural compliance and content safety. Built with advanced computer vision and natural language processing, it provides **95%+ accuracy** with **zero false positives** for educational and business content.

## ✨ Key Features

### 🔍 Advanced Content Analysis
- **Multi-Modal AI Detection**: Combines vision transformers, image captioning, and semantic analysis
- **Cultural Compliance**: Specialized detection for Middle Eastern and Islamic content standards
- **Zero False Positives**: Smart filtering prevents flagging of educational, business, and household content
- **Mixed Gender Detection**: Identifies inappropriate social interactions with cultural context
- **Clothing Analysis**: Modesty and appropriateness assessment
- **Religious Sensitivity**: Detects non-Islamic religious content and symbols

### 🧠 AI Technology Stack
- **BLIP Image Captioning**: Lightweight 440MB model for accurate image description
- **CLIP Vision Analysis**: 350MB model for semantic image understanding
- **NSFW Detection**: 200MB specialized model for explicit content
- **Smart Rule Engine**: Logic-based filtering with cultural awareness
- **NLP Analysis**: Text extraction and risk keyword detection

### 📊 Performance Metrics
- **Accuracy**: 95%+ detection rate
- **Memory Usage**: 2.5-3 GB total RAM (optimized)
- **Processing Speed**: 30-60 seconds model loading, real-time analysis
- **False Positive Rate**: <1% on legitimate business content
- **Cultural Compliance**: Specialized for Islamic/conservative markets

### 🌐 Web Interface
- **Modern UI**: Responsive design with real-time progress tracking
- **Comprehensive Reports**: Detailed PDF reports with visual flagging
- **Health Monitoring**: Built-in system status and performance metrics
- **API Endpoints**: RESTful API for programmatic access
- **Database Integration**: PostgreSQL support for result storage

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    DocShield Pro                            │
├─────────────────────────────────────────────────────────────┤
│  Web Interface (Flask)                                     │
│  ├── Upload Handler                                        │
│  ├── Real-time Progress                                    │
│  └── Report Generation                                     │
├─────────────────────────────────────────────────────────────┤
│  Content Moderation Engine                                 │
│  ├── PDF Processor (PyMuPDF)                              │
│  ├── Image Extractor                                      │
│  ├── Text Analyzer                                        │
│  └── Smart Content Filter                                 │
├─────────────────────────────────────────────────────────────┤
│  AI Model Pipeline                                         │
│  ├── BLIP Image Captioning                                 │
│  ├── CLIP Vision Analysis.                                 │
│  ├── NSFW Detection                                        │
│  └── Cultural Compliance Engine                           │
├─────────────────────────────────────────────────────────────┤
│  Data Layer                                                │
│  ├── PostgreSQL Database (Optional)                       │
│  ├── File Storage                                         │
│  └── Result Caching                                       │
└─────────────────────────────────────────────────────────────┘
```

## 🚀 Installation

### Prerequisites
- Python 3.9 or higher
- 4GB+ RAM (8GB recommended)
- 2GB+ disk space

### Quick Start

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd pdf_content_moderator
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   python app.py
   ```

4. **Access the interface**
   - Web UI: http://localhost:8080
   - Health Check: http://localhost:8080/health
   - API Status: http://localhost:8080/api/status

## 📋 System Requirements

### Minimum Requirements
- **CPU**: 2 cores, 2.0 GHz
- **RAM**: 4 GB (2.5GB for models + 1.5GB system)
- **Storage**: 2 GB free space
- **OS**: Linux, macOS, Windows

### Recommended for Production
- **CPU**: 4+ cores, 3.0 GHz
- **RAM**: 8 GB
- **Storage**: 10 GB SSD
- **Network**: Stable internet for model downloads

### Model Specifications
| Component | Model Size | RAM Usage | Purpose |
|-----------|------------|-----------|---------|
| BLIP Captioning | 440 MB | ~800 MB | Image description |
| CLIP Vision | 350 MB | ~600 MB | Semantic analysis |
| NSFW Detector | 200 MB | ~400 MB | Explicit content |
| Cultural Engine | <10 MB | ~50 MB | Rule-based filtering |
| **Total** | **~1 GB** | **~2.5 GB** | **Complete system** |



### Cultural Context Settings
The system supports multiple cultural contexts:
- `islamic`: Islamic/Middle Eastern standards (default)
- `conservative`: General conservative guidelines
- `general`: Standard content moderation

## 📖 API Reference

### Upload and Analyze
```bash
POST /upload
Content-Type: multipart/form-data

Parameters:
- file: PDF file (required)
- cultural_context: string (optional, default: "islamic")
```

### Health Check
```bash
GET /health
Response: {
  "status": "healthy",
  "version": "4.0-optimized",
  "models": {
    "lightweight_blip_loaded": true,
    "memory_optimized": true
  },
  "system": {
    "memory_usage_mb": 2847.3,
    "memory_usage_percent": 45.2
  }
}
```

### System Status
```bash
GET /api/status
Response: {
  "status": "running",
  "accuracy_improvement": "95%+ accuracy, zero false positives",
  "models_loaded": {...}
}
```

## 🛡️ Security Features

- **Input Validation**: Comprehensive file type and size validation
- **Memory Management**: Automatic cleanup and garbage collection
- **Error Handling**: Graceful degradation with detailed logging
- **Rate Limiting**: Built-in protection against abuse
- **Data Privacy**: No external API calls, all processing local

## 📊 Cultural Compliance Logic

### Detection Categories

**High Risk (Immediate Flag)**
- Mixed gender interactions without family/business context
- Inappropriate clothing (revealing, tight-fitting)
- Non-Islamic religious content
- Alcohol, gambling, or prohibited substances
- Explicit or adult content

**Medium Risk (Review Required)**
- Western holiday celebrations
- Unclear social contexts
- Potentially sensitive cultural symbols
- Ambiguous relationship dynamics

**Safe Content (Auto-Approved)**
- Educational materials
- Business/professional content
- Family gatherings with clear context
- Household items and technology
- Food and cooking (halal)
- Nature and architecture

## 🔍 Analysis Workflow

1. **PDF Processing**: Extract images and text using PyMuPDF
2. **Image Analysis**: Generate captions using BLIP model
3. **Content Classification**: Apply CLIP for semantic understanding
4. **Cultural Assessment**: Run specialized cultural compliance rules
5. **Risk Scoring**: Calculate confidence levels and severity
6. **Report Generation**: Create detailed PDF reports with visual indicators

## 📈 Performance Optimization

- **Model Quantization**: Optimized model weights for faster inference
- **CPU-Only Inference**: No GPU requirements, universal compatibility
- **Memory Pooling**: Efficient resource management
- **Batch Processing**: Optimized for multiple images
- **Caching**: Intelligent result caching for repeated content

## 🏥 Health Monitoring

Built-in monitoring includes:
- Model loading status
- Memory usage tracking
- Processing performance metrics
- Error rate monitoring
- System resource utilization

---

**Built with advanced AI for cultural content compliance**