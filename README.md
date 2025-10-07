# RadarBot AI Optimization Package

## Overview
This package contains the core AI modules for OCR processing and portfolio analysis optimization. The code has been cleaned and prepared for performance improvements.

## Key Components

### OCR Processing
- `bot/ai/vision.py` - Main OCR processing with OpenAI Vision API
- `bot/ai/vision_speed.py` - Speed optimizations for OCR
- OCR prompt files for different versions (v10, v12, v13)

### Pipeline Processing
- `bot/pipeline/portfolio_ingest_pipeline.py` - Main processing pipeline
- `bot/pipeline/portfolio_ingest_pipeline_speed.py` - Optimized version

### AI Analysis
- `bot/analytics/portfolio_analyzer_parallel.py` - Parallel portfolio analysis
- `bot/analytics/portfolio_analyzer.py` - Base analysis logic

### Utilities
- `bot/utils/timing.py` - Performance measurement decorators
- `bot/utils/speed_cache.py` - High-performance caching system
- `bot/utils/ocr_cache.py` - OCR result caching
- `bot/utils/bond_cache.py` - Bond data caching
- `bot/utils/cache.py` - General caching mechanisms
- `bot/utils/normalize.py` - Data normalization functions
- `bot/utils/render.py` - Result rendering functions

## Configuration
The `bot/core/config.py` file contains all performance-related settings:
- OCR image compression settings
- Concurrency parameters
- Cache TTL values
- Model routing options

## Setup Instructions
1. Install dependencies: `pip install -r requirements.txt`
2. Configure environment variables as needed
3. The code is ready for optimization work

## Areas for Optimization
- OCR processing speed and accuracy
- Parallel processing improvements
- Caching strategy enhancements
- Memory usage optimization
- API call efficiency

## Notes
- All sensitive data has been removed
- Comments have been stripped for cleaner code review
- Focus on performance improvements only
- Maintain existing functionality while optimizing speed

## Dependencies
See `requirements.txt` for required packages. Main dependencies include:
- OpenAI API client
- PIL/Pillow for image processing
- asyncio for parallel processing
- Various caching and utility libraries
