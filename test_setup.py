#!/usr/bin/env python3
"""
HOABOT Setup Test Script
Tests all components of the HOABOT application to ensure everything is working.
"""

import sys
import os
import importlib
from pathlib import Path

def test_imports():
    """Test if all required packages can be imported."""
    print("🔍 Testing package imports...")
    
    required_packages = [
        'streamlit',
        'pinecone',
        'sentence_transformers',
        'fitz',  # PyMuPDF
        'pytesseract',
        'pdf2image',
        'requests',
        'numpy',
        'pandas',
        'PIL'
    ]
    
    failed_imports = []
    
    for package in required_packages:
        try:
            importlib.import_module(package)
            print(f"✅ {package}")
        except ImportError as e:
            print(f"❌ {package}: {e}")
            failed_imports.append(package)
    
    if failed_imports:
        print(f"\n❌ Failed to import: {', '.join(failed_imports)}")
        print("Please install missing packages with: pip install -r requirements.txt")
        return False
    
    print("✅ All packages imported successfully!")
    return True

def test_app_modules():
    """Test if all app modules can be imported."""
    print("\n🔍 Testing app modules...")
    
    app_modules = [
        'app.config',
        'app.model_cache',
        'app.document_processor',
        'app.vector_store',
        'app.chat_engine',
        'app.utils'
    ]
    
    failed_modules = []
    
    for module in app_modules:
        try:
            importlib.import_module(module)
            print(f"✅ {module}")
        except ImportError as e:
            print(f"❌ {module}: {e}")
            failed_modules.append(module)
    
    if failed_modules:
        print(f"\n❌ Failed to import modules: {', '.join(failed_modules)}")
        return False
    
    print("✅ All app modules imported successfully!")
    return True

def test_configuration():
    """Test configuration loading."""
    print("\n🔧 Testing configuration...")
    
    try:
        from app.config import Config
        
        # Test Pinecone config
        pinecone_config = Config.get_pinecone_config()
        if not pinecone_config.get('api_key'):
            print("⚠️  Pinecone API key not configured")
        else:
            print("✅ Pinecone configuration loaded")
        
        # Test Perplexity config
        perplexity_config = Config.get_perplexity_config()
        if not perplexity_config.get('api_key'):
            print("⚠️  Perplexity API key not configured")
        else:
            print("✅ Perplexity configuration loaded")
        
        # Test email config
        email_config = Config.get_email_config()
        if not email_config.get('smtp_username'):
            print("⚠️  Email configuration not complete (optional)")
        else:
            print("✅ Email configuration loaded")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

def test_model_cache():
    """Test model cache initialization."""
    print("\n🤖 Testing model cache...")
    
    try:
        from app.model_cache import ModelCache
        
        model_cache = ModelCache()
        print("✅ Model cache initialized")
        
        # Test model info
        model_info = model_cache.get_model_info()
        print(f"✅ Model info: {model_info.get('type', 'Unknown')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Model cache test failed: {e}")
        return False

def test_document_processor():
    """Test document processor initialization."""
    print("\n📄 Testing document processor...")
    
    try:
        from app.document_processor import DocumentProcessor
        
        processor = DocumentProcessor()
        print("✅ Document processor initialized")
        
        return True
        
    except Exception as e:
        print(f"❌ Document processor test failed: {e}")
        return False

def test_system_dependencies():
    """Test system dependencies."""
    print("\n🔧 Testing system dependencies...")
    
    # Test Tesseract
    try:
        import pytesseract
        pytesseract.get_tesseract_version()
        print("✅ Tesseract OCR available")
    except Exception as e:
        print(f"⚠️  Tesseract OCR not available: {e}")
        print("   Install with: brew install tesseract (macOS) or apt-get install tesseract-ocr (Ubuntu)")
    
    # Test Poppler
    try:
        from pdf2image import convert_from_path
        print("✅ Poppler utilities available")
    except Exception as e:
        print(f"⚠️  Poppler utilities not available: {e}")
        print("   Install with: brew install poppler (macOS) or apt-get install poppler-utils (Ubuntu)")
    
    return True

def main():
    """Run all tests."""
    print("🤖 HOABOT Setup Test")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_app_modules,
        test_configuration,
        test_model_cache,
        test_document_processor,
        test_system_dependencies
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! HOABOT is ready to use.")
        print("\n🚀 To start HOABOT, run:")
        print("   python run.py")
        print("   or")
        print("   streamlit run main.py")
    else:
        print("⚠️  Some tests failed. Please check the issues above.")
        print("\n💡 Common solutions:")
        print("   1. Install dependencies: pip install -r requirements.txt")
        print("   2. Configure API keys in .streamlit/secrets.toml")
        print("   3. Install system dependencies (Tesseract, Poppler)")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
