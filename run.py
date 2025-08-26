#!/usr/bin/env python3
"""
HOABOT Launch Script
Simple script to launch the HOABOT RAG application with proper error handling.
"""

import subprocess
import sys
import os
from pathlib import Path

def check_dependencies():
    """Check if required dependencies are installed."""
    try:
        import streamlit
        print("✅ Streamlit is installed")
    except ImportError:
        print("❌ Streamlit is not installed. Please run: pip install -r requirements.txt")
        return False
    
    try:
        import pinecone
        print("✅ Pinecone client is installed")
    except ImportError:
        print("❌ Pinecone client is not installed. Please run: pip install -r requirements.txt")
        return False
    
    try:
        import sentence_transformers
        print("✅ Sentence Transformers is installed")
    except ImportError:
        print("❌ Sentence Transformers is not installed. Please run: pip install -r requirements.txt")
        return False
    
    return True

def check_configuration():
    """Check if configuration files exist."""
    secrets_file = Path(".streamlit/secrets.toml")
    
    if not secrets_file.exists():
        print("⚠️  Configuration file not found: .streamlit/secrets.toml")
        print("Please create the configuration file with your API keys.")
        print("See README.md for instructions.")
        return False
    
    print("✅ Configuration file found")
    return True

def main():
    """Main launch function."""
    print("🤖 HOABOT - AI Document Assistant")
    print("=" * 50)
    
    # Check dependencies
    print("\n🔍 Checking dependencies...")
    if not check_dependencies():
        sys.exit(1)
    
    # Check configuration
    print("\n🔧 Checking configuration...")
    if not check_configuration():
        print("\n💡 You can still run the app, but some features may not work.")
        response = input("Continue anyway? (y/N): ")
        if response.lower() != 'y':
            sys.exit(1)
    
    # Launch the application
    print("\n🚀 Launching HOABOT...")
    print("📱 The application will open in your default browser")
    print("🔗 URL: http://localhost:8501")
    print("\n" + "=" * 50)
    
    try:
        # Run streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "main.py",
            "--server.port=8501",
            "--server.address=localhost",
            "--browser.gatherUsageStats=false"
        ], check=True)
    except KeyboardInterrupt:
        print("\n👋 HOABOT stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error launching HOABOT: {e}")
        sys.exit(1)
    except FileNotFoundError:
        print("\n❌ Streamlit not found. Please install it with: pip install streamlit")
        sys.exit(1)

if __name__ == "__main__":
    main()
