#!/usr/bin/env python3
"""
Demo Mode Startup Script
Runs the RAG system in demo mode without external dependencies.
"""

import os
import sys
import uvicorn
import logging
from pathlib import Path

def setup_demo_environment():
    """Setup environment for demo mode"""
    print("Setting up demo environment...")

    # Copy demo env file to .env if it doesn't exist
    env_file = Path(".env")
    demo_env = Path(".env.demo")

    if demo_env.exists():
        if not env_file.exists():
            # Copy demo env to .env
            with open(demo_env, 'r') as f:
                demo_content = f.read()

            with open(env_file, 'w') as f:
                f.write(demo_content)

            print("Created .env file from .env.demo")
        else:
            print(".env file already exists")

            # Check if demo mode is enabled
            with open(env_file, 'r') as f:
                content = f.read()

            if 'DEMO_MODE=true' not in content:
                # Ask user if they want to enable demo mode
                response = input("Demo mode not enabled in .env. Enable it? (y/n): ")
                if response.lower() in ['y', 'yes']:
                    # Add or update demo mode setting
                    lines = content.split('\n')
                    demo_added = False

                    for i, line in enumerate(lines):
                        if line.startswith('DEMO_MODE='):
                            lines[i] = 'DEMO_MODE=true'
                            demo_added = True
                            break

                    if not demo_added:
                        lines.append('DEMO_MODE=true')

                    with open(env_file, 'w') as f:
                        f.write('\n'.join(lines))

                    print("Enabled demo mode in .env file")
    else:
        print("Warning: .env.demo not found, creating minimal demo configuration...")

        demo_content = """# Demo Mode Configuration
DEMO_MODE=true
DEFAULT_LLM_PROVIDER=demo
DEFAULT_LLM_MODEL=demo-gpt-4
DEMO_RESPONSE_DELAY=1
DEMO_ENABLE_SOURCES=true
"""

        with open(env_file, 'w') as f:
            f.write(demo_content)

        print("Created demo configuration")

    # Set environment variable for this session
    os.environ['DEMO_MODE'] = 'true'

def check_demo_requirements():
    """Check if demo mode can run"""
    print("Checking demo requirements...")

    requirements_met = True

    # Check Python version
    if sys.version_info < (3, 8):
        print("ERROR: Python 3.8+ required")
        requirements_met = False
    else:
        print(f"Python {sys.version_info.major}.{sys.version_info.minor}")

    # Check required packages
    required_packages = ['fastapi', 'uvicorn', 'numpy']
    missing_packages = []

    for package in required_packages:
        try:
            __import__(package)
            print(f"OK: {package}")
        except ImportError:
            print(f"ERROR: {package} (missing)")
            missing_packages.append(package)
            requirements_met = False

    if missing_packages:
        print(f"\nInstall missing packages:")
        print(f"pip install {' '.join(missing_packages)}")

    return requirements_met

def start_demo_server():
    """Start the demo server"""
    print("\nStarting demo server...")

    try:
        # Import the main app
        from main import app

        print("\nDemo Mode Active!")
        print("="*50)
        print("All AI responses are simulated")
        print("Document operations are mocked")
        print("No external API keys required")
        print("Perfect for UI demonstrations")
        print("="*50)
        print("\nServer starting at: http://127.0.0.1:8000")
        print("Main interface: http://127.0.0.1:8000")
        print("Admin panel: http://127.0.0.1:8000/upload")
        print("\nTry these demo features:")
        print("• Upload documents (simulated)")
        print("• Ask questions in chat")
        print("• Use 'Summarize Documents' button")
        print("• Try 'Key Findings' analysis")
        print("• Browse document management")
        print("\nPress Ctrl+C to stop the server")
        print("="*50)

        # Start the server
        uvicorn.run(
            app,
            host="127.0.0.1",
            port=8000,
            log_level="info",
            access_log=True
        )

    except KeyboardInterrupt:
        print("\n\nDemo server stopped. Thanks for trying the demo!")
    except Exception as e:
        print(f"\nERROR starting demo server: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure all requirements are installed")
        print("2. Check that port 8000 is available")
        print("3. Try running: python main.py")

def show_demo_help():
    """Show demo mode help"""
    print("\nDemo Mode Help")
    print("="*50)
    print("\nWhat is Demo Mode?")
    print("Demo mode runs the RAG system without requiring:")
    print("• OpenAI API keys")
    print("• Google API keys")
    print("• Model downloads")
    print("• Internet connection (for AI features)")

    print("\nWhat's Simulated?")
    print("• AI responses (realistic but fake)")
    print("• Document processing")
    print("• Embedding generation")
    print("• Source citations")
    print("• System statistics")

    print("\nPerfect For:")
    print("• UI demonstrations")
    print("• Feature showcasing")
    print("• Development testing")
    print("• Offline presentations")

    print("\nQuick Start:")
    print("1. Run: python run_demo.py")
    print("2. Open: http://127.0.0.1:8000")
    print("3. Try uploading documents")
    print("4. Ask questions in chat")
    print("5. Explore admin features")

    print("\nSwitch to Production:")
    print("1. Set DEMO_MODE=false in .env")
    print("2. Add real API keys")
    print("3. Restart the application")

def main():
    """Main demo startup function"""
    print("RAG System Demo Mode")
    print("="*30)

    if len(sys.argv) > 1:
        if sys.argv[1] in ['--help', '-h', 'help']:
            show_demo_help()
            return
        elif sys.argv[1] in ['--check', '-c', 'check']:
            setup_demo_environment()
            requirements_met = check_demo_requirements()
            if requirements_met:
                print("\nDemo mode ready!")
            else:
                print("\nERROR: Please install missing requirements")
            return

    # Setup and start demo
    setup_demo_environment()

    if not check_demo_requirements():
        print("\nERROR: Demo requirements not met. Install missing packages and try again.")
        return

    start_demo_server()

if __name__ == "__main__":
    main()