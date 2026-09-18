import os
import sys
import subprocess

def main():
    print("=" * 60)
    print("🩺 Starting Health Assistant AI Application...")
    print("=" * 60)

    # Use the current virtual environment python if available
    python_exe = sys.executable
    app_file = os.path.join(os.path.dirname(__file__), "app.py")

    cmd = [python_exe, "-m", "streamlit", "run", app_file]

    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\n👋 Health Assistant AI stopped gracefully.")
    except Exception as exc:
        print(f"\n❌ Error launching application: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
