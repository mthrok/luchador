set -e

ROM_DIR=$1
if [ ! -d "$ROM_DIR" ] || [ ! "$(ls -A $ROM_DIR)" ]; then
    python setup.py download_ale --output-dir="$ROM_DIR"
fi
