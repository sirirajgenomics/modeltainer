#!/usr/bin/env bash

echo "====================================================="
echo " DEPRECATION WARNING: "
echo " The modeltainer image builder method is deprecated."
echo " We have switched to using Official Docker Images"
echo " configured via Custom Profile scripts."
echo ""
echo " Please use the new runner script:"
echo "   bash scripts/run_profile.sh profiles/<your-profile>.sh"
echo "====================================================="
exit 1

