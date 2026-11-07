#!/bin/bash
# usage: bg.sh LABEL LOGFILE CMD...   run a one-off job under launchd (survives the Claude Code session ending).
# The job's plist removes itself when the command ends, so it never re-runs at the next login.
name=$1; label=com.teleopfix.$1; log=$2; shift 2
plist=~/Library/LaunchAgents/$label.plist
mkdir -p ~/Developer/teleopfix/runs/jobs
job=~/Developer/teleopfix/runs/jobs/$name.sh
{ echo "#!/bin/bash"; echo "cd /Users/teo/Developer/teleopfix"; printf '%q ' "$@"; echo; echo "rc=\$?; echo \"exit \$rc \$(date '+%F %T')\"; rm -f $plist"; } > $job
chmod +x $job
cat > $plist <<P
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
<key>Label</key><string>$label</string>
<key>ProgramArguments</key><array><string>/bin/bash</string><string>$job</string></array>
<key>EnvironmentVariables</key><dict><key>PATH</key><string>/Users/teo/.local/bin:/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin</string><key>PYTHONPATH</key><string>/Users/teo/Developer/teleopfix</string><key>HOME</key><string>/Users/teo</string></dict>
<key>RunAtLoad</key><true/>
<key>AbandonProcessGroup</key><true/>
<key>StandardOutPath</key><string>$log</string><key>StandardErrorPath</key><string>$log</string>
</dict></plist>
P
launchctl bootout gui/$(id -u)/$label 2>/dev/null
launchctl bootstrap gui/$(id -u) $plist && echo "started $label"
