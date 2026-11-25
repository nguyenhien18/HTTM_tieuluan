"""Stop only project-owned demo and LLM processes, verifying their command lines."""
from pathlib import Path
import subprocess, sys
if hasattr(sys.stdout,'reconfigure'):sys.stdout.reconfigure(encoding='utf-8',errors='replace')
root=Path(__file__).resolve().parent
# Verify the command belongs to this exact project before stopping its PID.
script=r"""
$demoRoot=$args[0]
$demoPid=[int]$args[1]
$expected=$args[2]
$demoProcess=Get-CimInstance Win32_Process -Filter ("ProcessId = " + $demoPid)
if ($null -eq $demoProcess) { Write-Output 'Demo is already stopped.'; exit 0 }
if (($null -eq $demoProcess.CommandLine) -or (-not $demoProcess.CommandLine.Contains($demoRoot))) { Write-Error 'PID does not belong to this demo folder.'; exit 1 }
if ($demoProcess.CommandLine -notmatch $expected) { Write-Error 'Unexpected project process.'; exit 1 }
Stop-Process -Id $demoPid
Write-Output 'HTTM Demo Studio stopped.'
"""
# Root is passed as a PowerShell literal; double apostrophes escape its only quote delimiter.
failed=False
for name,pattern in [('llm.pid',r'llama-server\.exe'),('demo.pid',r'(app|launch)\.py')]:
    pidfile=root/name
    if not pidfile.is_file():continue
    try:pid=int(pidfile.read_text().strip())
    except ValueError:print('Invalid PID file: '+name);failed=True;continue
    command="& { "+script+" } '"+str(root).replace("'","''")+"' "+str(pid)+" '"+pattern+"'"
    r=subprocess.run(['powershell','-NoProfile','-Command',command])
    if r.returncode==0:pidfile.unlink(missing_ok=True)
    else:failed=True
sys.exit(1 if failed else 0)
