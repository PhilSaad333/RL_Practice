param(
  [Parameter(Mandatory = $true)]
  [string] $InstanceIP,

  [string] $KeyPath = "$env:USERPROFILE\.ssh\lambda_new",

  # local port on your Windows machine
  [int] $LocalPort = 16006,

  # remote port where TensorBoard listens on the instance
  [int] $RemotePort = 16006,

  [string] $User = "ubuntu"
)

# Defensive checks
if (-not (Test-Path $KeyPath)) {
  Write-Error "Key file not found: $KeyPath"
  exit 1
}

# Ensure OpenSSH client is present (Windows 10/11 include it)
$ssh = Get-Command ssh -ErrorAction SilentlyContinue
if (-not $ssh) {
  Write-Error "OpenSSH client 'ssh' not found. Install 'OpenSSH Client' via Windows Features."
  exit 1
}

Write-Host "Opening SSH tunnel: localhost:$LocalPort  <->  $InstanceIP:127.0.0.1:$RemotePort"
Write-Host "Press Ctrl+C to stop port forwarding."

# -o IdentitiesOnly=yes forces using the provided key
ssh -o IdentitiesOnly=yes `
    -i "$KeyPath" `
    -N -L "$LocalPort:127.0.0.1:$RemotePort" `
    "$User@$InstanceIP"
