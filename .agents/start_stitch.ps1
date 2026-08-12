$env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")
$env:CLOUDSDK_PYTHON = (& "C:\Users\ASUS\AppData\Local\Google\Cloud SDK\google-cloud-sdk\bin\gcloud.cmd" components copy-bundled-python)
$env:STITCH_ACCESS_TOKEN = (& "C:\Users\ASUS\AppData\Local\Google\Cloud SDK\google-cloud-sdk\bin\gcloud.cmd" auth print-access-token)
npx -y @_davideast/stitch-mcp proxy
