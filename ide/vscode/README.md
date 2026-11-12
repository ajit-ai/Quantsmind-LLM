# TAKSH Engineering — VS Code Extension (local)

**Status:** Local client extension source. Not published to the VS Code Marketplace.

## What it does
- Opens the current workspace and inspects it via the TAKSH engineering service.
- Creates engineering tasks against the local REST service.
- Shows the proposed patch plan and applies it only after explicit user approval.
- Reports service errors and operation status in the `TAKSH` output channel.

## Requirements
- VS Code >= 1.85
- Running local TAKSH engineering service on `http://127.0.0.1:8000`
- `taksh.apiKey` configured if the service requires bearer authentication

## Live service test
```bash
# terminal 1
$env:TAKSH_ENGINEERING_TOKEN='dev-token'
$env:TAKSH_TOKEN_SCOPES='read,execute,mutate,memory,audit'
python -m uvicorn serving.engineering_app:app --host 127.0.0.1 --port 8000

# terminal 2
$env:TAKSH_BASE_URL='http://127.0.0.1:8000'
$env:TAKSH_ENGINEERING_TOKEN='dev-token'
cd ide/vscode
node ./test/liveService.test.js
```

## Install locally
1. Open this folder in VS Code.
2. Press `F5` to launch an Extension Development Host.
3. Open a Python project folder.
4. Run `TAKSH: Inspect Project`, `TAKSH: Create Engineering Task`, or `TAKSH: Apply Approved Patch`.

## Limitations
- No Marketplace publication.
- No autonomous capability claim; the extension is a client.
- Patch application requires explicit approval and a configured `mutate` scope.
