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

## Build and test
```bash
cd ide/vscode
node ./test/apiClient.test.js
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
