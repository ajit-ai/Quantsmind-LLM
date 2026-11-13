"use strict";

const vscode = require("vscode");
const client = require("./apiClient");
const credentials = require("./credentials");

const OUTPUT_CHANNEL = "TAKSH";

function createOutputChannel() {
  return vscode.window.createOutputChannel(OUTPUT_CHANNEL);
}

async function serviceConfig(context) {
  const cfg = vscode.workspace.getConfiguration("taksh");
  const baseUrl = cfg.get("baseUrl") || "http://127.0.0.1:8000";
  const token = await credentials.readToken(context);
  return { baseUrl, apiKey: token || "" };
}

async function inspectProjectCommand(context) {
  const root = vscode.workspace.workspaceFolders && vscode.workspace.workspaceFolders[0]
    ? vscode.workspace.workspaceFolders[0].uri.fsPath
    : undefined;
  if (!root) {
    vscode.window.showErrorMessage("TAKSH: open a workspace folder first.");
    return;
  }
  try {
    const cfg = await serviceConfig(context);
    await credentials.migrateLegacyCredential(context);
    const result = await client.inspectProject(cfg.baseUrl, root, cfg.apiKey);
    const channel = createOutputChannel();
    context.subscriptions.push(channel);
    channel.clear();
    channel.appendLine(JSON.stringify(result, null, 2));
    channel.show();
  } catch (error) {
    vscode.window.showErrorMessage("TAKSH inspect failed: " + safeMessage(error));
  }
}

async function createTaskCommand(context) {
  const root = vscode.workspace.workspaceFolders && vscode.workspace.workspaceFolders[0]
    ? vscode.workspace.workspaceFolders[0].uri.fsPath
    : undefined;
  if (!root) {
    vscode.window.showErrorMessage("TAKSH: open a workspace folder first.");
    return;
  }
  const objective = await vscode.window.showInputBox({ prompt: "Engineering objective" });
  if (!objective) return;
  const taskId = "task-" + Date.now();
  try {
    const cfg = await serviceConfig(context);
    const result = await client.createTask(cfg.baseUrl, { task_id: taskId, objective, root }, cfg.apiKey);
    vscode.window.showInformationMessage("TAKSH task created: " + result.task_id + " (" + result.state + ")");
  } catch (error) {
    vscode.window.showErrorMessage("TAKSH task creation failed: " + safeMessage(error));
  }
}

async function applyApprovedPatchCommand(context) {
  const taskId = await vscode.window.showInputBox({ prompt: "Task ID" });
  if (!taskId) return;
  const planText = await vscode.window.showInputBox({ prompt: "Patch plan JSON" });
  if (!planText) return;
  let plan;
  try { plan = JSON.parse(planText); } catch (error) {
    vscode.window.showErrorMessage("Invalid patch JSON");
    return;
  }
  const approved = await vscode.window.showQuickPick(["No", "Yes"], { placeHolder: "Approve patch application?" });
  if (approved !== "Yes") {
    vscode.window.showInformationMessage("Patch application cancelled.");
    return;
  }
  try {
    const cfg = await serviceConfig(context);
    const result = await client.applyPatch(cfg.baseUrl, taskId, plan, true, cfg.apiKey);
    vscode.window.showInformationMessage("TAKSH apply result: " + result.status);
  } catch (error) {
    vscode.window.showErrorMessage("TAKSH apply failed: " + safeMessage(error));
  }
}

function safeMessage(error) {
  const message = error && error.message ? String(error.message) : "unknown error";
  return message.replace(/[A-Za-z0-9._\-]{24,}/g, "<redacted>");
}

async function setCredentialCommand(context) {
  const token = await vscode.window.showInputBox({
    prompt: "TAKSH service bearer token",
    password: true,
    ignoreFocusOut: true,
  });
  if (!token) return;
  try {
    await credentials.writeToken(context, token);
    vscode.window.showInformationMessage("TAKSH credential stored in secret storage.");
  } catch (error) {
    vscode.window.showErrorMessage(safeMessage(error));
  }
}

async function clearCredentialCommand(context) {
  try {
    await credentials.deleteToken(context);
    vscode.window.showInformationMessage("TAKSH credential cleared from secret storage.");
  } catch (error) {
    vscode.window.showErrorMessage(safeMessage(error));
  }
}

async function activate(context) {
  const migration = await credentials.migrateLegacyCredential(context, vscode);
  if (migration.status === "manual_removal_required") {
    vscode.window.showWarningMessage(
      "TAKSH legacy taksh.apiKey found in workspace settings; remove it manually and store the credential in SecretStorage instead."
    );
  }

  context.subscriptions.push(
    vscode.commands.registerCommand("taksh.inspectProject", () => inspectProjectCommand(context)),
    vscode.commands.registerCommand("taksh.createTask", () => createTaskCommand(context)),
    vscode.commands.registerCommand("taksh.applyApprovedPatch", () => applyApprovedPatchCommand(context)),
    vscode.commands.registerCommand("taksh.setCredential", () => setCredentialCommand(context)),
    vscode.commands.registerCommand("taksh.clearCredential", () => clearCredentialCommand(context))
  );

  if (process.env.TAKSH_TEST_HOOKS === "1") {
    context.subscriptions.push(
      vscode.commands.registerCommand("taksh._testCredentialRoundTrip", () => credentials.testRoundTrip(context))
    );
  }
}

function deactivate() {}

module.exports = { activate, deactivate };
