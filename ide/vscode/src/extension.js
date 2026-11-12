"use strict";

const vscode = require("vscode");
const client = require("./apiClient");

function config() {
  const cfg = vscode.workspace.getConfiguration("taksh");
  return {
    baseUrl: cfg.get("baseUrl") || "http://127.0.0.1:8000",
    apiKey: cfg.get("apiKey") || "",
  };
}

async function inspectProjectCommand() {
  const root = vscode.workspace.workspaceFolders && vscode.workspace.workspaceFolders[0]
    ? vscode.workspace.workspaceFolders[0].uri.fsPath
    : undefined;
  if (!root) {
    vscode.window.showErrorMessage("TAKSH: open a workspace folder first.");
    return;
  }
  try {
    const cfg = config();
    const result = await client.inspectProject(cfg.baseUrl, root, cfg.apiKey);
    const channel = vscode.window.createOutputChannel("TAKSH");
    channel.clear();
    channel.appendLine(JSON.stringify(result, null, 2));
    channel.show();
  } catch (error) {
    vscode.window.showErrorMessage("TAKSH inspect failed: " + error.message);
  }
}

async function createTaskCommand() {
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
    const cfg = config();
    const result = await client.createTask(cfg.baseUrl, { task_id: taskId, objective, root }, cfg.apiKey);
    vscode.window.showInformationMessage("TAKSH task created: " + result.task_id + " (" + result.state + ")");
  } catch (error) {
    vscode.window.showErrorMessage("TAKSH task creation failed: " + error.message);
  }
}

async function applyApprovedPatchCommand() {
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
    const cfg = config();
    const result = await client.applyPatch(cfg.baseUrl, taskId, plan, true, cfg.apiKey);
    vscode.window.showInformationMessage("TAKSH apply result: " + result.status);
  } catch (error) {
    vscode.window.showErrorMessage("TAKSH apply failed: " + error.message);
  }
}

function activate(context) {
  context.subscriptions.push(
    vscode.commands.registerCommand("taksh.inspectProject", inspectProjectCommand),
    vscode.commands.registerCommand("taksh.createTask", createTaskCommand),
    vscode.commands.registerCommand("taksh.applyApprovedPatch", applyApprovedPatchCommand)
  );
}

function deactivate() {}

module.exports = { activate, deactivate };
