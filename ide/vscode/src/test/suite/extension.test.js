const assert = require("assert");
const vscode = require("vscode");
const fs = require("fs");
const path = require("path");
const client = require("../../apiClient.js");

describe("TAKSH extension host", function () {
  const baseUrl = process.env.TAKSH_BASE_URL || "http://127.0.0.1:8000";
  const token = process.env.TAKSH_ENGINEERING_TOKEN || "";

  it("activates and registers commands", async function () {
    const extension = vscode.extensions.getExtension("taksh-local.taksh-engineering");
    assert.ok(extension, "extension is not installed");

    await extension.activate();
    assert.strictEqual(extension.isActive, true);

    const commands = await vscode.commands.getCommands(true);
    assert.ok(commands.includes("taksh.inspectProject"));
    assert.ok(commands.includes("taksh.createTask"));
    assert.ok(commands.includes("taksh.applyApprovedPatch"));
    assert.ok(commands.includes("taksh.setCredential"));
    assert.ok(commands.includes("taksh.clearCredential"));
  });

  it("does not declare taksh.apiKey as ordinary configuration", function () {
    const fs = require("fs");
    const path = require("path");
    const manifest = path.resolve(__dirname, "../../../package.json");
    const text = fs.readFileSync(manifest, "utf8");
    assert.ok(!text.includes("taksh.apiKey"));
  });

  it("round-trips a credential through SecretStorage", async function () {
    const command = vscode.commands.getCommands(true).then((list) => list.includes("taksh._testCredentialRoundTrip"));
    assert.strictEqual(await command, true);
    const result = await vscode.commands.executeCommand("taksh._testCredentialRoundTrip");
    assert.strictEqual(result.status, "passed");
    assert.strictEqual(result.recorded, false);
  });

  it("connects to the local service and inspects the workspace", async function () {
    const root = vscode.workspace.workspaceFolders && vscode.workspace.workspaceFolders[0]
      ? vscode.workspace.workspaceFolders[0].uri.fsPath
      : process.cwd();
    const inspected = await client.inspectProject(baseUrl, root, token);
    assert.ok(inspected.summary.total_files >= 0);
  });

  it("surfaces authentication failure", async function () {
    const root = vscode.workspace.workspaceFolders && vscode.workspace.workspaceFolders[0]
      ? vscode.workspace.workspaceFolders[0].uri.fsPath
      : process.cwd();
    await assert.rejects(
      () => client.inspectProject(baseUrl, root, "wrong-token"),
      /TAKSH request failed \(401\)/
    );
  });

  it("submits a task and displays plan state", async function () {
    const root = vscode.workspace.workspaceFolders && vscode.workspace.workspaceFolders[0]
      ? vscode.workspace.workspaceFolders[0].uri.fsPath
      : process.cwd();
    const taskId = "host-" + Date.now();
    const task = await client.createTask(baseUrl, { task_id: taskId, objective: "inspect project", root }, token);
    assert.strictEqual(task.task_id, taskId);
    const got = await client.getTask(baseUrl, taskId, token);
    assert.strictEqual(got.task_id, taskId);
  });

  it("requires explicit approval before applying a mutation", async function () {
    await assert.rejects(
      () => client.applyPatch(baseUrl, "host-task", { changes: [] }, false, token),
      /explicit approval/
    );
  });

  it("applies an approved mutation in a temporary workspace", async function () {
    const root = vscode.workspace.workspaceFolders && vscode.workspace.workspaceFolders[0]
      ? vscode.workspace.workspaceFolders[0].uri.fsPath
      : process.cwd();

    const taskId = "host-apply-" + Date.now();
    await client.createTask(baseUrl, { task_id: taskId, objective: "apply host change", root }, token);

    const plan = {
      task_id: taskId,
      changes: [
        { path: "host_probe.txt", operation: "create", content: "host-ok\n", reason: "host integration test" },
      ],
    };

    const applied = await client.applyPatch(baseUrl, taskId, plan, true, token);
    assert.strictEqual(applied.status, "APPLIED");

    const probe = path.join(root, "host_probe.txt");
    assert.strictEqual(fs.readFileSync(probe, "utf8"), "host-ok\n");
    fs.unlinkSync(probe);
  });
});
