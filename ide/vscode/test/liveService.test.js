const assert = require("assert");
const fs = require("fs");
const os = require("os");
const path = require("path");
const client = require("../src/apiClient.js");

const BASE = process.env.TAKSH_BASE_URL || "http://127.0.0.1:8000";
const KEY = process.env.TAKSH_ENGINEERING_TOKEN || "";

async function main() {
  // health is public
  const health = await client.health(BASE, KEY);
  assert.strictEqual(health.status, "ok");

  // create a temporary Python project
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "taksh-live-"));
  fs.mkdirSync(path.join(root, "pkg"));
  fs.writeFileSync(path.join(root, "pkg", "math.py"), "def add(a, b):\n    return a + b\n");
  fs.mkdirSync(path.join(root, "tests"));
  fs.writeFileSync(path.join(root, "tests", "test_math.py"), "from pkg.math import add\n\ndef test_add():\n    assert add(1, 2) == 3\n");

  // inspect project
  const inspected = await client.inspectProject(BASE, root, KEY);
  assert.strictEqual(inspected.summary.python_files, 2);

  // create task
  const taskId = "live-" + Date.now();
  const task = await client.createTask(BASE, { task_id: taskId, objective: "adjust add", root }, KEY);
  assert.strictEqual(task.task_id, taskId);

  // get task
  const got = await client.getTask(BASE, taskId, KEY);
  assert.strictEqual(got.task_id, taskId);

  // validate project
  const validation = await client.validateProject(BASE, taskId, null, KEY);
  assert.strictEqual(validation.status, "PASS");

  // run tests
  const tests = await client.runTests(BASE, taskId, null, 60, KEY);
  assert.strictEqual(tests.status, "PASS");

  // apply an approved patch
  const plan = {
    task_id: taskId,
    changes: [
      { path: "pkg/math.py", operation: "modify", content: "def add(a, b):\n    return a + b + 0\n", reason: "live integration" },
    ],
  };
  const applied = await client.applyPatch(BASE, taskId, plan, true, KEY);
  assert.strictEqual(applied.status, "APPLIED");
  const content = fs.readFileSync(path.join(root, "pkg", "math.py"), "utf8");
  assert.ok(content.includes("a + b + 0"));

  // cleanup
  fs.rmSync(root, { recursive: true, force: true });
  console.log("Live service integration test passed");
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
