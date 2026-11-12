const http = require("http");
const assert = require("assert");
const client = require("../src/apiClient.js");

function startServer(handler) {
  return new Promise((resolve) => {
    const server = http.createServer(handler);
    server.listen(0, "127.0.0.1", () => resolve(server));
  });
}

async function run() {
  // health
  let server = await startServer((req, res) => {
    res.setHeader("Content-Type", "application/json");
    res.end(JSON.stringify({ status: "ok" }));
  });
  const addr = server.address();
  const base = "http://127.0.0.1:" + addr.port;
  const health = await client.health(base);
  assert.strictEqual(health.status, "ok");
  server.close();

  // inspect project
  server = await startServer((req, res) => {
    res.setHeader("Content-Type", "application/json");
    res.end(JSON.stringify({ summary: { total_files: 2 } }));
  });
  const addr2 = server.address();
  const base2 = "http://127.0.0.1:" + addr2.port;
  const inspected = await client.inspectProject(base2, "/tmp/project");
  assert.strictEqual(inspected.summary.total_files, 2);
  server.close();

  // apply requires approval
  await assert.rejects(
    () => client.applyPatch(base2, "task-1", { changes: [] }, false),
    /explicit approval/
  );

  // unauthorized response is surfaced
  server = await startServer((req, res) => {
    res.statusCode = 401;
    res.setHeader("Content-Type", "application/json");
    res.end(JSON.stringify({ detail: "missing token" }));
  });
  const addr3 = server.address();
  const base3 = "http://127.0.0.1:" + addr3.port;
  await assert.rejects(() => client.health(base3), /TAKSH request failed \(401\)/);
  server.close();

  console.log("IDE client tests passed");
}

run().catch((error) => {
  console.error(error);
  process.exit(1);
});
