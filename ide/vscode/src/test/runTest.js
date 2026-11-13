const { spawn } = require("child_process");
const http = require("http");
const os = require("os");
const fs = require("fs");
const path = require("path");
const { runTests } = require("@vscode/test-electron");

const PYTHON = process.env.TAKSH_PYTHON || "C:\\Users\\Lenovo\\AppData\\Local\\Programs\\Python\\Python314\\python.exe";
const REPO_ROOT = path.resolve(__dirname, "../../../..");

function waitForHealth(baseUrl, attempts = 60) {
  return new Promise((resolve, reject) => {
    let count = 0;
    const attempt = () => {
      http.get(baseUrl + "/v1/health", (res) => {
        if (res.statusCode === 200) return resolve();
        if (count >= attempts) return reject(new Error("service did not become healthy"));
        count += 1;
        setTimeout(attempt, 500);
      }).on("error", () => {
        if (count >= attempts) return reject(new Error("service did not become healthy"));
        count += 1;
        setTimeout(attempt, 500);
      });
    };
    attempt();
  });
}

async function main() {
  // temporary python fixture workspace
  const fixture = fs.mkdtempSync(path.join(os.tmpdir(), "taksh-host-"));
  fs.mkdirSync(path.join(fixture, "pkg"));
  fs.writeFileSync(path.join(fixture, "pkg", "math.py"), "def add(a, b):\n    return a + b\n");
  fs.writeFileSync(path.join(fixture, "README.md"), "host fixture\n");

  const env = {
    ...process.env,
    TAKSH_ENGINEERING_TOKEN: "host-test-token",
    TAKSH_TOKEN_SCOPES: "read,execute,mutate,memory,audit",
    TAKSH_BASE_URL: "http://127.0.0.1:8000",
  };

  const service = spawn(
    PYTHON,
    ["-m", "uvicorn", "serving.engineering_app:app", "--host", "127.0.0.1", "--port", "8000"],
    { cwd: REPO_ROOT, env, stdio: "inherit" }
  );

  try {
    await waitForHealth("http://127.0.0.1:8000");
    await runTests({
      extensionDevelopmentPath: path.resolve(__dirname, "../../"),
      extensionTestsPath: path.resolve(__dirname, "suite/index.js"),
      launchArgs: [fixture],
      extensionTestsEnv: {
        TAKSH_BASE_URL: "http://127.0.0.1:8000",
        TAKSH_ENGINEERING_TOKEN: "host-test-token",
      },
    });
  } finally {
    service.kill();
    fs.rmSync(fixture, { recursive: true, force: true });
  }
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
