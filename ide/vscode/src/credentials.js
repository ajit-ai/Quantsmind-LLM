"use strict";

const SECRET_KEY = "taksh.apiToken.v1";
const LEGACY_KEY = "apiKey";
const LEGACY_SCOPE = ["Global", "Workspace"];

function secretStorageAvailable(secrets) {
  return !!secrets && typeof secrets.get === "function" && typeof secrets.store === "function" && typeof secrets.delete === "function";
}

async function readToken(context) {
  if (!secretStorageAvailable(context.secrets)) {
    throw new Error("TAKSH credentials are unavailable: secret storage is not available");
  }
  return context.secrets.get(SECRET_KEY);
}

async function writeToken(context, token) {
  if (typeof token !== "string" || token.length < 16 || /\s/.test(token)) {
    throw new Error("TAKSH credential must be at least 16 non-whitespace characters");
  }
  await context.secrets.store(SECRET_KEY, token);
  return true;
}

async function deleteToken(context) {
  if (!secretStorageAvailable(context.secrets)) return false;
  await context.secrets.delete(SECRET_KEY);
  return true;
}

async function migrateLegacyCredential(context, vscode) {
  const configuration = (vscode || require("vscode")).workspace.getConfiguration("taksh");
  const inspection = configuration.inspect ? configuration.inspect(LEGACY_KEY) : {};
  const globalLegacy = typeof inspection.globalValue === "string" && inspection.globalValue.length > 0
    ? inspection.globalValue
    : typeof inspection.global === "string" && inspection.global.length > 0
      ? inspection.global
      : null;
  const workspaceLegacy = typeof inspection.workspaceValue === "string" && inspection.workspaceValue.length > 0
    ? inspection.workspaceValue
    : typeof inspection.workspace === "string" && inspection.workspace.length > 0
      ? inspection.workspace
      : null;

  if (!globalLegacy && !workspaceLegacy) return { status: "not_present" };

  const existing = await context.secrets.get(SECRET_KEY);
  const chosen = globalLegacy || workspaceLegacy;
  if (!existing) {
    await context.secrets.store(SECRET_KEY, chosen);
    const restored = await context.secrets.get(SECRET_KEY);
    if (restored !== chosen) {
      return { status: "migration_failed", target: globalLegacy ? "Global" : "Workspace", reason: "secret verification failed" };
    }
  }

  if (globalLegacy) {
    try {
      await configuration.update(LEGACY_KEY, undefined, (vscode || require("vscode")).ConfigurationTarget.Global);
    } catch (error) {
      return { status: "migration_failed", target: "Global", reason: "could not clear legacy global setting" };
    }
  }

  if (workspaceLegacy) {
    return {
      status: "manual_removal_required",
      target: "Workspace",
      reason: "legacy credential detected in workspace settings; remove taksh.apiKey manually and never commit it"
    };
  }

  return { status: globalLegacy ? "migrated" : "not_present" };
}

async function testRoundTrip(context) {
  if (!secretStorageAvailable(context.secrets)) return { status: "unavailable", recorded: false };
  const nonce = "host-test-nonce-" + Date.now().toString().padStart(16, "0");
  await context.secrets.store(SECRET_KEY, nonce);
  const restored = await context.secrets.get(SECRET_KEY);
  await context.secrets.delete(SECRET_KEY);
  return { status: restored === nonce ? "passed" : "failed", recorded: false };
}

module.exports = { SECRET_KEY, LEGACY_KEY, readToken, writeToken, deleteToken, migrateLegacyCredential, testRoundTrip, secretStorageAvailable };
