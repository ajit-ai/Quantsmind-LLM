const assert = require("assert");
const credentials = require("../src/credentials.js");

class MockSecrets {
  constructor() { this.values = new Map(); }
  async get(key) { return this.values.get(key); }
  async store(key, value) { this.values.set(key, value); }
  async delete(key) { this.values.delete(key); }
}

async function main() {
  // write/read/delete
  const secrets = new MockSecrets();
  const context = { secrets };
  const token = "abcdefghijklmnopqrstuvwxyz0123456789";
  assert.strictEqual(await credentials.writeToken(context, token), true);
  assert.strictEqual(await credentials.readToken(context), token);
  assert.strictEqual(await credentials.deleteToken(context), true);
  assert.strictEqual(await credentials.readToken(context), undefined);

  // replacement
  await credentials.writeToken(context, token);
  const replacement = "ABCDEFGHIJKLMNOPQRSTUVWXYZ9876543210";
  await credentials.writeToken(context, replacement);
  assert.strictEqual(await credentials.readToken(context), replacement);

  // missing and invalid
  await credentials.deleteToken(context);
  assert.strictEqual(await credentials.readToken(context), undefined);
  await assert.rejects(() => credentials.writeToken(context, "short"), /at least 16/);

  // existing SecretStorage precedence over legacy
  await credentials.writeToken(context, replacement);
  const vscodeMock = {
    workspace: { getConfiguration: () => ({ get: () => undefined, inspect: () => ({}), update: async () => undefined }) },
    ConfigurationTarget: { Global: 1 },
  };
  const result = await credentials.migrateLegacyCredential(context, vscodeMock);
  assert.strictEqual(result.status, "not_present");
  assert.strictEqual(await credentials.readToken(context), replacement);

  // legacy migration succeeds and clears the legacy setting
  const legacy = "legacy-token-ABCDEF1234567890";
  let updated = false;
  const legacyVscode = {
    ConfigurationTarget: { Global: 1 },
    workspace: {
      getConfiguration: () => ({
        get: () => legacy,
        inspect: () => ({ globalValue: legacy }),
        update: async (key, value) => { updated = key === "apiKey" && value === undefined; },
      }),
    },
  };
  await credentials.deleteToken(context);
  const migration = await credentials.migrateLegacyCredential(context, legacyVscode);
  assert.strictEqual(migration.status, "migrated");
  assert.strictEqual(await credentials.readToken(context), legacy);
  assert.strictEqual(updated, true);

  // existing SecretStorage value is not overwritten by legacy migration
  await credentials.writeToken(context, replacement);
  const migration2 = await credentials.migrateLegacyCredential(context, legacyVscode);
  assert.strictEqual(migration2.status, "migrated");
  assert.strictEqual(await credentials.readToken(context), replacement);

  console.log("Credential unit tests passed");
}

main().catch((error) => { console.error(error); process.exit(1); });
