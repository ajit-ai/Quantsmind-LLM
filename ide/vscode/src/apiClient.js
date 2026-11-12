"use strict";

async function request(baseUrl, path, options = {}) {
  const url = baseUrl.replace(/\/$/, "") + path;
  const headers = { "Content-Type": "application/json", ...(options.headers || {}) };
  if (options.apiKey) headers.Authorization = "Bearer " + options.apiKey;
  const response = await fetch(url, {
    method: options.method || "GET",
    headers,
    body: options.body ? JSON.stringify(options.body) : undefined,
  });
  const text = await response.text();
  let data = null;
  try { data = text ? JSON.parse(text) : null; } catch (error) { data = { raw: text }; }
  if (!response.ok) {
    const message = (data && data.detail) || (data && data.error && data.error.message) || response.statusText;
    throw new Error("TAKSH request failed (" + response.status + "): " + message);
  }
  return data;
}

async function health(baseUrl, apiKey) {
  return request(baseUrl, "/v1/health", { apiKey });
}

async function inspectProject(baseUrl, root, apiKey) {
  return request(baseUrl, "/v1/projects/inspect?root=" + encodeURIComponent(root), { apiKey });
}

async function createTask(baseUrl, task, apiKey) {
  return request(baseUrl, "/v1/tasks", { method: "POST", apiKey, body: task });
}

async function getTask(baseUrl, taskId, apiKey) {
  return request(baseUrl, "/v1/tasks/" + encodeURIComponent(taskId), { apiKey });
}

async function validateProject(baseUrl, taskId, files, apiKey) {
  return request(baseUrl, "/v1/tasks/" + encodeURIComponent(taskId) + "/validate", {
    method: "POST",
    apiKey,
    body: { files: files || null },
  });
}

async function runTests(baseUrl, taskId, files, timeout, apiKey) {
  return request(baseUrl, "/v1/tasks/" + encodeURIComponent(taskId) + "/tests", {
    method: "POST",
    apiKey,
    body: { files: files || null, timeout: timeout || 120 },
  });
}

async function applyPatch(baseUrl, taskId, plan, approved, apiKey) {
  if (!approved) throw new Error("Patch application requires explicit approval");
  return request(baseUrl, "/v1/tasks/" + encodeURIComponent(taskId) + "/apply", {
    method: "POST",
    apiKey,
    body: { plan, approved: true },
  });
}

module.exports = {
  request,
  health,
  inspectProject,
  createTask,
  getTask,
  validateProject,
  runTests,
  applyPatch,
};
